"""
Standalone test/evaluation script for ContourPose.

Output files are saved to: results/{class_type}_{YYMMDD_HHMM}/
    {sensor}_bop.csv          - BOP-format pose results
    {sensor}_detailed.csv     - Per-instance detailed metrics

Usage:
    # BOP format (all sensors, default output dir)
    python test.py \
        --bop_root data/ROBI_BOP/Gear_BOP \
        --class_type gear \
        --checkpoint_path model/gear/best_model.pkl

    # BOP format (single sensor)
    python test.py \
        --bop_root data/ROBI_BOP/Gear_BOP \
        --class_type gear \
        --sensor ensenso_left \
        --checkpoint_path model/gear/best_model.pkl

    # Legacy ContourPose format
    python test.py \
        --class_type obj1 \
        --scene 13 --index 2 \
        --checkpoint_path model/obj1/150.pkl
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

import gin
import torch
import numpy as np

from network import ContourPose
from eval import evaluator
from dataset.data_utils import load_keypoints, create_test_loader


def create_output_dir(args):
    """
    Create and return the output directory path.

    Default: results/{class_type}_{wandb_run_name}_{pecp|no_pecp}/
    Falls back to results/{class_type}_{YYMMDD_HHMM}/ when no run name is available.
    """
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_label = None
        ckpt = getattr(args, 'checkpoint_path', None)
        if ckpt:
            candidate = Path(ckpt).parent.name
            # Use the subdir name only if it looks like a wandb run name
            # (not the class dir itself, not a bare epoch number like "150")
            if candidate and candidate != args.class_type and not candidate.isdigit():
                run_label = candidate

        if run_label:
            pecp_label = "pecp" if getattr(args, 'use_pecp', True) else "no_pecp"
            dir_name = f"{args.class_type}_{run_label}_{pecp_label}"
        else:
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            dir_name = f"{args.class_type}_{timestamp}"

        output_dir = Path("results") / dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Test] Output directory: {output_dir}")
    return output_dir


def _sanitize_best_pose_error(value):
    """Return None when value is a sentinel (inf, nan, ≥1e8) so metadata.json stays clean."""
    if value is None:
        return None
    try:
        if not (value < 1e8):  # catches inf, nan, and the 1e9 sentinel
            return None
    except TypeError:
        return None
    return value


def save_run_metadata(output_dir, args, ckpt_info, wandb_run_name=None):
    """Write run provenance to metadata.json so analysis notebooks can auto-populate paths."""
    import json
    meta = {
        "class_type": args.class_type,
        "checkpoint_path": getattr(args, 'checkpoint_path', None),
        "bop_root": getattr(args, 'bop_root', None),
        "train_mask_dir": getattr(args, 'train_mask_dir', None),
        "use_pecp": getattr(args, 'use_pecp', True),
        "epoch": ckpt_info.get("epoch") if ckpt_info else None,
        "best_pose_error": _sanitize_best_pose_error(
            ckpt_info.get("best_pose_error") if ckpt_info else None
        ),
        "wandb_run_name": wandb_run_name,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Test] Saved run metadata: {meta_path}")


def get_csv_path(output_dir, sensor=None, masked=None):
    """Get the BOP CSV path for a given sensor and mask variant."""
    parts = []
    if sensor:
        parts.append(sensor)
    if masked is not None:
        parts.append("masked" if masked else "rgb")
    parts.append("bop")
    name = "_".join(parts) + ".csv"
    return str(output_dir / name)


def discover_sensors(bop_root):
    """
    Discover available sensors by scanning for scene_gt_{sensor}.json files.

    Returns:
        List of sensor names found across all scene directories.
    """
    bop_path = Path(bop_root)
    exclude_prefixes = {"keypoints", "info", "debug", "test", "backup"}
    sensors = set()

    # Scan scene directories for scene_gt_{sensor}.json files
    for scene_dir in sorted(bop_path.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.isdigit():
            continue
        for f in scene_dir.iterdir():
            m = re.match(r"scene_gt_(.+)\.json$", f.name)
            if m:
                sensor = m.group(1)
                # Skip sensors whose name starts with an excluded prefix
                # (e.g., "keypoints_ensenso_left" starts with "keypoints")
                if any(sensor.lower().startswith(p) for p in exclude_prefixes):
                    continue
                # Verify matching camera file exists
                cam_file = scene_dir / f"scene_camera_{sensor}.json"
                if cam_file.exists():
                    sensors.add(sensor)

    return sorted(sensors)


def build_model(args, num_keypoints, model_cls=ContourPose):
    """Build model and load checkpoint (single GPU)."""
    print(f"[Model] Using {model_cls.__name__}")
    model = model_cls(
        heatmap_dim=num_keypoints,
        data_root=args.data_root,
        class_type=args.class_type,
    )
    model = model.cuda() if torch.cuda.is_available() else model

    # Load checkpoint
    ckpt_info = model.load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        load_best=args.load_best,
        strict=True,
    )
    if ckpt_info is None:
        raise RuntimeError(f"No checkpoint found at {args.checkpoint_path}")

    print(f"[Test] Loaded checkpoint (epoch {ckpt_info['epoch']})")
    return model, ckpt_info


def run_eval(args, model, device):
    """Create test loader, run evaluation, return metrics."""
    test_result = create_test_loader(args)
    if isinstance(test_result, tuple):
        test_loader, test_samples = test_result
    else:
        test_loader, test_samples = test_result, None

    eval_instance = evaluator(args, model, test_loader, device)
    metrics = eval_instance.evaluate()
    # eval.py's evaluate() prints results but returns None; normalise to empty dict
    return metrics if metrics is not None else {}


def log_metrics_to_wandb(wandb, metrics, sensor=None):
    """Log evaluation metrics to wandb."""
    if not metrics:
        return
    prefix = f"test/{sensor}" if sensor else "test"
    wandb.log({
        f"{prefix}/2d_projection": metrics["proj_2d"],
        f"{prefix}/add": metrics["add"],
        f"{prefix}/x_error_mm": metrics["x_error"],
        f"{prefix}/y_error_mm": metrics["y_error"],
        f"{prefix}/z_error_mm": metrics["z_error"],
        f"{prefix}/translation_error_mm": metrics["translation_error"],
        f"{prefix}/alpha_error_deg": metrics["alpha_error"],
        f"{prefix}/beta_error_deg": metrics["beta_error"],
        f"{prefix}/gamma_error_deg": metrics["gamma_error"],
        f"{prefix}/rotation_error_deg": metrics["rotation_error"],
        f"{prefix}/pck_1px": metrics["pck_1px"],
        f"{prefix}/pck_2px": metrics["pck_2px"],
    })


def log_csv_to_wandb(wandb, csv_path):
    """Upload a CSV file and its detailed counterpart to wandb."""
    if csv_path and os.path.exists(csv_path):
        wandb.save(csv_path, policy="now")
        print(f"[wandb] Uploaded {csv_path}")
        # Also upload the detailed CSV if it exists
        base = csv_path.rsplit(".", 1)[0] if "." in csv_path else csv_path
        detailed_path = f"{base.replace('_bop', '_detailed')}.csv"
        if os.path.exists(detailed_path):
            wandb.save(detailed_path, policy="now")
            print(f"[wandb] Uploaded {detailed_path}")


@gin.configurable
def main(args, model_cls=ContourPose, batch_size=8, threshold=5,
         compute_edge_input=False, **kwargs):
    # Expose gin params on args for downstream code
    args.batch_size = batch_size
    args.threshold = threshold
    args.compute_edge_input = compute_edge_input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")

    # Create output directory
    output_dir = create_output_dir(args)
    args.output_dir = str(output_dir)

    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            tags=["eval", args.class_type, model_cls.__name__] + (["pecp"] if args.use_pecp else []) + [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
            config={
                "class_type": args.class_type,
                "checkpoint_path": args.checkpoint_path,
                "sensor": args.sensor,
                "bop_root": args.bop_root,
                "obj_id": args.obj_id,
                "threshold": threshold,
                "batch_size": batch_size,
                "output_dir": str(output_dir),
                "gin_config": gin.operative_config_str(),
            },
        )

    # Load keypoints
    corners = load_keypoints(args)

    # Build model
    model, ckpt_info = build_model(args, num_keypoints=corners.shape[0], model_cls=model_cls)

    # SpectraPose: disable input scaling (transformer branch is scale-sensitive)
    if model_cls is not ContourPose:
        if args.fixed_scale is not None or getattr(args, 'train_mask_dir', None) is not None:
            print(f"[Test] WARNING: {model_cls.__name__} detected — disabling input scaling "
                  f"(fixed_scale={args.fixed_scale}, train_mask_dir={getattr(args, 'train_mask_dir', None)})")
            args.fixed_scale = None
            args.train_mask_dir = None
            print(f"[Test] Images will be masked but NOT scaled for {model_cls.__name__}")

    # Save run provenance for notebook analysis
    _wandb_run_name = wandb.run.name if args.use_wandb else None
    save_run_metadata(output_dir, args, ckpt_info, wandb_run_name=_wandb_run_name)

    # Log checkpoint and gin config to wandb
    if args.use_wandb:
        ckpt_path = args.checkpoint_path
        if ckpt_path and os.path.exists(ckpt_path):
            wandb.save(ckpt_path, policy="now")
            print(f"[wandb] Uploaded checkpoint: {ckpt_path}")
        wandb.run.summary["checkpoint_epoch"] = ckpt_info.get("epoch", 0)
        wandb.run.summary["checkpoint_best_pose_error"] = ckpt_info.get("best_pose_error", float("inf"))
        wandb.run.summary["gin_operative_config"] = gin.operative_config_str()

    # Determine mask variants to evaluate
    if args.use_masks == "both":
        mask_variants = [True, False]
    else:
        mask_variants = [args.use_masks == "true"]

    original_sensor = args.sensor
    all_results = {}  # (mask_variant, sensor) -> metrics

    for use_masks in mask_variants:
        args.use_masks_bool = use_masks
        variant_label = "masked" if use_masks else "rgb"
        print(f"\n{'#'*60}")
        print(f"[Test] Image variant: {variant_label}")
        print(f"{'#'*60}")

        args.sensor = original_sensor

        # Handle --sensor all
        if args.bop_root and args.sensor == "all":
            sensors = discover_sensors(args.bop_root)
            if not sensors:
                print("[Test] No sensors discovered, falling back to default evaluation")
                args.output_csv = get_csv_path(output_dir, masked=use_masks)
                metrics = run_eval(args, model, device)
                all_results[(variant_label, "default")] = metrics
                if args.use_wandb:
                    log_metrics_to_wandb(wandb, metrics, sensor=f"default/{variant_label}")
                    log_csv_to_wandb(wandb, args.output_csv)
                continue

            print(f"[Test] Discovered sensors: {sensors}")

            for sensor in sensors:
                print(f"\n{'='*60}")
                print(f"[Test] Evaluating sensor: {sensor} ({variant_label})")
                print(f"{'='*60}")

                args.sensor = sensor
                args.output_csv = get_csv_path(output_dir, sensor=sensor, masked=use_masks)

                try:
                    metrics = run_eval(args, model, device)
                except ValueError as e:
                    print(f"[Test] Skipping sensor '{sensor}' ({variant_label}): {e}")
                    continue

                all_results[(variant_label, sensor)] = metrics

                if args.use_wandb:
                    log_metrics_to_wandb(wandb, metrics, sensor=f"{sensor}/{variant_label}")
                    log_csv_to_wandb(wandb, args.output_csv)
        else:
            sensor_name = args.sensor if args.bop_root else None
            args.output_csv = get_csv_path(output_dir, sensor=sensor_name, masked=use_masks)
            metrics = run_eval(args, model, device)
            all_results[(variant_label, sensor_name or "default")] = metrics
            if args.use_wandb:
                log_metrics_to_wandb(wandb, metrics, sensor=f"{sensor_name or 'default'}/{variant_label}")
                log_csv_to_wandb(wandb, args.output_csv)

    # Print combined summary
    print(f"\n{'='*60}")
    print("[Test] Combined Summary")
    print(f"{'='*60}")
    for (variant, sensor), m in all_results.items():
        if m:
            print(f"  {sensor} ({variant}): 2D={m['proj_2d']:.4f}  ADD={m['add']:.4f}  "
                  f"trans={m['translation_error']:.2f}mm  rot={m['rotation_error']:.2f}deg")
        else:
            print(f"  {sensor} ({variant}): (metrics printed above by evaluator)")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ContourPose evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--class_type", type=str, required=True,
                        help="Object class name")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint (.pkl)")

    # BOP format
    parser.add_argument("--bop_root", type=str, default=None,
                        help="BOP dataset root directory")
    parser.add_argument("--obj_id", type=int, default=1,
                        help="Object ID in BOP (default: 1)")
    parser.add_argument("--sensor", type=str, default="all",
                        help='Sensor name or "all" (default: all)')
    parser.add_argument("--scene_ids", type=str, default=None,
                        help="Comma-separated scene IDs (default: all)")
    parser.add_argument("--keypoints_dir", type=str, default="keypoints",
                        help="Directory with keypoint files")
    parser.add_argument("--valid3d_dir", type=str, default="Valid3D",
                        help="Directory with Valid3D files")
    parser.add_argument("--use_masks", type=str, default="true",
                        choices=["true", "false", "both"],
                        help="Use masked RGB: true, false, or both (default: true)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256],
                        help="Image size as H W (default: 256 256)")
    parser.add_argument("--train_mask_dir", type=str, default=None,
                        help="Training mask directory. Computes mean object-to-image "
                             "ratio and rescales test patches to match at inference time.")
    parser.add_argument("--fixed_scale", type=float, default=None,
                        help="Fixed scale factor applied to all test images instead of "
                             "auto-computing per-instance. E.g. 1.5 zooms in 1.5x.")

    # Legacy format
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(os.getcwd(), "data"),
                        help="Legacy test data directory (contains test/sceneN/)")
    parser.add_argument("--scene", type=int, default=13,
                        help="Scene number (legacy format)")
    parser.add_argument("--index", type=int, default=2,
                        help="Object index in scene (legacy format)")
    parser.add_argument("--legacy_test", action="store_true", default=False,
                        help="Force legacy test loader even when --bop_root is set")

    # Model data root (keypoints, Valid3D, cad)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root for model-related files (keypoints/, Valid3D/, cad/). "
                             "Defaults to --bop_root if set, else current working directory. "
                             "Use this when keypoints/Valid3D differ from --bop_root or --data_path.")

    # Checkpoint
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run subdirectory for checkpoint")
    parser.add_argument("--load_best", action="store_true", default=False,
                        help="Load best_model.pkl instead of model.pkl")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/{class_type}_{YYMMDD_HHMM})")

    # PECP
    parser.add_argument("--use_pecp", action="store_true", default=True,
                        help="Enable PECP pose refinement (default: enabled)")
    parser.add_argument("--no_pecp", action="store_false", dest="use_pecp",
                        help="Disable PECP, use standard PnP only")

    # Visualization
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="Save inference visualizations during eval")
    parser.add_argument("--num_vis", type=int, default=20,
                        help="Number of samples to visualize (default: 20)")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Log metrics, CSVs, and checkpoint to wandb")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="contourpose-eval",
                        help="Wandb project name (default: contourpose-eval)")

    # Gin config
    parser.add_argument("--gin_config", type=str, nargs="+", default=[],
                        help="Gin config files (e.g. configs/contourpose.gin)")
    parser.add_argument("--gin_param", type=str, nargs="+", default=[],
                        help="Gin parameter overrides (e.g. 'main.batch_size=16')")

    args = parser.parse_args()
    args.img_size = tuple(args.img_size)

    # Resolve data_root: explicit flag > bop_root > cwd
    if args.data_root is None:
        args.data_root = args.bop_root if args.bop_root else os.getcwd()

    # Auto-enable legacy_test when --bop_root is set but user also provided
    # --data_path explicitly (i.e. they want BOP for model, legacy for test data)
    if args.legacy_test and not args.bop_root:
        # legacy_test without bop_root is the default path anyway
        pass

    # Parse gin config files and CLI overrides
    gin.parse_config_files_and_bindings(args.gin_config, args.gin_param, skip_unknown=True)

    main(args)
