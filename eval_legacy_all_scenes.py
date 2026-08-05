"""
Evaluate a ContourPose checkpoint across ALL scenes for a given object and
report aggregated metrics, matching the original eval.py pipeline exactly.

Outputs saved to results/rtless/authors_checkpoints/{YYMMDD_HHMM}_{pecp|no_pecp}/:
  run.log      — full stdout transcript
  summary.txt  — plain-English run description
  results.csv  — per-scene rows + per-object aggregate + overall average

Usage:
    # Single object
    python eval_legacy_all_scenes.py \\
        --class_type obj1 \\
        --model_dir model/paper_checkpoints/obj1 \\
        --data_path data/ContourPose_Original

    # Single object, specific scenes only
    python eval_legacy_all_scenes.py \\
        --class_type obj1 \\
        --model_dir model/paper_checkpoints/obj1 \\
        --data_path data/ContourPose_Original \\
        --scenes 3 13

    # All 10 objects
    python eval_legacy_all_scenes.py \\
        --all_objects \\
        --model_dir_root model/paper_checkpoints \\
        --data_path data/ContourPose_Original

    # All 10 objects with PECP
    python eval_legacy_all_scenes.py \\
        --all_objects \\
        --model_dir_root model/paper_checkpoints \\
        --data_path data/ContourPose_Original \\
        --pecp
"""

import argparse
import csv
import math
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.Dataset import MyDataset
from eval import evaluator
from network import ContourPose

CHECKPOINT_SUBDIRS = {
    "authors":    "authors_checkpoints",
    "reproduced": "reproduced_checkpoints",
}

# All scenes each object appears in, derived from sceneObjs.yml.
# Format: class_type -> [(scene, index), ...]
# Scenes verified present in data/ContourPose_Original/test/ (ls on 2026-05-14):
#   1 3 4 5 8 10 13 18 21 24 25 26 29 30 31 32
# Missing from the full sceneObjs.yml set: 7 (obj3), 11 (obj18), 23 (obj6)
OBJECT_SCENES = {
    "obj1":  [(3, 3),  (13, 2), (24, 0)],
    "obj2":  [(13, 1), (18, 1), (25, 3)],
    "obj3":  [(13, 0), (24, 1), (32, 2)],          # scene 7 not present on disk
    "obj6":  [(3, 0),  (24, 2), (31, 0)],           # scene 23 not present on disk
    "obj7":  [(1, 3),  (26, 2), (31, 1)],
    "obj13": [(10, 2), (25, 0), (32, 0)],
    "obj16": [(4, 0),  (8, 0)],
    "obj18": [(21, 2), (25, 2)],                    # scene 11 not present on disk
    "obj21": [(5, 2),  (29, 0)],
    "obj32": [(30, 1)],
}

ALL_OBJECTS = ["obj1", "obj2", "obj3", "obj6", "obj7",
               "obj13", "obj16", "obj18", "obj21", "obj32"]

CSV_COLUMNS = [
    "object_id", "scene_id", "n_frames",
    "add", "proj_2d",
    "trans_error_mm", "x_error_mm", "y_error_mm", "z_error_mm",
    "rot_error_deg", "alpha_error_deg", "beta_error_deg", "gamma_error_deg",
]


# ---------------------------------------------------------------------------
# Stdout tee — mirrors output to both terminal and log file
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "w")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate(raw):
    """Compute scalar metrics from raw per-frame lists."""
    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    x     = _mean(raw["x_error"])
    y     = _mean(raw["y_error"])
    z     = _mean(raw["z_error"])
    alpha = _mean(raw["alpha_error"])
    beta  = _mean(raw["beta_error"])
    gamma = _mean(raw["gamma_error"])

    return {
        "n_frames":      len(raw["proj_2d"]),
        "add":           _mean(raw["add"]),
        "proj_2d":       _mean(raw["proj_2d"]),
        "x_error_mm":    x,
        "y_error_mm":    y,
        "z_error_mm":    z,
        "trans_error_mm": math.sqrt(x**2 + y**2 + z**2) if not math.isnan(x) else float("nan"),
        "alpha_error_deg": alpha,
        "beta_error_deg":  beta,
        "gamma_error_deg": gamma,
        "rot_error_deg": math.sqrt(alpha**2 + beta**2 + gamma**2) if not math.isnan(alpha) else float("nan"),
    }


def _csv_row(object_id, scene_id, metrics):
    return {
        "object_id": object_id,
        "scene_id":  scene_id,
        **{k: metrics[k] for k in CSV_COLUMNS if k not in ("object_id", "scene_id")},
    }


def _print_aggregate(label, metrics):
    print(f"  2D proj : {metrics['proj_2d']:.4f}   ADD : {metrics['add']:.4f}")
    print(f"  x={metrics['x_error_mm']:.3f} mm  "
          f"y={metrics['y_error_mm']:.3f} mm  "
          f"z={metrics['z_error_mm']:.3f} mm  "
          f"trans={metrics['trans_error_mm']:.3f} mm")
    print(f"  alpha={metrics['alpha_error_deg']:.3f}°  "
          f"beta={metrics['beta_error_deg']:.3f}°  "
          f"gamma={metrics['gamma_error_deg']:.3f}°  "
          f"rot={metrics['rot_error_deg']:.3f}°")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(class_type, model_dir, device):
    corners = np.loadtxt(os.path.join("keypoints", f"{class_type}.txt"))
    model = ContourPose(heatmap_dim=corners.shape[0])
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)

    pkls = [int(f.split(".")[0]) for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError(f"No .pkl checkpoints found in {model_dir}")
    epoch = max(pkls)
    ckpt_path = os.path.join(model_dir, f"{epoch}.pkl")
    print(f"  Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["net"] if "net" in ckpt else ckpt
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Per-object evaluation
# ---------------------------------------------------------------------------

def evaluate_object(class_type, model_dir, data_path, device, scenes, use_pecp, skips):
    """
    Evaluate one object across all specified scenes.

    Returns:
        scene_rows  — list of CSV-ready dicts, one per scene
        obj_row     — CSV-ready dict for the per-object aggregate
    """
    print(f"\n{'='*60}")
    print(f"  {class_type}  —  {len(scenes)} scene(s): {scenes}")
    print(f"{'='*60}")

    model = load_model(class_type, model_dir, device)

    scene_rows = []
    combined = {k: [] for k in
                ("proj_2d", "add", "x_error", "y_error", "z_error",
                 "alpha_error", "beta_error", "gamma_error")}

    for scene, index in scenes:
        scene_dir = os.path.join(data_path, "test", f"scene{scene}")
        if not os.path.isdir(scene_dir):
            msg = f"{class_type}  scene={scene}  index={index} — directory not found: {scene_dir}"
            print(f"\n  [SKIP] {msg}")
            skips.append(msg)
            continue

        print(f"\n  scene={scene}  index={index}")

        class _Args:
            pass
        args = _Args()
        args.class_type = class_type
        args.threshold  = 5
        args.used_epoch = -1
        args.use_pecp   = use_pecp

        test_set    = MyDataset(data_path, class_type,
                                is_train=False, scene=scene, index=index)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=4)

        result = evaluator(args, model, test_loader, device).evaluate()

        if result:
            scene_metrics = _aggregate(result)
            scene_rows.append(_csv_row(class_type, str(scene), scene_metrics))
            for k in combined:
                combined[k].extend(result[k])

    # Per-object aggregate
    obj_metrics = _aggregate(combined)
    n = obj_metrics["n_frames"]
    print(f"\n  --- {class_type} aggregate ({n} frames, {len(scenes)} scene(s)) ---")
    _print_aggregate(class_type, obj_metrics)

    obj_row = _csv_row(class_type, "all_scenes", obj_metrics)
    return scene_rows, obj_row


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path, all_scene_rows, all_obj_rows, overall_metrics):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for obj_scene_rows, obj_row in zip(all_scene_rows, all_obj_rows):
            writer.writerows(obj_scene_rows)
            writer.writerow(obj_row)
            writer.writerow({})          # blank separator between objects

        writer.writerow(_csv_row("overall_average", "all_objects", overall_metrics))


def write_summary(path, args, objects, scenes_used, run_dir, timestamp):
    pecp_str = "enabled" if args.pecp else "disabled"
    cmd = " ".join(sys.argv)

    with open(path, "w") as f:
        f.write("ContourPose Legacy Evaluation — Run Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp   : {timestamp}\n")
        f.write(f"Output dir  : {run_dir}\n\n")
        f.write(f"Command     : {cmd}\n\n")
        f.write(f"PECP        : {pecp_str}\n")
        f.write(f"Data path   : {args.data_path}\n")
        f.write(f"Checkpoint  : {'--model_dir_root ' + args.model_dir_root if args.all_objects else '--model_dir ' + args.model_dir}\n\n")
        f.write(f"Objects evaluated ({len(objects)}):\n")
        for obj in objects:
            scene_list = scenes_used.get(obj, [])
            scene_nums = [str(s) for s, _ in scene_list]
            f.write(f"  {obj}: scenes {', '.join(scene_nums)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ContourPose across all scenes per object (legacy pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--class_type", type=str, default=None,
                        help="Single object to evaluate (e.g. obj1)")
    parser.add_argument("--all_objects", action="store_true",
                        help="Evaluate all 10 objects sequentially")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Checkpoint directory for a single object")
    parser.add_argument("--model_dir_root", type=str, default=None,
                        help="Root containing per-object checkpoint dirs "
                             "(used with --all_objects, e.g. model/paper_checkpoints)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Legacy dataset root (e.g. data/ContourPose_Original)")
    parser.add_argument("--scenes", type=int, nargs="+", default=None,
                        help="Restrict evaluation to specific scene numbers")
    parser.add_argument("--pecp", action="store_true", default=False,
                        help="Enable PECP pose refinement (default: disabled)")
    parser.add_argument("--checkpoint_source", choices=["authors", "reproduced"],
                        default="authors",
                        help="Checkpoint source: 'authors' → authors_checkpoints/, "
                             "'reproduced' → reproduced_checkpoints/ (default: authors)")
    args = parser.parse_args()

    if not args.all_objects and not args.class_type:
        parser.error("Provide --class_type or --all_objects")
    if args.all_objects and not args.model_dir_root:
        parser.error("--all_objects requires --model_dir_root")
    if not args.all_objects and not args.model_dir:
        parser.error("--class_type requires --model_dir")

    # Create output directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    pecp_label = "pecp" if args.pecp else "no_pecp"
    run_label  = f"{timestamp}_{pecp_label}"
    output_root = os.path.join("results", "rtless", CHECKPOINT_SUBDIRS[args.checkpoint_source])
    run_dir     = os.path.join(output_root, run_label)
    os.makedirs(run_dir, exist_ok=True)

    # Mirror stdout to log file
    tee = _Tee(os.path.join(run_dir, "run.log"))
    sys.stdout = tee

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Output dir  : {run_dir}")
        print(f"PECP        : {pecp_label}")

        objects = ALL_OBJECTS if args.all_objects else [args.class_type]

        # Build per-object scene lists
        scenes_used = {}
        for class_type in objects:
            if args.scenes:
                filtered = [(s, i) for s, i in OBJECT_SCENES[class_type]
                            if s in args.scenes]
            else:
                filtered = OBJECT_SCENES[class_type]
            scenes_used[class_type] = filtered

        all_scene_rows = []
        all_obj_rows   = []
        skips          = []

        for class_type in objects:
            scenes = scenes_used[class_type]
            if not scenes:
                print(f"  [SKIP] {class_type}: none of scenes {args.scenes} contain this object")
                continue

            model_dir = (os.path.join(args.model_dir_root, class_type)
                         if args.all_objects else args.model_dir)

            scene_rows, obj_row = evaluate_object(
                class_type, model_dir, args.data_path, device, scenes, args.pecp, skips)

            all_scene_rows.append(scene_rows)
            all_obj_rows.append(obj_row)

        # Overall average across all objects
        overall_combined = {k: [] for k in
                            ("proj_2d", "add", "x_error", "y_error", "z_error",
                             "alpha_error", "beta_error", "gamma_error")}
        for scene_rows in all_scene_rows:
            for row in scene_rows:
                # We don't have raw lists at this point — reconstruct from obj rows
                pass

        # Recompute overall from obj-level scalars (weighted by n_frames)
        total_frames = sum(r["n_frames"] for r in all_obj_rows)

        def _weighted(key):
            return sum(r[key] * r["n_frames"] for r in all_obj_rows) / total_frames if total_frames else float("nan")

        overall_metrics = {
            "n_frames":        total_frames,
            "add":             _weighted("add"),
            "proj_2d":         _weighted("proj_2d"),
            "x_error_mm":      _weighted("x_error_mm"),
            "y_error_mm":      _weighted("y_error_mm"),
            "z_error_mm":      _weighted("z_error_mm"),
            "trans_error_mm":  _weighted("trans_error_mm"),
            "alpha_error_deg": _weighted("alpha_error_deg"),
            "beta_error_deg":  _weighted("beta_error_deg"),
            "gamma_error_deg": _weighted("gamma_error_deg"),
            "rot_error_deg":   _weighted("rot_error_deg"),
        }

        # Print overall summary table
        if len(all_obj_rows) > 1:
            print(f"\n{'='*60}")
            print("  Overall Summary")
            print(f"{'='*60}")
            print(f"  {'Object':8} {'Frames':>7} {'2D':>8} {'ADD':>8} "
                  f"{'Trans(mm)':>10} {'Rot(°)':>8}")
            print(f"  {'-'*55}")
            for r in all_obj_rows:
                print(f"  {r['object_id']:8} {r['n_frames']:>7} "
                      f"{r['proj_2d']:>8.4f} {r['add']:>8.4f} "
                      f"{r['trans_error_mm']:>10.3f} {r['rot_error_deg']:>8.3f}")
            print(f"  {'Average':8} {total_frames:>7} "
                  f"{overall_metrics['proj_2d']:>8.4f} {overall_metrics['add']:>8.4f} "
                  f"{overall_metrics['trans_error_mm']:>10.3f} {overall_metrics['rot_error_deg']:>8.3f}")

        # Write outputs
        write_csv(
            os.path.join(run_dir, "results.csv"),
            all_scene_rows, all_obj_rows, overall_metrics,
        )
        write_summary(
            os.path.join(run_dir, "summary.txt"),
            args, objects, scenes_used, run_dir, timestamp,
        )
        # Write skip log
        skip_log_path = os.path.join(run_dir, "skipped.log")
        if skips:
            with open(skip_log_path, "w") as f:
                f.write("Scenes skipped during evaluation\n")
                f.write("=" * 60 + "\n\n")
                for entry in skips:
                    f.write(f"  SKIP: {entry}\n")
            print(f"\n  {len(skips)} scene(s) skipped — see {skip_log_path}")
        else:
            with open(skip_log_path, "w") as f:
                f.write("No scenes were skipped.\n")

        print(f"\nResults saved to: {run_dir}")

    finally:
        sys.stdout = tee._terminal
        tee.close()


if __name__ == "__main__":
    main()
