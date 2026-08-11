"""
Sanity-check the BOP test DALI loader.

Pulls one batch from get_bop_test_dali_loader() and renders a grid PNG per sample:
  [ RGB (denormalized) | mask overlay | heatmap sum overlay | GT pose overlay ]

Confirms:
  1. Loader instantiates against a converted BOP root
  2. Images come out at 480×640 with the expected normalization stats
  3. Heatmaps land on the object (peaks == keypoint positions)
  4. GT pose projects the mesh onto the object silhouette

Run from repo root inside Docker (after rtless_test_to_bop.py has run):
    python scripts/visualize_bop_test_batch.py \
        --bop_root data/RTLESS_BOP/obj21 \
        --obj_id 21 \
        --scene_id 000029

    # Multi-scene, larger batch
    python scripts/visualize_bop_test_batch.py \
        --bop_root data/RTLESS_BOP/obj21 --obj_id 21 --batch_size 8
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.BOPTestDALIDataset import get_bop_test_dali_loader
from utils.utils import load_ply, project

# Same normalization stats as BOPTestDALIPipeline
MEAN = np.array([0.419, 0.427, 0.424], dtype=np.float32)
STD  = np.array([0.184, 0.206, 0.197], dtype=np.float32)


def denorm(img_chw):
    """[3, H, W] float tensor → uint8 HWC RGB (C-contiguous)."""
    img = img_chw.transpose(1, 2, 0)          # HWC
    img = img * STD + MEAN
    return np.ascontiguousarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def heatmap_overlay(rgb_uint8, heatmap_khw, alpha=0.55):
    """Sum keypoint heatmaps and blend as a JET overlay on rgb."""
    hm = heatmap_khw.sum(axis=0)              # [H, W]
    hm = (hm / (hm.max() + 1e-8) * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, hm_rgb, alpha, 0)


def pose_overlay(rgb_uint8, mesh_pts_m, K, pose, color=(0, 220, 0)):
    """Project mesh points using GT pose; scatter them on the image.

    pose is [3,4] where t is in mm; mesh_pts_m is in m from load_ply.
    Convert mesh_pts to mm to match pose units.
    """
    pts_mm  = mesh_pts_m * 1000.0
    pts_2d  = project(pts_mm, K.astype(np.float64), pose.astype(np.float64))
    out = rgb_uint8.copy()
    H, W = out.shape[:2]
    for x, y in pts_2d:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            out[yi, xi] = color
    return out


def mask_overlay(rgb_uint8, mask_hw):
    """Show the mask channel — for BOP test loader, RGB is already mask-multiplied,
    so this just displays the raw mask presence as a red tint on non-zero pixels."""
    out = rgb_uint8.copy()
    m = (mask_hw > 0).astype(np.uint8)
    red = np.zeros_like(out)
    red[..., 0] = 255
    return cv2.addWeighted(out, 0.7, red * m[..., None], 0.3, 0)


def _resolve_mask_path(bop_root, scene_id, frame_id, inst_idx):
    """When use_masks=False the loader doesn't populate mask_path; reconstruct it
    from the (scene, frame, instance) triple using the standard BOP layout."""
    scene_dir = Path(bop_root) / "test" / f"{int(scene_id):06d}"
    for fmt in [f"{frame_id:06d}_{inst_idx:06d}.png",
                f"{frame_id:06d}_{inst_idx:02d}.png"]:
        candidate = scene_dir / "mask_visib" / fmt
        if candidate.exists():
            return candidate
    return None


def add_header(img, text, bar_h=28):
    H, W = img.shape[:2]
    bar = np.full((bar_h, W, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (6, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bop_root",   required=True,
                   help="Per-object BOP root, e.g. data/RTLESS_BOP/obj21")
    p.add_argument("--obj_id",     type=int, required=True)
    p.add_argument("--scene_id",   type=str, default=None,
                   help="Optional single scene (e.g. 000029)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--keypoints_dir", default="keypoints",
                   help="Location of keypoint files (relative to repo root)")
    p.add_argument("--cad_dir",    default="cad",
                   help="Location of PLY meshes")
    p.add_argument("--out_dir",    default="results/bop_test_batch_viz")
    p.add_argument("--split",      default="test")
    p.add_argument("--use_mask", action="store_true",
                   help="Ask DALI to zero out non-object pixels (silhouette view). "
                        "Default: OFF — show the full unmasked image DALI is loading.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = [args.scene_id] if args.scene_id else None
    print(f"\nInstantiating BOP test DALI loader:")
    print(f"  bop_root  = {args.bop_root}")
    print(f"  obj_id    = {args.obj_id}")
    print(f"  scene_ids = {scene_ids}")
    print(f"  batch_size= {args.batch_size}")

    loader, samples = get_bop_test_dali_loader(
        bop_root=args.bop_root,
        obj_id=args.obj_id,
        sensor=None,                # RT-Less is single-camera
        keypoints_dir=args.keypoints_dir,
        scene_ids=scene_ids,
        batch_size=args.batch_size,
        num_threads=2,
        device_id=0,
        img_size=(480, 640),
        use_masks=args.use_mask,    # False by default → DALI outputs full unmasked RGB
        split=args.split,
    )
    print(f"  → {len(samples)} samples total  (DALI use_masks={args.use_mask})")

    # Load mesh once for GT pose overlay
    obj_name = f"obj{args.obj_id}"
    mesh_path = Path(args.cad_dir) / f"{obj_name}.ply"
    if not mesh_path.exists():
        print(f"  [warn] mesh not found: {mesh_path} — pose overlay disabled")
        mesh_pts_m = None
    else:
        mesh_pts_m = load_ply(str(mesh_path))["pts"]

    # Pull one batch
    print("\nPulling batch...")
    batch = next(iter(loader))
    data  = batch[0]

    imgs      = data["images"].cpu().numpy()      # [B, 3, H, W]
    heatmaps  = data["heatmaps"].cpu().numpy()    # [B, K, H, W]
    Ks        = data["K"].cpu().numpy()           # [B, 3, 3]
    poses     = data["pose"].cpu().numpy()        # [B, 3, 4]

    B = imgs.shape[0]
    print(f"  batch shapes:  images={imgs.shape}  heatmaps={heatmaps.shape}  "
          f"K={Ks.shape}  pose={poses.shape}")

    # --- Diagnostic: dump K, pose, and keypoint projection for sample 0 ---
    print(f"\n[diag] K[0]:\n{Ks[0]}")
    print(f"[diag] pose[0]:\n{poses[0]}")

    kp_path = Path(args.bop_root) / "keypoints" / f"obj{args.obj_id}_mm.txt"
    kp = np.loadtxt(kp_path)
    print(f"[diag] loaded {len(kp)} keypoints from {kp_path} "
          f"(max |xyz| = {np.max(np.abs(kp)):.3f})")

    R, t = poses[0][:, :3], poses[0][:, 3]
    pts_cam = kp @ R.T + t
    pts_pix = pts_cam @ Ks[0].T
    pts_2d = pts_pix[:, :2] / pts_pix[:, 2:]
    print(f"[diag] first 3 kp projections (px):\n{pts_2d[:3]}")
    print(f"[diag] pts_2d bounds: x∈[{pts_2d[:,0].min():.1f}, {pts_2d[:,0].max():.1f}] "
          f"y∈[{pts_2d[:,1].min():.1f}, {pts_2d[:,1].max():.1f}]  "
          f"(image is 640×480)")

    for i in range(B):
        rgb = denorm(imgs[i])

        # Sample metadata for the i-th item in the first batch — the test loader
        # yields in the same order as pipeline.samples (no shuffle).
        s = samples[i]
        mask_file = _resolve_mask_path(args.bop_root, s['scene_id'],
                                        s['frame_id'], s['instance_idx'])
        if mask_file is not None:
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask_panel = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
            mask_label = f"Mask file (mask_visib/{mask_file.name})"
        else:
            mask_panel = np.zeros_like(rgb)
            mask_label = "Mask file (not found)"

        header_rgb = (f"RGB from DALI  scene={s['scene_id']} frame={s['frame_id']}"
                      f"  use_masks={args.use_mask}")

        panels = [
            add_header(rgb,                                        header_rgb),
            add_header(mask_panel,                                 mask_label),
            add_header(heatmap_overlay(rgb, heatmaps[i]),          "Heatmap sum from DALI (JET)"),
        ]
        if mesh_pts_m is not None:
            panels.append(add_header(
                pose_overlay(rgb, mesh_pts_m, Ks[i], poses[i]),
                "GT pose overlay (mesh projection)"))
        else:
            panels.append(add_header(rgb, "(no mesh — pose overlay skipped)"))

        grid = np.hstack(panels)
        out_path = out_dir / f"sample_{i:02d}_scene{s['scene_id']}_frame{s['frame_id']:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  saved: {out_path}")

    print(f"\nDone. Inspect: {out_dir}")


if __name__ == "__main__":
    main()
