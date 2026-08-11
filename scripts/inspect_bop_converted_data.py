"""
Inspect converted BOP test data — reads raw files from disk (no DALI).

Complements scripts/visualize_bop_test_batch.py:
  - visualize_bop_test_batch.py  → validates the DALI loader
  - inspect_bop_converted_data.py → validates the raw converted files on disk

Shows FULL unmasked RGB (whatever the raw rgb/{frame}.png actually contains).
Renders a grid PNG per sample:
  [ RGB (raw)  |  Mask file  |  Heatmap on RGB  |  GT pose overlay ]

Run from repo root inside Docker (after rtless_test_to_bop.py has run):
    python scripts/inspect_bop_converted_data.py \
        --bop_root data/RTLESS_BOP/obj21 --obj_id 21 --scene_id 000029

    # Multiple samples, all scenes
    python scripts/inspect_bop_converted_data.py \
        --bop_root data/RTLESS_BOP/obj21 --obj_id 21 --n_samples 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_ply, project


def load_scene_metadata(scene_dir):
    with open(scene_dir / "scene_gt.json") as f:
        scene_gt = json.load(f)
    with open(scene_dir / "scene_camera.json") as f:
        scene_camera = json.load(f)
    return scene_gt, scene_camera


def gen_heatmap_2d(keypoints_2d, H, W, sigma=5.0):
    """Generate a summed Gaussian heatmap over all 2D keypoints."""
    hm = np.zeros((H, W), dtype=np.float32)
    for (x, y) in keypoints_2d:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            hm[yi, xi] = 1.0
    ksize = int(6 * sigma) | 1
    hm = cv2.GaussianBlur(hm, (ksize, ksize), sigmaX=sigma)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def heatmap_overlay(rgb, hm, alpha=0.55):
    hm_u8  = (hm * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, hm_rgb, alpha, 0)


def pose_overlay(rgb, mesh_pts_m, K, pose_Rt, color=(0, 220, 0)):
    """Project mesh_pts_m (m) using pose (t in mm) and scatter on rgb."""
    pts_mm = mesh_pts_m * 1000.0
    pts_2d = project(pts_mm, K.astype(np.float64), pose_Rt.astype(np.float64))
    out = rgb.copy()
    H, W = out.shape[:2]
    for x, y in pts_2d:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            out[yi, xi] = color
    return out


def add_header(img, text, bar_h=28):
    H, W = img.shape[:2]
    bar = np.full((bar_h, W, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (6, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def find_keypoints(bop_root, obj_name):
    """Find keypoints file in per-object BOP root (from rtless_test_to_bop.py --> keypoints/)."""
    for candidate in [
        Path(bop_root) / "keypoints" / f"{obj_name}.txt",
        Path("keypoints") / f"{obj_name}.txt",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No keypoints file for {obj_name}; tried per-object BOP root and repo /keypoints")


def collect_samples(bop_root, obj_id, scene_id_filter, n_samples):
    """Walk BOP tree and collect (scene_dir, frame_id, gt, K) samples, capped at n_samples."""
    test_dir = Path(bop_root) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"No test/ dir at {test_dir}")

    if scene_id_filter:
        scene_dirs = [test_dir / scene_id_filter]
    else:
        scene_dirs = sorted(d for d in test_dir.iterdir()
                             if d.is_dir() and d.name.isdigit())

    samples = []
    for scene_dir in scene_dirs:
        scene_gt, scene_camera = load_scene_metadata(scene_dir)
        for frame_key in sorted(scene_gt.keys(), key=lambda x: int(x)):
            gt_list = scene_gt[frame_key]
            for inst_idx, gt in enumerate(gt_list):
                if gt["obj_id"] != obj_id:
                    continue
                samples.append({
                    "scene_dir":   scene_dir,
                    "scene_id":    scene_dir.name,
                    "frame_id":    int(frame_key),
                    "instance":    inst_idx,
                    "R":           np.array(gt["cam_R_m2c"]).reshape(3, 3),
                    "t_mm":        np.array(gt["cam_t_m2c"]).reshape(3, 1),
                    "K":           np.array(scene_camera[frame_key]["cam_K"]).reshape(3, 3),
                })
                if len(samples) >= n_samples:
                    return samples
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bop_root",  required=True,
                   help="Per-object BOP root, e.g. data/RTLESS_BOP/obj21")
    p.add_argument("--obj_id",    type=int, required=True)
    p.add_argument("--scene_id",  type=str, default=None,
                   help="Filter to a single scene (e.g. 000029). Default: all scenes.")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--cad_dir",   default="cad")
    p.add_argument("--out_dir",   default="results/bop_converted_inspect")
    args = p.parse_args()

    obj_name = f"obj{args.obj_id}"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keypoints_3d_m = np.loadtxt(find_keypoints(args.bop_root, obj_name)).astype(np.float64)
    print(f"Loaded {keypoints_3d_m.shape[0]} keypoints for {obj_name}")

    mesh_path = Path(args.cad_dir) / f"{obj_name}.ply"
    mesh_pts_m = load_ply(str(mesh_path))["pts"] if mesh_path.exists() else None
    if mesh_pts_m is None:
        print(f"[warn] mesh not found at {mesh_path} — pose overlay panel will be skipped")

    samples = collect_samples(args.bop_root, args.obj_id, args.scene_id, args.n_samples)
    if not samples:
        print(f"No samples found for obj_id={args.obj_id} in {args.bop_root}")
        return
    print(f"Rendering {len(samples)} samples\n")

    for i, s in enumerate(samples):
        rgb_path  = s["scene_dir"] / "rgb"        / f"{s['frame_id']:06d}.png"
        mask_path = s["scene_dir"] / "mask_visib" / f"{s['frame_id']:06d}_{s['instance']:06d}.png"

        rgb_bgr = cv2.imread(str(rgb_path))
        if rgb_bgr is None:
            print(f"  [skip] cannot read {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) if mask is not None else \
                     np.zeros_like(rgb)

        pose_Rt = np.hstack([s["R"], s["t_mm"]])

        kpts_2d = project(keypoints_3d_m * 1000.0,
                          s["K"].astype(np.float64),
                          pose_Rt.astype(np.float64))
        H, W = rgb.shape[:2]
        hm = gen_heatmap_2d(kpts_2d, H, W, sigma=5.0)

        panels = [
            add_header(rgb,                                    f"RGB (raw)  scene={s['scene_id']} frame={s['frame_id']}"),
            add_header(mask_panel,                             "Mask file (mask_visib/)"),
            add_header(heatmap_overlay(rgb, hm),               "Heatmap on RGB (GT keypoints projected)"),
        ]
        if mesh_pts_m is not None:
            panels.append(add_header(
                pose_overlay(rgb, mesh_pts_m, s["K"], pose_Rt),
                "GT pose overlay (mesh projection)"))
        else:
            panels.append(add_header(rgb, "(no mesh — pose overlay skipped)"))

        grid = np.hstack(panels)
        out_path = out_dir / f"{obj_name}_scene{s['scene_id']}_frame{s['frame_id']:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  saved: {out_path}")

    print(f"\nDone. Inspect: {out_dir}")


if __name__ == "__main__":
    main()
