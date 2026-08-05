"""
GT sanity check for obj21, scene 29, across all 416 frames.

Checks:
  1. obj_id at slot 0 is always 21
  2. GT centroid always projects within image bounds
  3. CAD mesh overlap with object mask (quantifies GT alignment)

Run from repo root inside the Docker container:
  python scripts/check_gt_scene29.py --data_path /data/Datasets/ContourPose
"""

import argparse, os, sys
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_ply  # requires torch; run inside Docker container

SCENE = "29"
SLOT  = 0
OBJ   = "obj21"
H, W  = 480, 640


def load_ply_pts(ply_path):
    """Load mesh vertices via the repo's load_ply, convert mm→m if needed."""
    mesh = load_ply(ply_path)
    pts = mesh["pts"].astype(np.float64)   # shape [N, 3]
    # eval.py:28 multiplies by 1000 (mm→m not needed here; gt.yml T is in meters).
    # load_ply returns pts in the PLY's native units (mm for RT-Less).
    if np.abs(pts).max() > 10:             # clearly in mm, not meters
        pts /= 1000.0
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/data/Datasets/ContourPose")
    p.add_argument("--n_sample_pts", type=int, default=2000,
                   help="Mesh vertices to sample for projection (default 2000)")
    args = p.parse_args()

    scene_dir = os.path.join(args.data_path, "test", f"scene{SCENE}")

    print(f"Loading GT/intrinsics for scene {SCENE}...")
    with open(os.path.join(scene_dir, "gt.yml")) as f:
        gt = yaml.safe_load(f)
    with open(os.path.join(scene_dir, "Intrinsic.yml")) as f:
        K_data = yaml.safe_load(f)

    keys = sorted(gt.keys(), key=lambda x: int(x))
    print(f"  Frames in gt.yml: {len(keys)}  (keys {keys[0]}–{keys[-1]})")

    # --- Load mesh ----------------------------------------------------------
    ply_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cad", f"{OBJ}.ply")
    pts3d = load_ply_pts(ply_path)
    print(f"  Mesh: {pts3d.shape[0]} vertices  "
          f"range ±{np.abs(pts3d).max()*1000:.0f} mm")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts3d), size=min(args.n_sample_pts, len(pts3d)),
                     replace=False)
    pts_sample = pts3d[idx]

    # --- Per-frame checks ---------------------------------------------------
    obj_ids          = []
    centroid_inbounds = []
    translations     = []
    inmask_fracs     = []
    inbound_fracs    = []

    photo_dir = os.path.join(scene_dir, "photo_cut")
    mask_dir  = os.path.join(scene_dir, "mask")

    print(f"\nRunning checks on all {len(keys)} frames...")
    for ki, k in enumerate(keys):
        entry = gt[k][SLOT]
        obj_ids.append(entry["obj_id"])
        R  = np.array(entry["m2c_R"])
        T  = np.array(entry["m2c_T"]).reshape(3, 1)
        translations.append(T.flatten())

        # Camera intrinsics
        K = None
        for kd in K_data[k]:
            if 21 in kd:
                K = np.array(kd[21])
                break
        assert K is not None, f"No K for obj21 at frame {k}"

        # Centroid projection
        xc = K[0, 0] * T[0, 0] / T[2, 0] + K[0, 2]
        yc = K[1, 1] * T[1, 0] / T[2, 0] + K[1, 2]
        centroid_inbounds.append(0 <= xc <= W and 0 <= yc <= H)

        # Project mesh
        pts_cam = R @ pts_sample.T + T          # 3×N
        xs = K[0, 0] * pts_cam[0] / pts_cam[2] + K[0, 2]
        ys = K[1, 1] * pts_cam[1] / pts_cam[2] + K[1, 2]
        inbnd = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        inbound_fracs.append(inbnd.mean())

        # Mask overlap
        mask = cv2.imread(os.path.join(mask_dir, f"{k}_21.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is not None and inbnd.any():
            xi = np.clip(xs[inbnd].astype(int), 0, W - 1)
            yi = np.clip(ys[inbnd].astype(int), 0, H - 1)
            inmask_fracs.append(float(mask[yi, xi].mean() > 0))
        else:
            inmask_fracs.append(np.nan)

        if (ki + 1) % 50 == 0:
            print(f"  ... processed {ki+1}/{len(keys)} frames")

    translations   = np.array(translations)      # [N, 3] meters
    T_mm           = translations * 1000
    inmask_arr     = np.array(inmask_fracs)
    inbound_arr    = np.array(inbound_fracs)

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 1 — slot-0 obj_id consistency")
    uid = set(obj_ids)
    print(f"  Unique obj_ids at slot 0: {uid}")
    print(f"  {'PASS' if uid == {21} else 'FAIL'}: all {len(keys)} frames have obj_id=21")

    print("\nCHECK 2 — GT centroid in image bounds")
    n_in = sum(centroid_inbounds)
    print(f"  {n_in}/{len(keys)} frames ({100*n_in/len(keys):.1f}%) — centroid projects "
          f"into [0,{W}]×[0,{H}]")
    print(f"  {'PASS' if n_in == len(keys) else 'PARTIAL'}")

    print("\nCHECK 3 — GT translation stats (mm)")
    for i, ax in enumerate(["tx", "ty", "tz"]):
        print(f"  {ax}: mean={T_mm[:,i].mean():.1f}  std={T_mm[:,i].std():.1f}  "
              f"min={T_mm[:,i].min():.1f}  max={T_mm[:,i].max():.1f}")

    print("\nCHECK 4 — CAD mesh in-bounds fraction")
    print(f"  Mean={100*inbound_arr.mean():.1f}%  "
          f"min={100*inbound_arr.min():.1f}%  max={100*inbound_arr.max():.1f}%")

    print("\nCHECK 5 — CAD mesh overlap with obj21 mask")
    valid = inmask_arr[~np.isnan(inmask_arr)]
    n_good  = (valid > 0.5).sum()
    n_total = len(valid)
    print(f"  Frames with >50% projected pts inside mask: {n_good}/{n_total} "
          f"({100*n_good/n_total:.1f}%)")
    print(f"  Mean mask-overlap fraction: {100*valid.mean():.1f}%")

    fail_frames = [keys[i] for i, v in enumerate(inmask_arr)
                   if not np.isnan(v) and v < 0.3]
    print(f"  Frames with <30% overlap (potential GT error): {len(fail_frames)}")
    if fail_frames:
        print(f"    {fail_frames[:20]}")

    print("\n" + "=" * 60)
    all_pass = (uid == {21} and n_in == len(keys)
                and n_good / n_total > 0.90)
    print(f"OVERALL: {'GT IS SOUND across scene 29' if all_pass else 'CHECK FAILURES — inspect above'}")


if __name__ == "__main__":
    main()
