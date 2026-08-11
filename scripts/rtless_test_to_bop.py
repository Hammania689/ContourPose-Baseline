"""
Convert RT-Less native test data → BOP format (per-object roots, real file copies).

RT-Less native layout:
    {rtless_root}/test/scene{N}/
        photo_cut/{frame}_{obj_id}.png     (per-object crop)
        mask/{frame}_{obj_id}.png
        gt.yml                              (per-frame list of {m2c_R, m2c_T, obj_id})
        Intrinsic.yml                       (per-frame list of [{obj_id: K}])

Emitted BOP layout (one root per object, per Option A):
    {bop_root}/{obj_name}/
        test/{scene_id:06d}/
            rgb/{frame_id:06d}.png              (copied from photo_cut)
            mask_visib/{frame_id:06d}_000000.png (copied from mask)
            scene_gt.json      {"1": [{obj_id, cam_R_m2c[9], cam_t_m2c[3]}], ...}
            scene_camera.json  {"1": {cam_K: [9], depth_scale: 1.0}, ...}
        models/models_info.json  {"21": {diameter: 123.05}}
        keypoints/{obj_name}_mm.txt  (converted from repo-root keypoints/, so root is self-contained)
        Valid3D/{obj_name}_mm.txt    (converted from repo-root Valid3D/, used by PECP)

Default is real file copies (portable — safe to tar/upload to cloud).
Pass --symlinks for local iteration only.

Units: RT-Less gt.yml translations are in metres. BOP cam_t_m2c is in millimetres.
       Conversion multiplies by 1000.

Idempotent — safe to re-run; overwrites JSON and re-copies files in place.

Run from repo root inside Docker:
    python scripts/rtless_test_to_bop.py --object obj21
    python scripts/rtless_test_to_bop.py --all
    python scripts/rtless_test_to_bop.py --object obj21 --symlinks   # local-only
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import diameters
from eval_legacy_all_scenes import ALL_OBJECTS, OBJECT_SCENES


def obj_name_to_id(obj_name):
    """obj21 → 21"""
    return int(obj_name.replace("obj", ""))


def _place(src, dest, use_symlinks):
    """Place src at dest — real copy by default, symlink if requested. Idempotent."""
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if use_symlinks:
        dest.symlink_to(os.path.relpath(src, dest.parent))
    else:
        shutil.copy2(src, dest)


def convert_scene(rtless_root, bop_root, obj_name, obj_id, scene_id, use_symlinks):
    """Convert one (obj, scene) into a BOP scene dir. Returns number of frames written."""
    scene_src = rtless_root / "test" / f"scene{scene_id}"
    if not scene_src.is_dir():
        print(f"    [skip] scene dir missing: {scene_src}")
        return 0

    scene_dst = bop_root / obj_name / "test" / f"{scene_id:06d}"
    rgb_dst   = scene_dst / "rgb"
    mask_dst  = scene_dst / "mask_visib"
    rgb_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    with open(scene_src / "gt.yml") as f:
        gt = yaml.safe_load(f)
    with open(scene_src / "Intrinsic.yml") as f:
        intr = yaml.safe_load(f)

    scene_gt     = {}
    scene_camera = {}
    n_written    = 0
    n_missing_img = 0

    frame_keys = sorted(gt.keys(), key=lambda x: int(x))
    pbar_desc  = f"    scene {scene_id:>3}"
    for frame_key in tqdm(frame_keys, desc=pbar_desc, leave=False, unit="frame"):
        frame_id = int(frame_key)

        # Find obj_id entry in this frame's gt list
        obj_entry = next((e for e in gt[frame_key] if e["obj_id"] == obj_id), None)
        if obj_entry is None:
            continue  # this scene doesn't have this object in this frame

        # Source paths (RT-Less native)
        photo_src = scene_src / "photo_cut" / f"{frame_id}_{obj_id}.png"
        mask_src  = scene_src / "mask"      / f"{frame_id}_{obj_id}.png"
        if not photo_src.exists() or not mask_src.exists():
            n_missing_img += 1
            continue

        # Destination paths (BOP)
        rgb_dest  = rgb_dst  / f"{frame_id:06d}.png"
        mask_dest = mask_dst / f"{frame_id:06d}_000000.png"

        _place(photo_src, rgb_dest,  use_symlinks)
        _place(mask_src,  mask_dest, use_symlinks)

        # GT pose: 3×3 R (row-major flattened) + 3×1 t (m → mm)
        R_flat = np.array(obj_entry["m2c_R"]).flatten().tolist()
        t_mm   = (np.array(obj_entry["m2c_T"]).flatten() * 1000.0).tolist()

        scene_gt[str(frame_id)] = [{
            "obj_id":     obj_id,
            "cam_R_m2c":  R_flat,
            "cam_t_m2c":  t_mm,
        }]

        # K matrix — Intrinsic.yml is a list of {obj_id: K}
        K = None
        for kd in intr[frame_key]:
            if obj_id in kd:
                K = np.array(kd[obj_id]).flatten().tolist()
                break
        if K is None:
            # Fallback: use first available K in the frame (RT-Less scenes share one camera)
            K = np.array(list(intr[frame_key][0].values())[0]).flatten().tolist()

        scene_camera[str(frame_id)] = {
            "cam_K":       K,
            "depth_scale": 1.0,
        }
        n_written += 1

    with open(scene_dst / "scene_gt.json", "w") as f:
        json.dump(scene_gt, f, indent=2)
    with open(scene_dst / "scene_camera.json", "w") as f:
        json.dump(scene_camera, f, indent=2)

    msg = f"    scene {scene_id}: {n_written} frames written"
    if n_missing_img:
        msg += f"  ({n_missing_img} missing on disk)"
    print(msg)
    return n_written


def write_models_info(bop_root, obj_name, obj_id):
    """Write per-object models_info.json (diameter only; symmetries can be added later)."""
    models_dir = bop_root / obj_name / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    info = {
        str(obj_id): {
            "diameter": float(diameters[obj_name]),   # mm
        }
    }
    with open(models_dir / "models_info.json", "w") as f:
        json.dump(info, f, indent=2)


def _write_mm_xyz(src, dst):
    """Read (N, 3) metre-scale xyz from src, write mm-scale copy at dst."""
    arr = np.loadtxt(src)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) xyz, got {arr.shape} in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, arr * 1000.0, fmt="%.6f", delimiter="\t")


def write_keypoints_mm(repo_root, bop_root, obj_name):
    """Convert repo-root keypoints/{obj_name}.txt (metres, legacy) into
    {bop_root}/{obj_name}/keypoints/{obj_name}_mm.txt (millimetres, BOP convention).
    The DALI loaders and eval_spectra_pose BOP path prefer the _mm.txt file."""
    src = repo_root / "keypoints" / f"{obj_name}.txt"
    if not src.exists():
        print(f"    [warn] keypoints not found: {src}")
        return
    dst = bop_root / obj_name / "keypoints" / f"{obj_name}_mm.txt"
    _write_mm_xyz(src, dst)


def write_valid3d_mm(repo_root, bop_root, obj_name):
    """Convert repo-root Valid3D/{obj_name}.txt (metres, legacy) into
    {bop_root}/{obj_name}/Valid3D/{obj_name}_mm.txt. Used by PECP."""
    src = repo_root / "Valid3D" / f"{obj_name}.txt"
    if not src.exists():
        print(f"    [warn] Valid3D not found: {src}")
        return
    dst = bop_root / obj_name / "Valid3D" / f"{obj_name}_mm.txt"
    _write_mm_xyz(src, dst)


def convert_object(rtless_root, bop_root, obj_name, use_symlinks, repo_root):
    obj_id = obj_name_to_id(obj_name)
    scenes = [s for s, _ in OBJECT_SCENES[obj_name]]
    print(f"\n[{obj_name}]  obj_id={obj_id}  scenes={scenes}")

    total = 0
    for scene_id in scenes:
        total += convert_scene(rtless_root, bop_root, obj_name, obj_id,
                               scene_id, use_symlinks)

    write_models_info(bop_root, obj_name, obj_id)
    write_keypoints_mm(repo_root, bop_root, obj_name)
    write_valid3d_mm(repo_root, bop_root, obj_name)
    print(f"  total: {total} frames across {len(scenes)} scene(s)  →  "
          f"{bop_root / obj_name}")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtless_root", default="/data/Datasets/ContourPose",
                   help="RT-Less native root (contains test/scene{N}/...)")
    p.add_argument("--bop_root",    default="data/RTLESS_BOP",
                   help="Output BOP root (per-object subdirs will be created)")
    p.add_argument("--symlinks", action="store_true",
                   help="Use symlinks instead of real file copies "
                        "(faster/less disk, but not portable — do NOT use for cloud upload)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--object", type=str,
                       help="Convert a single object (e.g. obj21)")
    group.add_argument("--all", action="store_true",
                       help="Convert all 10 trainable objects")
    args = p.parse_args()

    rtless_root = Path(args.rtless_root)
    bop_root    = Path(args.bop_root)
    repo_root   = Path(__file__).resolve().parent.parent

    mode = "SYMLINKS (local-only)" if args.symlinks else "REAL COPIES (portable)"
    print(f"Mode: {mode}")

    objects = ALL_OBJECTS if args.all else [args.object]
    for obj in tqdm(objects, desc="Objects", unit="obj",
                    disable=len(objects) == 1):
        if obj not in OBJECT_SCENES:
            print(f"[skip] {obj} not in OBJECT_SCENES")
            continue
        convert_object(rtless_root, bop_root, obj, args.symlinks, repo_root)

    print(f"\nDone. BOP root: {bop_root}")


if __name__ == "__main__":
    main()
