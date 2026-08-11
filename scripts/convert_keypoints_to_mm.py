"""
One-shot: write BOP mm-scale keypoints and Valid3D from repo-root
metre-scale files into per-object BOP roots.

    keypoints/{obj}.txt (m)  →  {bop_root}/{obj}/keypoints/{obj}_mm.txt (mm)
    Valid3D/{obj}.txt   (m)  →  {bop_root}/{obj}/Valid3D/{obj}_mm.txt   (mm)

Use when rtless_test_to_bop.py has already been run and per-object BOP roots
exist, but the mm files inside them are missing. Idempotent — safe to re-run.

Run from repo root inside Docker:
    python scripts/convert_keypoints_to_mm.py --all
    python scripts/convert_keypoints_to_mm.py --object obj21
"""

import argparse
from pathlib import Path

import numpy as np


TRAINABLE_OBJECTS = [
    "obj1", "obj2", "obj3", "obj6", "obj7",
    "obj13", "obj16", "obj18", "obj21", "obj32",
]


def _convert_xyz_file(src: Path, dst: Path, label: str, obj_name: str) -> bool:
    if not src.exists():
        print(f"[skip] {obj_name} {label}: source not found ({src})")
        return False
    arr = np.loadtxt(src)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) xyz, got {arr.shape} in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, arr * 1000.0, fmt="%.6f", delimiter="\t")
    print(f"[ok]  {obj_name} {label}: {arr.shape[0]} pts → {dst}")
    return True


def convert_one(repo_root: Path, bop_root: Path, obj_name: str) -> int:
    """Convert both keypoints and Valid3D for one object. Returns count written."""
    n = 0
    n += _convert_xyz_file(
        repo_root / "keypoints" / f"{obj_name}.txt",
        bop_root / obj_name / "keypoints" / f"{obj_name}_mm.txt",
        "keypoints", obj_name,
    )
    n += _convert_xyz_file(
        repo_root / "Valid3D" / f"{obj_name}.txt",
        bop_root / obj_name / "Valid3D" / f"{obj_name}_mm.txt",
        "Valid3D", obj_name,
    )
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bop_root", default="data/RTLESS_BOP",
                   help="BOP root (per-object subdirs)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--object", type=str, help="Convert a single object (e.g. obj21)")
    group.add_argument("--all", action="store_true", help="Convert all trainable objects")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bop_root = Path(args.bop_root)

    objects = TRAINABLE_OBJECTS if args.all else [args.object]
    total = sum(convert_one(repo_root, bop_root, obj) for obj in objects)
    print(f"\nWrote {total} files across {len(objects)} object(s) under {bop_root}")


if __name__ == "__main__":
    main()
