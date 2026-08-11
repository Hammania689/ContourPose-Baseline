"""
Sanity-check the BOP *training* DALI loader — the exact loader train_bop.py uses.

Answers two questions:
  1. Is the SUN2012 background being composited onto the PBR render?
  2. Are the DALI augmentations (HSV / brightness-contrast / blur) actually firing?

For each batch sample, writes a grid PNG:
  [ raw PBR from disk | mask overlay | DALI output (post-composite + augs) | |Δ| pixel diff ]

The right two panels come out of the same DALI pipeline train_bop.py builds via
create_bop_validation_setup(), so any config mismatch you see here is the same
one training sees.

Run inside the container from the repo root:

  python scripts/visualize_bop_train_batch.py \
      --bop_root /contourpose-baseline-4090/data/RTLESS_BOP \
      --obj_id 18 --class_type obj18 \
      --background_dir /data/SUN2012pascalformat/JPEGImages

Add `--no_background` to reproduce the "no bg_dir configured" case for comparison.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.data_utils import create_bop_validation_setup

MEAN = np.array([0.419, 0.427, 0.424], dtype=np.float32)
STD  = np.array([0.184, 0.206, 0.197], dtype=np.float32)


def denorm(img_chw):
    img = img_chw.transpose(1, 2, 0)          # HWC
    img = img * STD + MEAN
    return np.ascontiguousarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def add_header(img, text, bar_h=28):
    H, W = img.shape[:2]
    bar = np.full((bar_h, W, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (6, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def mask_overlay(rgb_uint8, mask_hw):
    out = rgb_uint8.copy()
    m = (mask_hw > 0).astype(np.uint8)
    red = np.zeros_like(out)
    red[..., 0] = 255
    return cv2.addWeighted(out, 0.7, red * m[..., None], 0.3, 0)


def diff_panel(raw_uint8, dali_uint8):
    """Per-pixel |Δ| between raw disk RGB and DALI output, JET-colored."""
    if raw_uint8.shape != dali_uint8.shape:
        raw_uint8 = cv2.resize(raw_uint8, (dali_uint8.shape[1], dali_uint8.shape[0]))
    diff = np.abs(raw_uint8.astype(np.int16) - dali_uint8.astype(np.int16)).mean(axis=2)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def load_raw_rgb(path, size_hw):
    """Load the PNG straight from disk (BGR→RGB, resized to loader size)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((*size_hw, 3), dtype=np.uint8)
    bgr = cv2.resize(bgr, (size_hw[1], size_hw[0]))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_raw_mask(path, size_hw):
    if path is None or not Path(path).exists():
        return np.zeros(size_hw, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)


def describe_bg_dir(bg_dir):
    if not bg_dir:
        return "background_dir=None → compositing SKIPPED"
    p = Path(bg_dir)
    if not p.exists():
        return f"background_dir={bg_dir} → DOES NOT EXIST → compositing SKIPPED"
    n_jpg = len(list(p.glob("*.jpg")))
    n_png = len(list(p.glob("*.png")))
    total = n_jpg + n_png
    if total == 0:
        return f"background_dir={bg_dir} → exists but 0 .jpg/.png files → compositing SKIPPED"
    return f"background_dir={bg_dir} → {total} bg images ({n_jpg} jpg, {n_png} png) → compositing ON"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bop_root",       required=True,
                   help="Dataset ROOT (contains train_pbr/, keypoints/, Valid3D/, models/). "
                        "Same value you pass to train_bop.py --bop_root.")
    p.add_argument("--obj_id",         type=int, required=True)
    p.add_argument("--class_type",     required=True, help="e.g. obj18")
    p.add_argument("--background_dir", default="/data/SUN2012pascalformat/JPEGImages")
    p.add_argument("--no_background",  action="store_true",
                   help="Run the loader WITHOUT background_dir — for A/B comparison.")
    p.add_argument("--keypoints_dir",  default="keypoints")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--img_size",       type=int, nargs=2, default=[480, 640])
    p.add_argument("--out_dir",        default="results/bop_train_batch_viz")
    args_cli = p.parse_args()

    img_size = tuple(args_cli.img_size)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bop_dataset_root = args_cli.bop_root
    bop_scene_dir = str(Path(bop_dataset_root) / "train_pbr" / f"{args_cli.obj_id:06d}")

    bg_dir = None if args_cli.no_background else args_cli.background_dir

    print("=" * 72)
    print("BOP training loader sanity check")
    print("=" * 72)
    print(f"  dataset root : {bop_dataset_root}")
    print(f"  scene dir    : {bop_scene_dir}")
    print(f"  obj_id       : {args_cli.obj_id}   class_type: {args_cli.class_type}")
    print(f"  {describe_bg_dir(bg_dir)}")
    print(f"  keypoints_dir: {args_cli.keypoints_dir}")
    print(f"  batch_size   : {args_cli.batch_size}   img_size: {img_size}")
    print("=" * 72 + "\n")

    if not Path(bop_scene_dir).exists():
        print(f"[FATAL] scene dir does not exist: {bop_scene_dir}")
        sys.exit(1)

    # Match train_bop.py's args shape.
    loader_args = SimpleNamespace(
        bop_root       = bop_scene_dir,
        obj_id         = args_cli.obj_id,
        class_type     = args_cli.class_type,
        keypoints_dir  = args_cli.keypoints_dir,
        batch_size     = args_cli.batch_size,
        img_size       = img_size,
        background_dir = bg_dir,
        compute_edge_input = False,
    )

    print("[Setup] Building train/val loaders (mirrors train_bop.py exactly)...\n")
    val_setup = create_bop_validation_setup(loader_args, num_gpus=1)
    train_loader = val_setup["train_loader"]

    # Peek at the underlying pipeline to confirm what was actually configured.
    pipe = train_loader._pipes[0] if hasattr(train_loader, "_pipes") else None
    if pipe is not None:
        print("\n[Pipeline state]")
        print(f"  num_samples : {getattr(pipe, 'num_samples', '?')}")
        print(f"  bg_files    : {len(getattr(pipe, 'bg_files', []))}")
        print(f"  first sample rgb : {pipe.samples[0]['rgb_path']}")
        print(f"  first sample mask: {pipe.samples[0]['mask_path']}")
        if len(getattr(pipe, 'bg_files', [])) == 0:
            print("  >>> bg_files is EMPTY: background compositing is a no-op.\n")
        else:
            print("  >>> compositing WILL run for every batch.\n")

    print("[Batch] Pulling one training batch...")
    batch = next(iter(train_loader))[0]
    imgs = batch["images"].cpu().numpy()   # [B, 3, H, W]
    B = imgs.shape[0]
    print(f"  images shape: {imgs.shape}   dtype: {imgs.dtype}")

    # Per-sample diagnostics + panel writing
    for i in range(B):
        # DALI pipeline iterates samples in the order they appear in file_indices
        # (random_shuffle=False in fn.readers.file, no shuffle when file_indices is set).
        s = pipe.samples[i]
        raw_rgb  = load_raw_rgb(s['rgb_path'], img_size)
        raw_mask = load_raw_mask(s['mask_path'], img_size)
        dali_rgb = denorm(imgs[i])
        diff = diff_panel(raw_rgb, dali_rgb)

        mean_diff = float(np.mean(np.abs(raw_rgb.astype(np.int16) -
                                          dali_rgb.astype(np.int16))))

        panels = [
            add_header(raw_rgb,                    f"raw PBR from disk (frame {s['frame_id']:06d})"),
            add_header(mask_overlay(raw_rgb, raw_mask),
                       "raw mask overlay"),
            add_header(dali_rgb,                   "DALI output (composited + augmented)"),
            add_header(diff,                       f"|Δ| raw vs DALI  (mean={mean_diff:.1f}/255)"),
        ]
        grid = np.hstack(panels)
        out_path = out_dir / f"sample_{i:02d}_frame{s['frame_id']:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  saved {out_path}   mean|Δ|={mean_diff:.1f}")

    print("\n[Interpretation]")
    print("  - If bg_files=0 above → fix background_dir (path or contents), that's why no overlay.")
    print("  - If mean|Δ| is ~0 on every sample → augmentations are not firing.")
    print("  - If DALI panel shows scene background from SUN2012 → compositing works.")
    print(f"\nDone. Inspect: {out_dir}")


if __name__ == "__main__":
    main()
