"""
Draw N samples from either the legacy MyDataset or the BOP DALI train loader
and dump per-sample quantities to an NPZ for downstream distribution analysis.

Emits, per loader, one NPZ with:
  poses            [N, 3, 4]    — as returned by the loader (post-augment for legacy)
  Ks               [N, 3, 3]
  keypoints_3d     [num_kp, 3]  — same for all samples (loaded from file)
  keypoints_2d_pr  [N, num_kp, 2] — projected via emitted (pose, K)
  heatmap_peaks    [N, num_kp, 2] — argmax over emitted heatmap (x, y)
  channel_stats    [N, 3, 2]   — per-channel (mean, std) of normalized image
  loader           str          — "legacy" or "bop_dali"
  class_type       str
  num_kp           int

Usage (inside Docker):
  python tests/loader_comparison/sample_poses.py \
    --loader legacy --class_type obj1 \
    --data_path /data/Datasets/ContourPose \
    --n 1000 --out results/loader_compare/obj1/legacy.npz

  python tests/loader_comparison/sample_poses.py \
    --loader bop_dali --class_type obj1 \
    --bop_root /mnt/bigbertha_mount/GithubWorkspace/ContourPose/data/ContourPose_PBR/obj1_train_pbr \
    --keypoints_file /mnt/bigbertha_mount/GithubWorkspace/ContourPose-Baseline/data/RTLESS_BOP/obj1/keypoints/obj1_mm.txt \
    --n 1000 --out results/loader_compare/obj1/bop_dali.npz

The two output NPZs are then compared by tests/loader_comparison/compare_distributions.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add repo root so `from dataset...` / `from utils...` imports resolve when
# invoked as `python tests/loader_comparison/sample_poses.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _project(kp3d, K, pose):
    """kp3d [N,3], K [3,3], pose [3,4] -> [N,2]."""
    R = pose[:, :3]
    t = pose[:, 3]
    cam = kp3d @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def _heatmap_peaks(heatmap):
    """heatmap [num_kp, H, W] -> [num_kp, 2] as (x, y)."""
    K, H, W = heatmap.shape
    flat = heatmap.reshape(K, -1)
    idx = flat.argmax(axis=1)
    y = idx // W
    x = idx % W
    return np.stack([x, y], axis=1).astype(np.float32)


def _channel_stats(img_chw):
    """img_chw [3, H, W] normalized -> [3, 2] (mean, std) per channel."""
    return np.stack([img_chw.mean(axis=(1, 2)),
                     img_chw.std(axis=(1, 2))], axis=1)


def sample_legacy(class_type, data_path, kp_path, n, batch_size, num_workers, seed,
                  render_dir=None, render_edge_dir=None, sun_path=None):
    """Draw n samples from MyDataset via a PyTorch DataLoader.
    If render_dir/render_edge_dir/sun_path are provided, they override the
    default legacy layout (useful when your renders live outside {root}/train/)."""
    import torch
    from torch.utils.data import DataLoader
    from dataset.Dataset import MyDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = MyDataset(data_path, class_type, is_train=True,
                   render_dir=render_dir,
                   render_edge_dir=render_edge_dir,
                   sun_path=sun_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    keypoints_3d = np.loadtxt(kp_path).astype(np.float32)  # (num_kp, 3), metres
    num_kp = keypoints_3d.shape[0]

    poses = np.zeros((n, 3, 4), dtype=np.float32)
    Ks    = np.zeros((n, 3, 3), dtype=np.float32)
    kp2d_pr = np.zeros((n, num_kp, 2), dtype=np.float32)
    hm_peaks = np.zeros((n, num_kp, 2), dtype=np.float32)
    ch_stats = np.zeros((n, 3, 2), dtype=np.float32)

    filled = 0
    for batch in loader:
        # MyDataset returns (img, heatmap, K, pose, gt_contour) in training
        img, heatmap, K, pose, _ = batch
        b = img.shape[0]
        take = min(b, n - filled)
        for i in range(take):
            p = pose[i].numpy().astype(np.float32)  # [3, 4]
            k = K[i].numpy().astype(np.float32)     # [3, 3]
            hm = heatmap[i].numpy().astype(np.float32)  # [num_kp, H, W]
            im = img[i].numpy().astype(np.float32)      # [3, H, W]
            poses[filled] = p
            Ks[filled] = k
            kp2d_pr[filled] = _project(keypoints_3d, k, p)
            hm_peaks[filled] = _heatmap_peaks(hm)
            ch_stats[filled] = _channel_stats(im)
            filled += 1
        if filled >= n:
            break

    return {
        "poses": poses[:filled], "Ks": Ks[:filled],
        "keypoints_3d": keypoints_3d, "keypoints_2d_pr": kp2d_pr[:filled],
        "heatmap_peaks": hm_peaks[:filled], "channel_stats": ch_stats[:filled],
        "loader": "legacy", "class_type": class_type, "num_kp": num_kp,
    }


def sample_bop_dali(class_type, bop_root, kp_file, n, batch_size, num_workers, seed,
                    background_dir):
    """Draw n samples from BOPDALIDataset."""
    from dataset.BOPDALIDataset import get_bop_dali_loader

    obj_id = int(class_type.replace("obj", ""))
    kp_dir = str(Path(kp_file).parent)

    loader = get_bop_dali_loader(
        data_dir=bop_root,
        obj_id=obj_id,
        keypoints_dir=kp_dir,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        seed=seed,
        img_size=(480, 640),
        background_dir=background_dir,
    )

    keypoints_3d = np.loadtxt(kp_file).astype(np.float32)  # mm-scale
    num_kp = keypoints_3d.shape[0]

    poses = np.zeros((n, 3, 4), dtype=np.float32)
    Ks    = np.zeros((n, 3, 3), dtype=np.float32)
    kp2d_pr = np.zeros((n, num_kp, 2), dtype=np.float32)
    hm_peaks = np.zeros((n, num_kp, 2), dtype=np.float32)
    ch_stats = np.zeros((n, 3, 2), dtype=np.float32)

    filled = 0
    for batch in loader:
        data = batch[0]
        imgs      = data["images"].cpu().numpy()      # [B, 3, H, W]
        heatmaps  = data["heatmaps"].cpu().numpy()    # [B, num_kp, H, W]
        Ks_b      = data["K"].cpu().numpy()           # [B, 3, 3]
        poses_b   = data["pose"].cpu().numpy()        # [B, 3, 4]
        b = imgs.shape[0]
        take = min(b, n - filled)
        for i in range(take):
            p = poses_b[i].astype(np.float32)
            k = Ks_b[i].astype(np.float32)
            poses[filled] = p
            Ks[filled] = k
            kp2d_pr[filled] = _project(keypoints_3d, k, p)
            hm_peaks[filled] = _heatmap_peaks(heatmaps[i].astype(np.float32))
            ch_stats[filled] = _channel_stats(imgs[i].astype(np.float32))
            filled += 1
        if filled >= n:
            break

    return {
        "poses": poses[:filled], "Ks": Ks[:filled],
        "keypoints_3d": keypoints_3d, "keypoints_2d_pr": kp2d_pr[:filled],
        "heatmap_peaks": hm_peaks[:filled], "channel_stats": ch_stats[:filled],
        "loader": "bop_dali", "class_type": class_type, "num_kp": num_kp,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loader", choices=["legacy", "bop_dali"], required=True)
    p.add_argument("--class_type", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    # Legacy-only
    p.add_argument("--data_path",
                   help="Legacy: root of ContourPose dataset (contains train/, test/, SUN2012...)")
    p.add_argument("--render_dir", default=None,
                   help="Legacy override: explicit renders dir (default: {data_path}/train/renders/{class_type})")
    p.add_argument("--render_edge_dir", default=None,
                   help="Legacy override: explicit render-edges dir (default: {data_path}/train/renders/gtEdge/{class_type})")
    p.add_argument("--sun_path", default=None,
                   help="Legacy override: SUN2012 root (default: {data_path}/SUN2012pascalformat)")
    # BOP DALI-only
    p.add_argument("--bop_root",
                   help="BOP DALI: data dir with rgb/, mask/, scene_gt.json, scene_camera.json")
    p.add_argument("--background_dir", default=None,
                   help="BOP DALI: optional SUN2012 dir. Omit to skip load-time bg composite.")
    # Both
    p.add_argument("--keypoints_file", required=True,
                   help="Path to keypoints file. Legacy expects metres; BOP DALI mm.")
    args = p.parse_args()

    if args.loader == "legacy":
        if not args.data_path:
            sys.exit("--data_path required for legacy loader")
        result = sample_legacy(args.class_type, args.data_path, args.keypoints_file,
                               args.n, args.batch_size, args.num_workers, args.seed,
                               render_dir=args.render_dir,
                               render_edge_dir=args.render_edge_dir,
                               sun_path=args.sun_path)
    else:
        if not args.bop_root:
            sys.exit("--bop_root required for bop_dali loader")
        result = sample_bop_dali(args.class_type, args.bop_root, args.keypoints_file,
                                 args.n, args.batch_size, args.num_workers, args.seed,
                                 args.background_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(f"[sample] wrote {result['poses'].shape[0]} samples ({args.loader}, {args.class_type}) → {out}")


if __name__ == "__main__":
    main()
