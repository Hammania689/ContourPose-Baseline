"""
Per-frame ADD distribution for obj21 (paper: Obj 9), scene 29.

Matches the official PECP evaluation run (260609_0543_pecp).
Reuses the unmodified evaluator from eval.py with PECP enabled; only reads
the add_raw field added alongside the existing boolean add list.

Run from repo root inside Docker:
  python scripts/add_distribution_scene29.py \
      --data_path /data/Datasets/ContourPose \
      --model_dir model/paper_checkpoints/obj21

Output goes to:
  results/rtless/authors_checkpoints/260609_0543_pecp/scene29_add_distribution/
    add_distribution_scene29.png   — log-scale histogram
    add_perframe_scene29.png       — log-scale per-frame scatter
    add_summary_scene29.txt        — aggregate + distribution stats
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.Dataset import MyDataset
from eval import evaluator
from config import diameters

SCENE        = 29
INDEX        = 0        # obj21 is slot 0 in scene 29 (verified via sceneObjs.yml)
OBJ          = "obj21"
EXPECTED_ADD = 0.3005   # official PECP aggregate; script aborts if result differs >0.5pp
DISPLAY_CEIL = 1e4      # mm — distances above this are PnP collapses; clipped for display


def run_eval(model_dir, data_path, device):
    class _Args:
        pass
    a = _Args()
    a.class_type = OBJ
    a.threshold  = 5
    a.used_epoch = -1
    a.use_pecp   = True   # must match the official 260609_0543_pecp run

    from network import ContourPose
    from torch import nn

    keypoints = np.loadtxt(os.path.join("keypoints", f"{OBJ}.txt"))
    model = ContourPose(heatmap_dim=keypoints.shape[0])
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)

    pkls = [int(os.path.splitext(f)[0])
            for f in os.listdir(model_dir) if f.endswith(".pkl")]
    epoch = max(pkls)
    ckpt_path = os.path.join(model_dir, f"{epoch}.pkl")
    print(f"  Checkpoint: {ckpt_path}  (epoch {epoch})")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
    model.eval()

    test_set    = MyDataset(data_path, OBJ, is_train=False,
                            scene=SCENE, index=INDEX)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=4)

    return evaluator(a, model, test_loader, device).evaluate()


def make_plots(add_raw_m, add_bool, threshold_m, out_dir, add_pct):
    os.makedirs(out_dir, exist_ok=True)

    add_mm       = np.array(add_raw_m) * 1000
    threshold_mm = threshold_m * 1000
    n_frames     = len(add_mm)
    pass_mask    = np.array(add_bool, dtype=bool)

    n_clipped = int((add_mm > DISPLAY_CEIL).sum())
    add_disp  = np.clip(add_mm, 1e-1, DISPLAY_CEIL)  # floor at 0.1 mm for log scale

    clip_note = (f"\n{n_clipped} frame(s) clipped to display ceiling "
                 f"({DISPLAY_CEIL:.0e} mm) — PnP collapse, exact magnitude meaningless"
                 if n_clipped else "")

    # ------------------------------------------------------------------
    # 1. Log-scale histogram (PDF for print)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.logspace(np.log10(0.1), np.log10(DISPLAY_CEIL), 55)

    ax.hist(add_disp[pass_mask],  bins=bins, color="#2ecc71", alpha=0.80,
            label=f"Correct pose ({pass_mask.sum()} frames)")
    ax.hist(add_disp[~pass_mask], bins=bins, color="#e74c3c", alpha=0.80,
            label=f"Failed pose ({(~pass_mask).sum()} frames)")

    ax.axvline(threshold_mm, color="black", linewidth=1.8, linestyle="--")
    ax.text(threshold_mm * 1.12, 0.97,
            f"Pass/fail line\n(pose within {threshold_mm:.1f} mm\nof ground truth)",
            fontsize=8, va="top", ha="left", color="black",
            transform=ax.get_xaxis_transform())

    if n_clipped:
        ax.axvline(DISPLAY_CEIL, color="grey", linewidth=1.2, linestyle=":")
        ax.text(DISPLAY_CEIL * 0.88, 0.97,
                "Total failures\n(pose meaningless,\nclipped here)",
                fontsize=8, va="top", ha="right", color="grey",
                transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_xlabel("Pose error (mm, log scale) — lower is better", fontsize=12)
    ax.set_ylabel("Frame count", fontsize=12)
    ax.set_title("Obj 9, scene 29: per-frame pose error", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0.1, DISPLAY_CEIL * 1.5)

    path1 = os.path.join(out_dir, "add_distribution_scene29.pdf")
    fig.tight_layout()
    fig.savefig(path1)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ------------------------------------------------------------------
    # 2. Log-scale per-frame scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    frames = np.arange(n_frames)

    # Non-clipped points: normal markers
    clip_pass  = pass_mask  & (add_mm <= DISPLAY_CEIL)
    clip_fail  = ~pass_mask & (add_mm <= DISPLAY_CEIL)
    ceil_pass  = pass_mask  & (add_mm >  DISPLAY_CEIL)
    ceil_fail  = ~pass_mask & (add_mm >  DISPLAY_CEIL)

    ax.scatter(frames[clip_pass], add_disp[clip_pass], s=6,
               color="#2ecc71", alpha=0.8, label="Pass")
    ax.scatter(frames[clip_fail], add_disp[clip_fail], s=6,
               color="#e74c3c", alpha=0.8, label="Fail")
    # Clipped points: triangle marker at ceiling
    if ceil_pass.any():
        ax.scatter(frames[ceil_pass], add_disp[ceil_pass], s=20,
                   color="#2ecc71", marker="^", alpha=0.9,
                   label=f"Pass (clipped, n={ceil_pass.sum()})")
    if ceil_fail.any():
        ax.scatter(frames[ceil_fail], add_disp[ceil_fail], s=20,
                   color="#e74c3c", marker="^", alpha=0.9,
                   label=f"Fail (clipped, n={ceil_fail.sum()})")

    ax.axhline(threshold_mm, color="black", linewidth=1.4, linestyle="--",
               label=f"Threshold  {threshold_mm:.1f} mm")
    if n_clipped:
        ax.axhline(DISPLAY_CEIL, color="grey", linewidth=1.0, linestyle=":",
                   label=f"Display ceiling ({DISPLAY_CEIL:.0e} mm)")

    ax.set_yscale("log")
    ax.set_xlabel("Frame index (dataset order)", fontsize=11)
    ax.set_ylabel("ADD distance (mm, log scale)", fontsize=11)
    ax.set_title(
        f"obj21 (paper: Obj 9) — scene 29 — per-frame ADD  [PECP]\n"
        f"ADD = {add_pct:.2f}%  |  {n_frames} frames" + clip_note,
        fontsize=10)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, n_frames)
    ax.set_ylim(0.1, DISPLAY_CEIL * 3)

    path2 = os.path.join(out_dir, "add_perframe_scene29.png")
    fig.tight_layout()
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/data/Datasets/ContourPose")
    p.add_argument("--model_dir", default="model/paper_checkpoints/obj21")
    p.add_argument("--out_dir",
                   default="results/rtless/authors_checkpoints/"
                           "260609_0543_pecp/scene29_add_distribution")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning evaluator (PECP=True): {OBJ}  scene={SCENE}  slot={INDEX}")

    result = run_eval(args.model_dir, args.data_path, device)

    add_bool    = result["add"]
    add_raw     = result["add_raw"]
    n_frames    = len(add_bool)
    add_pct     = 100 * np.mean(add_bool)
    threshold_m = diameters[OBJ] / 1000.0 * 0.1

    print(f"\nAggregate ADD  : {add_pct:.2f}%  (expected {EXPECTED_ADD*100:.2f}%)")
    print(f"ADD threshold  : {threshold_m*1000:.2f} mm")
    print(f"Pass frames    : {sum(add_bool)}/{n_frames}")

    # Warn if numbers deviate more than PECP's known run-to-run variance (~1pp).
    # Tolerance is 1.5pp — tighter deviations are expected PECP stochasticity
    # (unseeded random.choice in eval.py:PECP); larger gaps suggest a config error.
    # See docs/pecp_notes.md for details.
    diff_pp = abs(add_pct / 100 - EXPECTED_ADD) * 100
    if diff_pp > 1.5:
        print(f"\nERROR: aggregate {add_pct:.2f}% differs from expected "
              f"{EXPECTED_ADD*100:.2f}% by {diff_pp:.2f}pp (> 1.5pp tolerance).")
        print("Likely config mismatch — check use_pecp, checkpoint, scene/index, data_path.")
        print("Figures NOT saved.")
        sys.exit(1)
    if diff_pp > 0.0:
        print(f"  Note: {diff_pp:.2f}pp gap from expected {EXPECTED_ADD*100:.2f}% "
              f"is within PECP run-to-run variance (unseeded random sampling).")

    print(f"  Aggregate matches {EXPECTED_ADD*100:.2f}% within tolerance. ✓")

    add_mm = np.array(add_raw) * 1000
    pass_mask = np.array(add_bool, dtype=bool)
    n_clipped = int((add_mm > DISPLAY_CEIL).sum())

    print(f"\nRaw ADD distance stats (mm):")
    print(f"  mean   = {add_mm.mean():.1f}")
    print(f"  median = {np.median(add_mm):.1f}")
    print(f"  p25    = {np.percentile(add_mm, 25):.1f}")
    print(f"  p75    = {np.percentile(add_mm, 75):.1f}")
    print(f"  max    = {add_mm.max():.3e}")
    print(f"  Frames above display ceiling ({DISPLAY_CEIL:.0e} mm): {n_clipped}")
    if pass_mask.any():
        print(f"  Pass — mean={add_mm[pass_mask].mean():.1f}  "
              f"max={add_mm[pass_mask].max():.1f} mm")
    if (~pass_mask).any():
        print(f"  Fail — median={np.median(add_mm[~pass_mask]):.1f}  "
              f"min={add_mm[~pass_mask].min():.1f} mm")

    make_plots(add_raw, add_bool, threshold_m, args.out_dir, add_pct)

    txt = os.path.join(args.out_dir, "add_summary_scene29.txt")
    # txt lives alongside add_distribution_scene29.pdf and add_perframe_scene29.png
    with open(txt, "w") as f:
        f.write(f"obj21 (paper: Obj 9) — scene 29 ADD distribution  [PECP]\n")
        f.write(f"{'='*54}\n")
        f.write(f"Frames evaluated  : {n_frames}\n")
        f.write(f"ADD threshold     : {threshold_m*1000:.2f} mm "
                f"(0.1 × {diameters[OBJ]:.2f} mm diameter)\n")
        f.write(f"Pass frames       : {sum(add_bool)} / {n_frames}\n")
        f.write(f"ADD               : {add_pct:.2f}%\n\n")
        f.write(f"Distance stats (mm):\n")
        f.write(f"  mean   = {add_mm.mean():.2f}\n")
        f.write(f"  median = {np.median(add_mm):.2f}\n")
        f.write(f"  p10    = {np.percentile(add_mm, 10):.2f}\n")
        f.write(f"  p25    = {np.percentile(add_mm, 25):.2f}\n")
        f.write(f"  p75    = {np.percentile(add_mm, 75):.2f}\n")
        f.write(f"  p90    = {np.percentile(add_mm, 90):.2f}\n")
        f.write(f"  max    = {add_mm.max():.3e}\n")
        f.write(f"  Frames clipped to display ceiling: {n_clipped}\n\n")
        if pass_mask.any():
            f.write(f"Pass frames only:\n")
            f.write(f"  mean = {add_mm[pass_mask].mean():.2f}  "
                    f"max = {add_mm[pass_mask].max():.2f} mm\n")
        if (~pass_mask).any():
            f.write(f"Fail frames only:\n")
            f.write(f"  median = {np.median(add_mm[~pass_mask]):.2f}  "
                    f"min = {add_mm[~pass_mask].min():.2f} mm\n")
    print(f"  Saved: {txt}")


if __name__ == "__main__":
    main()
