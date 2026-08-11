"""
Compare per-loader NPZ dumps produced by sample_loader_poses.py.

Reports per quantity:
  - Legacy summary stats (mean/std/percentiles)
  - BOP DALI summary stats
  - Two-sample KS statistic (scalar quantities)
  - Bhattacharyya-style histogram overlap (2D distributions)
  - Coverage verdict:
      (a) match     — distributions overlap closely
      (b) DALI covers legacy — BOP range spans or exceeds legacy range in every axis
      (c) coverage gap        — legacy has regions BOP never samples (or vice-versa)

Writes:
  {out_dir}/comparison_report.txt
  {out_dir}/{quantity}.png      — histograms for each scalar quantity
  {out_dir}/pose_coverage.png   — 3D view-vector sphere scatter
  {out_dir}/keypoint_scatter.png — per-keypoint 2D density

Usage:
  python tests/loader_comparison/compare_distributions.py \
    --legacy results/loader_compare/obj1/legacy.npz \
    --bop_dali results/loader_compare/obj1/bop_dali.npz \
    --out_dir results/loader_compare/obj1/report
"""

import argparse
from pathlib import Path

import numpy as np


# ---- Metrics ----------------------------------------------------------------

def _ks(a, b):
    """Two-sample KS statistic (no p-value, just the D)."""
    from scipy import stats
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def _coverage_verdict(a_lo, a_hi, b_lo, b_hi, tol=0.05, a_std=None, b_std=None):
    """(a) match  (b) B covers A  (c) gap. tol as fraction of A's range.
    Special case: if both distributions are effectively constants (std≈0)
    and their means are numerically close, that's a-match (fixed-value
    quantities like camera-object distance in these render sets)."""
    if a_std is not None and b_std is not None:
        a_const = a_std < 1e-6 * max(abs(a_hi), abs(a_lo), 1.0)
        b_const = b_std < 1e-6 * max(abs(b_hi), abs(b_lo), 1.0)
        if a_const and b_const:
            mean_a = 0.5 * (a_lo + a_hi)
            mean_b = 0.5 * (b_lo + b_hi)
            scale = max(abs(mean_a), abs(mean_b), 1.0)
            if abs(mean_a - mean_b) < 1e-3 * scale:
                return "a-match (both constant)"
    a_range = max(a_hi - a_lo, 1e-9)
    thr = tol * a_range
    a_in_b = (b_lo - thr) <= a_lo and a_hi <= (b_hi + thr)
    b_in_a = (a_lo - thr) <= b_lo and b_hi <= (a_hi + thr)
    if a_in_b and b_in_a:
        return "a-match"
    if a_in_b:
        return "b-BOP-covers-legacy"
    return "c-gap"


def _rot_angle_from_R(Rs):
    """R [N,3,3] -> angle (rad) via arccos((tr-1)/2)."""
    tr = np.einsum("nii->n", Rs)
    cos = np.clip((tr - 1) / 2, -1, 1)
    return np.arccos(cos)


def _view_vectors(poses):
    """Camera-facing direction of the object in camera frame: R.T @ [0,0,1]."""
    Rs = poses[:, :3, :3]
    v = Rs.transpose(0, 2, 1)[:, :, 2]   # last column of R.T
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def _image_plane_center(poses, Ks):
    """Project object origin (0,0,0) through pose,K → 2D pixel."""
    t = poses[:, :, 3]  # [N, 3]
    cam = t
    uv = (Ks @ cam[:, :, None])[:, :, 0]
    return uv[:, :2] / uv[:, 2:3]


# ---- Reporting --------------------------------------------------------------

def _stats(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    min=float("nan"), p25=float("nan"),
                    median=float("nan"), p75=float("nan"), max=float("nan"))
    return dict(
        n=int(x.size),
        mean=float(x.mean()),
        std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        min=float(x.min()),
        p25=float(np.percentile(x, 25)),
        median=float(np.median(x)),
        p75=float(np.percentile(x, 75)),
        max=float(x.max()),
    )


def _fmt(s):
    return (f"n={s['n']:>4d}  mean={s['mean']:.3f}  std={s['std']:.3f}  "
            f"min={s['min']:.3f}  p25={s['p25']:.3f}  med={s['median']:.3f}  "
            f"p75={s['p75']:.3f}  max={s['max']:.3f}")


def _hist_png(path, legacy, bop, title, xlabel, bins=50, xunit=""):
    """Overlay density histograms. Handles constant-value inputs (bin width 0
    with density=True → divide-by-zero) by falling back to unnormalized counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    legacy = np.asarray(legacy); bop = np.asarray(bop)
    both = np.concatenate([legacy, bop])
    lo, hi = float(both.min()), float(both.max())
    density = (hi - lo) > 1e-9
    hist_kw = dict(bins=bins, alpha=0.5, density=density,
                   range=(lo - 0.5, hi + 0.5) if not density else None)
    ax.hist(legacy, label="legacy",   **hist_kw)
    ax.hist(bop,    label="bop_dali", **hist_kw)
    ax.set_xlabel(f"{xlabel} {xunit}".strip())
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title + ("" if density else "  (constant-valued: count shown)"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _sphere_png(path, legacy_v, bop_v, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 5))
    for i, (v, label, color) in enumerate([(legacy_v, "legacy", "tab:blue"),
                                            (bop_v, "bop_dali", "tab:orange")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=6, c=color, alpha=0.5)
        ax.set_title(f"{label} view vectors (n={len(v)})")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _keypoint_scatter_png(path, legacy_2d, bop_2d, title, img_wh=(640, 480)):
    """One panel per keypoint pair; overlay both loaders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    num_kp = legacy_2d.shape[1]
    ncol = min(4, num_kp)
    nrow = int(np.ceil(num_kp / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    W, H = img_wh
    for k in range(num_kp):
        ax = axes[k // ncol][k % ncol]
        ax.scatter(legacy_2d[:, k, 0], legacy_2d[:, k, 1], s=3, alpha=0.4, c="tab:blue", label="legacy")
        ax.scatter(bop_2d[:, k, 0],    bop_2d[:, k, 1],    s=3, alpha=0.4, c="tab:orange", label="bop_dali")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f"kp {k}")
        if k == 0:
            ax.legend(loc="lower right", fontsize=7)
    for k in range(num_kp, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---- Main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", required=True, help="legacy NPZ from sample_loader_poses.py")
    p.add_argument("--bop_dali", required=True, help="bop_dali NPZ")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    L = np.load(args.legacy, allow_pickle=True)
    B = np.load(args.bop_dali, allow_pickle=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    class_type = str(L["class_type"])
    assert str(B["class_type"]) == class_type, "loaders sampled different classes"

    lines = []
    def emit(*s): lines.extend(s); print(*s, sep="\n")

    emit(f"Loader comparison — {class_type}",
         "=" * 60,
         f"  legacy   N = {L['poses'].shape[0]}",
         f"  bop_dali N = {B['poses'].shape[0]}",
         "")

    # 1. Rotation coverage
    ang_L = np.degrees(_rot_angle_from_R(L["poses"][:, :3, :3]))
    ang_B = np.degrees(_rot_angle_from_R(B["poses"][:, :3, :3]))
    emit("[1] Rotation angle (deg, R vs identity)",
         f"  legacy   {_fmt(_stats(ang_L))}",
         f"  bop_dali {_fmt(_stats(ang_B))}",
         f"  KS = {_ks(ang_L, ang_B):.3f}   "
         f"verdict = {_coverage_verdict(ang_L.min(), ang_L.max(), ang_B.min(), ang_B.max())}",
         "")
    _hist_png(out / "rotation_angle.png", ang_L, ang_B,
              f"{class_type} rotation angle", "angle", xunit="(deg)")

    # 2. View-vector sphere coverage
    vL = _view_vectors(L["poses"])
    vB = _view_vectors(B["poses"])
    # scalar surrogate: mean pairwise dot product (concentration)
    conc_L = float((vL @ vL.mean(axis=0)).mean())
    conc_B = float((vB @ vB.mean(axis=0)).mean())
    emit("[2] View-vector sphere concentration (higher = more clustered)",
         f"  legacy   mean_cos_to_mean = {conc_L:.4f}",
         f"  bop_dali mean_cos_to_mean = {conc_B:.4f}",
         "")
    _sphere_png(out / "pose_coverage.png", vL, vB,
                f"{class_type} view-vector coverage")

    # 3. Translation magnitude — legacy poses are in metres (raw _RT.pkl),
    # BOP in mm. Auto-scale legacy to mm if the two ranges differ by ~1000×
    # so the KS/verdict compares physical distances, not units.
    tL_raw = np.linalg.norm(L["poses"][:, :, 3], axis=1)
    tB     = np.linalg.norm(B["poses"][:, :, 3], axis=1)
    tL = tL_raw
    scale_note = ""
    if tL_raw.max() > 0 and tB.max() / tL_raw.max() > 100:
        tL = tL_raw * 1000.0
        scale_note = "  (legacy scaled ×1000, m → mm to match BOP)"
    sL, sB = _stats(tL), _stats(tB)
    emit(f"[3] Translation magnitude ||t||{scale_note}",
         f"  legacy   {_fmt(sL)}",
         f"  bop_dali {_fmt(sB)}",
         f"  KS = {_ks(tL, tB):.3f}   "
         f"verdict = {_coverage_verdict(tL.min(), tL.max(), tB.min(), tB.max(), a_std=sL['std'], b_std=sB['std'])}",
         "")
    _hist_png(out / "translation_norm.png", tL, tB,
              f"{class_type} translation magnitude", "||t||", xunit="(mm)")

    # 4. Image-plane center of object — projection is unit-invariant so no
    # scaling needed even though legacy t is in metres and BOP in mm.
    ctrL = _image_plane_center(L["poses"], L["Ks"])
    ctrB = _image_plane_center(B["poses"], B["Ks"])
    for i, ax_name in enumerate(("x", "y")):
        sL_i, sB_i = _stats(ctrL[:, i]), _stats(ctrB[:, i])
        emit(f"[4.{ax_name}] Image-plane center {ax_name}",
             f"  legacy   {_fmt(sL_i)}",
             f"  bop_dali {_fmt(sB_i)}",
             f"  KS = {_ks(ctrL[:, i], ctrB[:, i]):.3f}   "
             f"verdict = {_coverage_verdict(ctrL[:, i].min(), ctrL[:, i].max(), ctrB[:, i].min(), ctrB[:, i].max(), a_std=sL_i['std'], b_std=sB_i['std'])}",
             "")
    _hist_png(out / "image_center_x.png", ctrL[:, 0], ctrB[:, 0],
              f"{class_type} object center x", "x (px)")
    _hist_png(out / "image_center_y.png", ctrL[:, 1], ctrB[:, 1],
              f"{class_type} object center y", "y (px)")

    # 5. Per-keypoint 2D locations (from projected pose+K)
    kL, kB = L["keypoints_2d_pr"], B["keypoints_2d_pr"]
    num_kp = kL.shape[1]
    emit(f"[5] Per-keypoint 2D pixel distribution (projected via emitted pose,K; {num_kp} kps)",
         "     kp | legacy (mx, my, sx, sy)      | bop (mx, my, sx, sy)       | KS_x  KS_y")
    for k in range(num_kp):
        mxL, myL = kL[:, k, 0].mean(), kL[:, k, 1].mean()
        sxL, syL = kL[:, k, 0].std(),  kL[:, k, 1].std()
        mxB, myB = kB[:, k, 0].mean(), kB[:, k, 1].mean()
        sxB, syB = kB[:, k, 0].std(),  kB[:, k, 1].std()
        ksx = _ks(kL[:, k, 0], kB[:, k, 0])
        ksy = _ks(kL[:, k, 1], kB[:, k, 1])
        emit(f"     {k:2d} | {mxL:6.1f} {myL:6.1f} {sxL:6.1f} {syL:6.1f} "
             f"| {mxB:6.1f} {myB:6.1f} {sxB:6.1f} {syB:6.1f} | {ksx:.3f} {ksy:.3f}")
    emit("")
    _keypoint_scatter_png(out / "keypoint_scatter.png", kL, kB,
                          f"{class_type} keypoint 2D scatter (projected)")

    # 6. Photometric per-channel stats
    csL = L["channel_stats"]  # [N, 3, 2]
    csB = B["channel_stats"]
    emit("[6] Photometric — per-channel mean/std of normalized images (averaged over N samples)")
    for c, name in enumerate("RGB"):
        emit(f"  {name}: legacy   sample_mean={csL[:, c, 0].mean():+.3f}  sample_std={csL[:, c, 1].mean():.3f}")
        emit(f"     bop_dali sample_mean={csB[:, c, 0].mean():+.3f}  sample_std={csB[:, c, 1].mean():.3f}")
    emit("")

    # 7. Consistency check on each loader (projection ↔ heatmap peak)
    err_L = np.linalg.norm(L["keypoints_2d_pr"] - L["heatmap_peaks"], axis=2)
    err_B = np.linalg.norm(B["keypoints_2d_pr"] - B["heatmap_peaks"], axis=2)
    emit("[7] Projection ↔ heatmap-peak pixel error (per-loader invariant)",
         f"  legacy   median={np.median(err_L):.2f} px  p95={np.percentile(err_L, 95):.2f} px  max={err_L.max():.2f} px",
         f"  bop_dali median={np.median(err_B):.2f} px  p95={np.percentile(err_B, 95):.2f} px  max={err_B.max():.2f} px",
         "  NB: legacy expected to be nonzero — random_rotation_and_resize / random_translation warp",
         "      the heatmap but do NOT update pose. BOP DALI should be sub-pixel.",
         "")

    # 8. Final pose-coverage verdict
    emit("=" * 60,
         "Overall pose-coverage verdict (rotation + translation + center):",
         f"  rotation:   {_coverage_verdict(ang_L.min(), ang_L.max(), ang_B.min(), ang_B.max(), a_std=float(ang_L.std(ddof=1)), b_std=float(ang_B.std(ddof=1)))}",
         f"  ||t||:      {_coverage_verdict(tL.min(), tL.max(), tB.min(), tB.max(), a_std=sL['std'], b_std=sB['std'])}",
         f"  center_x:   {_coverage_verdict(ctrL[:, 0].min(), ctrL[:, 0].max(), ctrB[:, 0].min(), ctrB[:, 0].max(), a_std=float(ctrL[:, 0].std(ddof=1)), b_std=float(ctrB[:, 0].std(ddof=1)))}",
         f"  center_y:   {_coverage_verdict(ctrL[:, 1].min(), ctrL[:, 1].max(), ctrB[:, 1].min(), ctrB[:, 1].max(), a_std=float(ctrL[:, 1].std(ddof=1)), b_std=float(ctrB[:, 1].std(ddof=1)))}",
         "",
         "Legend: a-match / b-BOP-covers-legacy / c-gap")

    (out / "comparison_report.txt").write_text("\n".join(lines))
    print(f"\nWrote: {out}/comparison_report.txt")
    print(f"Plots: {out}/*.png")


if __name__ == "__main__":
    main()
