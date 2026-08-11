# tests/

Characterization suites, not pytest unit tests. Each subdirectory answers one
specific question about the codebase by producing measurements you can inspect.
Run them from the repo root inside Docker.

## Layout

```
tests/
├── pecp_variance/       Measures run-to-run variance of PECP-refined eval
│   ├── run_sweep.sh
│   └── aggregate.py
└── loader_comparison/   Compares legacy MyDataset vs BOP DALI train loader
    ├── run_sweep.sh
    ├── sample_poses.py
    └── compare_distributions.py
```

Outputs land under `results/rtless_bop_variance/{ts}_pecp/` and
`results/loader_compare/{ts}/` respectively — kept out of git via
`.gitignore`.

---

## pecp_variance

**Question**: PECP subset sampling (`random.choice` × 400) and OpenCV
`solvePnPRansac` are both unseeded in the authors' code. How much do the
ADD(-S) numbers move between runs on identical inputs?

**Method**: Runs `test.py` for 10 seeds × 10 objects (plus an `obj21`
excl-scene-29 variant), pinning `random`, `numpy.random`, and
`cv2.setRNGSeed` via `--eval_seed {k}`. This is a reproducibility layer —
the algorithm and iteration counts are unchanged.

**Recipe** (from repo root):

```bash
bash tests/pecp_variance/run_sweep.sh
```

Env vars (defaults in the script header): `NUM_SEEDS`, `OBJECTS`,
`BOP_ROOT_BASE`, `CKPT_ROOT`, `GIN_CONFIG`, `SWEEP_ROOT`.

**Outputs**:

- `{SWEEP_ROOT}/seed{k}/{obj}/rgb_bop.csv`, `rgb_detailed.csv`, `metadata.json`
  — raw per-run per-instance outputs from `test.py`.
- `{SWEEP_ROOT}/variance_raw.csv` — one row per (obj, scene, seed): ADD,
  proj_2d, trans_error_mm, rot_error_deg. Save this before computing CIs
  so you can re-aggregate without re-running the sweep.
- `{SWEEP_ROOT}/variance_summary.csv` — one row per (obj, scene) with
  mean/std/min/max across seeds.
- `{SWEEP_ROOT}/variance_report.txt` — sanity checks:
  1. Do the seeds actually produce distinct ADD numbers? (If not → seeding
     didn't take, stop.)
  2. Do prior single-run reference numbers fall inside the observed range?
  3. Per-object std ranking, so you can spot which objects are noisiest.

**Interpreting the report**: if seed diversity fails on any object, the
sweep is untrustworthy for that object. If a reference falls outside the
range, either the single-run was an outlier or something material changed
in the eval path. Otherwise, the per-object std tells you how confident
you can be in any single-run number.

---

## loader_comparison

**Question**: Is the BOP DALI train loader a sound substitute for the
legacy `MyDataset`? Specifically, does the BlenderProc PBR pipeline cover
the same pose / keypoint / photometric distributions that the legacy
synthetic renderer produced?

**Method**: For each object, draws N samples from both loaders
(independently — no attempt to align samples across loaders, that would be
meaningless for two different render sources), collects emitted poses,
K, projected 2D keypoints, and per-image channel statistics. Then
compares distributions and assigns a coverage verdict per quantity:

- `a-match` — distributions overlap closely
- `b-BOP-covers-legacy` — BOP range spans or exceeds legacy on every axis
- `c-gap` — legacy has regions BOP never samples (or vice versa)

**Recipe** (from repo root):

```bash
# Full 10-object sweep
CUDA_VISIBLE_DEVICES=1 \
  PBR_ROOT_TEMPLATE='./data/RTLESS_BOP/train_pbr/{obj_id}' \
  CONTOURPOSE_ROOT=/data/Datasets/ContourPose \
  bash tests/loader_comparison/run_sweep.sh

# Single-object smoke test
OBJECTS="obj1" \
  PBR_ROOT_TEMPLATE='./data/RTLESS_BOP/train_pbr/{obj_id}' \
  CONTOURPOSE_ROOT=/data/Datasets/ContourPose \
  bash tests/loader_comparison/run_sweep.sh
```

Env vars — `PBR_ROOT_TEMPLATE` uses `{obj}` (bare class name like `obj1`)
or `{obj_id}` (zero-padded like `000001`). `LEGACY_RENDER_DIR_TEMPLATE` /
`LEGACY_RENDER_EDGE_DIR_TEMPLATE` override where `MyDataset` looks for
renders — see the script header for defaults.

**Outputs**:

- `{SWEEP_ROOT}/{obj}/legacy.npz`, `bop_dali.npz` — raw sampled data
  (poses, K, projected keypoints, channel stats). Kept so you can
  re-aggregate without re-sampling.
- `{SWEEP_ROOT}/{obj}/report/comparison_report.txt` — per-panel numbers +
  verdict for rotation, translation, image-plane center, per-keypoint 2D,
  photometric, and the projection ↔ heatmap-peak invariant.
- `{SWEEP_ROOT}/{obj}/report/*.png` — histograms, sphere-coverage scatter,
  per-keypoint 2D density.

**Interpretation notes**:

- `||t||` is auto-scaled to mm before comparison (legacy poses are in metres
  from `_RT.pkl`, BOP is in mm from `cam_t_m2c`).
- Both renderers use a fixed camera-object distance, so `||t||` and the
  image-plane center will show std=0. The script marks these as
  `a-match (both constant)` when values agree, rather than reporting
  spurious `c-gap`.
- The projection ↔ heatmap-peak invariant is expected to FAIL on legacy
  (median ~85 px) because `random_rotation_and_resize` and
  `random_translation` warp the heatmap without updating the pose. This is
  by design per CLAUDE.md, not a bug. BOP DALI should be sub-pixel.
- Photometric coverage is expected to be wider on BOP DALI (HSV + blur +
  brightness/contrast) than legacy (`convertScaleAbs` only). Wider is
  fine — it means BOP DALI is a superset of legacy's photometric coverage.

**Two-step usage**: if you already have per-loader NPZs, you can re-run
just the comparison without re-sampling:

```bash
python tests/loader_comparison/compare_distributions.py \
  --legacy   results/loader_compare/{ts}/obj1/legacy.npz \
  --bop_dali results/loader_compare/{ts}/obj1/bop_dali.npz \
  --out_dir  results/loader_compare/{ts}/obj1/report
```

---

## Notes on data prerequisites

Both suites depend on data layouts documented in `CLAUDE.md` / `RT_Less.md`.
Concretely:

- `pecp_variance` needs the per-object BOP test roots at
  `data/RTLESS_BOP/{obj}/` (produced by `scripts/rtless_test_to_bop.py`)
  and paper checkpoints at `model/paper_checkpoints/{obj}/150.pkl`.
- `loader_comparison` needs `data/RTLESS_BOP/train_pbr/{obj_id:06d}/` (BOP
  DALI side), the legacy render tree at
  `${CONTOURPOSE_ROOT}/Train_Scenes/Real/{renders,gtEdge}/{obj}/`, and
  SUN2012 images at `${CONTOURPOSE_ROOT}/SUN2012pascalformat/JPEGImages/`.

Missing paths surface as `[SKIP]` lines in the sweep log with the missing
directory named, so failures are easy to diagnose.
