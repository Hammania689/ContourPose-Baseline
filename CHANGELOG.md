## fix(data_utils): forward background_dir to BOP val loaders

### Fixed
- `dataset/data_utils.py:create_bop_validation_setup`: both `fixed_val_loader`
  and `random_val_loader` now forward `getattr(args, 'background_dir', None)`
  instead of hardcoding `background_dir=None`. Previously the wandb viz
  panel showed black-background thumbnails even when the train loader was
  compositing SUN2012 correctly, because the val loaders that produced the
  viz batches skipped compositing entirely. `fixed_val_loader` keeps
  `seed=1111` so the composited backgrounds are stable across epochs; only
  `random_val_loader` (time-seeded, viz-only) varies its bg per epoch.

### Added
- `scripts/visualize_bop_train_batch.py`: sanity check that instantiates
  the *training* DALI loader via `create_bop_validation_setup` (same code
  path `train_bop.py` uses) and writes a per-sample 4-panel grid:
  raw PBR from disk / mask overlay / DALI output (post-composite +
  augmented) / |Δ| pixel diff. Prints diagnostics on `bg_files` count,
  first-sample paths, and per-sample mean|Δ| so the two failure modes
  ("bg_dir path is wrong" vs "augs are no-ops") are visible before opening
  any PNG. Companion to `scripts/visualize_bop_test_batch.py`.
- `docs/loader_comparison.md`: side-by-side reference for the legacy
  `MyDataset` and BOP DALI `BOPDALIPipeline` loaders — file layouts, unit
  conventions, augmentation pipelines with probabilities, determinism
  guarantees, known gotchas (pose-heatmap invariant break on legacy,
  silent bg no-op on BOP, keypoint unit auto-conversion), and a
  when-to-use-which guide. The mechanical distribution comparison stays
  in `tests/loader_comparison/`; this doc is the human-readable companion.
- `RT_Less.md`: cross-reference pointing at `docs/loader_comparison.md`
  for loader-specific engineering details.

---

## feat(scripts): BOP training driver for obj2/6/13/18

### Added
- `train_bop_4objs.sh`: 4-object driver script mirroring
  `train_all_objects.sh` but restricted to `obj2, obj6, obj13, obj18`.
  Points `BOP_ROOT` at the in-container path
  `/contourpose-baseline-4090/data/RTLESS_BOP` (dataset root; `train_bop.py`
  appends `train_pbr/{obj_id:06d}` internally), forwards
  `--background_dir /data/SUN2012pascalformat/JPEGImages`, and uses the
  paper-default gin config. Includes a commented-out smoke-run block
  (`main.epochs = 2` + tight val/viz intervals) for quick validation
  before committing to the full 150-epoch schedule.

---

## d03f6e6 - Add latex table generation from eval results.csv

### Added
- `scripts/make_latex_table.ipynb` + `scripts/make_latex_table_tmp.py`:
  notebook + throwaway helper for producing paper-formatted LaTeX tables
  from aggregated `results.csv` (both legacy and BOP eval pipelines emit
  compatible schemas). Kept for reproducibility of the paper submission.

---

## b625394 - Add visualization and inspection utilities

### Added
- `scripts/visualize_bop_test_batch.py`: pulls one batch from the BOP test
  DALI loader and renders per-sample grids (RGB / mask / heatmap overlay /
  GT-pose mesh projection). Confirms K, pose, keypoints, and heatmap peaks
  are all self-consistent after mm-scale unit fixes.
- `scripts/visualize_annotations.py`: annotation viz for the RT-Less test set.
- `scripts/visualize_inference.py`: overlays predicted pose on RGB, saves
  per-frame PNGs from a completed eval run's CSV.
- `scripts/inspect_bop_converted_data.py`: prints scene_gt.json /
  scene_camera.json summaries for a per-object BOP root; sanity check for
  `rtless_test_to_bop.py` output.

---

## 350622c - Add tests/README.md covering both characterization suites

### Added
- `tests/README.md`: index describing the `pecp_variance/` and
  `loader_comparison/` suites — what question each one answers, run
  recipes, output layouts, interpretation notes.

---

## 38a766e - Add PECP variance characterization suite

### Added
- `tests/pecp_variance/run_sweep.sh`: harness that runs `test.py` for
  NUM_SEEDS seeds × 10 objects (plus an obj21 excl-scene-29 variant), each
  with `--eval_seed {k}`. Writes to
  `results/rtless_bop_variance/{ts}_pecp/seed{k}/{obj}/`.
- `tests/pecp_variance/aggregate.py`: consumes the sweep output and writes
  `variance_raw.csv` (per-run, per-scene — saved so CIs can be recomputed
  without re-running), `variance_summary.csv` (mean/std/min/max over
  seeds), and `variance_report.txt` (sanity checks: seed diversity,
  reference numbers inside range, per-object std ranking).

Depends on the `--eval_seed` hook added to `test.py` in commit f5ea8d8.

---

## fa4adbc - Add legacy vs BOP DALI loader characterization suite

### Added
- `tests/loader_comparison/`: characterization suite that draws N samples
  from both loaders independently and compares distributions (rotation,
  translation, per-keypoint 2D, photometric). Assigns a coverage verdict
  per quantity: `a-match` / `b-BOP-covers-legacy` / `c-gap`. Also runs a
  per-loader invariant check (projection ↔ heatmap peak) — expected to
  fail on legacy due to the 2D-warp decoupling documented in CLAUDE.md,
  expected sub-pixel on BOP DALI.
  - `sample_poses.py`: samples one loader, dumps NPZ.
  - `compare_distributions.py`: reads two NPZs, writes report + histograms.
  - `run_sweep.sh`: orchestrates a 10-object sweep with matched
    configuration (both loaders receive SUN2012 backgrounds at load time).

### Changed
- `dataset/Dataset.py` (`MyDataset`): `render_dir` / `render_edge_dir` /
  `sun_path` kwargs to override the legacy layout so the harness can point
  at renders that don't live under `{root}/train/`. Constructor guards on
  `Intrinsic.yml` / `gt.yml` / `photo_cut/` so renders-only datasets work.
  `get_bg_imgs` invalidates stale `bg_imgs.npy` caches (paths that no
  longer resolve or don't share the current `sun_path` prefix) instead of
  blindly reusing them across container mounts.

---

## d523ba5 - Add BOP-format eval aggregator matching legacy accounting

### Added
- `scripts/aggregate_bop_results.py`: post-processes `test.py`'s per-object
  detailed CSVs into `results.csv` + `summary.txt` matching the legacy
  `results/rtless/authors_checkpoints/{ts}_pecp/` schema. Uses the same
  legacy accounting convention: per-axis errors are only averaged over
  frames where ADD passed (`eval.py:calculate_tra_and_rot` skips failed
  frames), and NaN values are masked via `nanmean`. Auto-detects
  `rgb_detailed.csv` vs `masked_detailed.csv` based on which one `test.py`
  produced.

The related `test.py --use_masks` default flip (from `"true"` to `"false"`
so the DALI test loader reads `rgb/` as-is, matching the legacy eval's
input distribution) is part of commit f5ea8d8.

---

## 4a3e147 - Enforce mm-scale keypoints in BOP DALI + eval paths

The legacy `keypoints/{obj}.txt` and `Valid3D/{obj}.txt` files are in
metres, but BOP `cam_t_m2c` is in mm. Silently mixing the two produced
silently wrong PnP results.

### Added
- `scripts/convert_keypoints_to_mm.py`: one-shot writer for
  `keypoints/{obj}_mm.txt` and `Valid3D/{obj}_mm.txt` files (multiplies
  repo-root metre-scale files by 1000). Idempotent.
- `scripts/rtless_test_to_bop.py`: RT-Less native → BOP-format converter
  for the test split. Writes mm-scale keypoints and Valid3D per-object as
  part of the conversion so downstream loaders see self-consistent data.

### Changed
- `dataset/BOPDALIDataset.py`, `dataset/BOPTestDALIDataset.py`,
  `network/contourpose.py`: keypoint and Valid3D loaders prefer
  `{obj}_mm.txt` over the legacy `{obj}.txt`, and assert `max |xyz| > 1.0`
  on load so silent metre-scale files can't reach PnP by accident. Error
  messages point at `convert_keypoints_to_mm.py`.
- `eval_spectra_pose.py::_init_bop_format`: same mm-scale enforcement for
  keypoints and Valid3D. CAD-file search falls back to repo-root `cad/`
  and auto-scales metre-scale PLYs to mm on load to match BOP
  `cam_t_m2c` and the mm-scale keypoints.

---

## f5ea8d8 - Add BOP eval pipeline (loaders, evaluator, orchestrators)

### Added
- `test.py` / `test_legacy_eval.py`: DALI-based test drivers. `test.py`
  exposes `--eval_seed` for the PECP variance suite (see commit 38a766e)
  and defaults `--use_masks false` to match the legacy input distribution
  (see commit d523ba5).
- `eval_spectra_pose.py`: evaluator class supporting both BOP and legacy
  formats, with symmetry-aware rotation error and per-instance CSV output.
- `dataset/BOPTestDALIDataset.py`: multi-scene, multi-sensor test loader.
- `dataset/bop_config.py`: `models_info.json` parser (diameters, symmetries).
- `eval_rtless_bop.sh`, `eval_all_rtless.sh`, `eval_all_rtless_legacy.sh`:
  sweep orchestrators. `eval_rtless_bop.sh` writes to a timestamped run
  dir and invokes `scripts/aggregate_bop_results.py` at the end.

---

## 585bb8c - Add .gitignore entries for env-local files, caches, and outputs

### Added
- `.gitignore`: NFS lock artifacts, `dataset/bg_imgs.npy` cache,
  `results/` sweep outputs, `.python-version`, `docker/start_container_4090`,
  `utils/utils.py_`, and `.ipynb_checkpoints/`.

---

## edd14bc - Add eval_legacy_all_scenes.py, eval_rtless_authors.sh, and RT_Less.md

### Added
- `eval_legacy_all_scenes.py`: Evaluation driver that runs `eval.py`'s evaluator across
  all scenes for a given object and aggregates metrics. Produced the official RT-Less PECP
  results (`260609_0543_pecp`). Supports single-object and all-objects modes, optional PECP,
  and saves `results.csv`, `run.log`, and `summary.txt` per run.
- `eval_rtless_authors.sh`: Canonical invocation record for the official RT-Less evaluation
  (renamed from `_tmp_rt-less_paper_eval.sh`). Contains both the no-PECP and PECP commands
  against the author's checkpoints and original ContourPose dataset layout.
- `RT_Less.md`: Dataset structure reference and evaluation findings for the RT-Less baseline,
  including the object→scene→index table, missing scenes, and analysis of obj21 scene 29.

---

## fd28ae3 - Evaluator fixes, analysis scripts, and rotation error utilities

### Fixed
- `eval.py` (`evaluate`): Was returning `None` silently; now returns a metrics dict
  (`proj_2d`, `add`, `add_raw`, translation/rotation error lists). Any caller that
  used the return value was broken before this change.
- `eval.py` (`evaluate`): PECP was hardcoded off (the `calculate_metric_PECP` call was
  commented out). Dispatch is now controlled by `args.use_pecp`; non-PECP path is the
  default, matching the original `main.py` behaviour.
- `main.py`: `model_path` was hardcoded to `model/{class_type}`; now uses a required
  `--model_dir` argument so arbitrary checkpoint directories can be targeted.
- `network/contourpose.py` (`_init_geo_info`): Crashed when `data_root` or `class_type`
  was `None` (e.g. when the network is instantiated from the legacy `eval.py` path which
  does not supply those). Now returns early in that case.
- `network/contourpose.py` (`_init_geo_info`): Keypoints were scaled by `* 1000` (m → mm)
  after loading, producing wrong units for PnP. Removed; keypoints are now loaded in their
  native metre units, consistent with the mesh and pose tensors.
- `dataset/Dataset.py`: `yaml.load()` → `yaml.safe_load()` (train and test paths).

### Added
- `eval.py`: `add_raw` list captures the continuous per-frame ADD distance (same units as
  mesh pts) alongside the existing boolean `add` list. Enables per-frame distribution
  analysis without re-implementing ADD.
- `utils/utils.py`: `geodesic_rotation_error`, `_optimal_continuous_symmetry_rotation`,
  and `min_symmetry_rotation_error` — utilities for computing rotation error under discrete
  and continuous object symmetries.
- `docs/pecp_notes.md`: Documents PECP stochasticity (unseeded `random.choice` in the 400-
  iteration subset loop), observed ~1 pp run-to-run variance, and how to seed for
  reproducibility.
- `scripts/check_gt_scene29.py`: Verifies GT soundness for obj21 scene 29 across all 416
  frames (slot-0 obj_id consistency, centroid in-bounds, CAD mask overlap).
- `scripts/add_distribution_scene29.py`: Produces per-frame ADD distribution figures
  (log-scale histogram PDF + per-frame scatter) for obj21 scene 29, using the evaluator
  directly so numbers match the official PECP run exactly.

---

## fd28ae3 - Legacy eval pipeline and LaTeX table notebook

### Added
- `test_legacy_eval.py`: Duplicate of `test.py` that uses `eval.py` (the original
  ContourPose evaluator) instead of `eval_spectra_pose.py`. Intended for apples-to-apples
  comparison with prior `main.py`-based experiments. Key differences from `test.py`:
  imports `from eval import evaluator`, drops the `samples_metadata` argument (plain
  `eval.py` does not accept it), and guards the summary print against the `None` return
  that `eval.py`'s `evaluate()` produces.
- `eval_all_rtless_legacy.sh`: Companion script to `eval_all_rtless.sh` that drives
  `test_legacy_eval.py` across all 10 objects using the original ContourPose dataset
  layout (`data/ContourPose_Original/` by default, overridable via `DATA_PATH`).
  PECP is disabled by default (`--no_pecp`) to match the original `main.py` behaviour;
  pass `--pecp` to enable it. Scene/index assignments are derived from `sceneObjs.yml`;
  several objects appear in multiple scenes — see the comments in the script for
  alternatives.
- `scripts/make_latex_table.ipynb`: Jupyter notebook that reads per-object
  `masked_detailed.csv` files from `results/rtless_paper/` and generates a LaTeX
  comparison table (ADD(-S) % reproduced vs. paper-reported values). Bold-faces the
  better value per row and writes output to `results/table_add.tex`.

### Fixed
- `README.md`: Object index mapping table had `obj18` listed twice (for both paper obj7
  and obj8). Corrected paper obj8 → code `obj6`, which is the only code object absent
  from the original mapping.
- `dataset/data_utils.py` (`create_test_loader`): Removed `data_root` keyword argument
  passed to `MyDataset.__init__`, which does not accept that parameter.

---

## f4378f0 - Add train_all_objects.sh

### Added
- `train_all_objects.sh`: Shell script that runs `train_bop.py` sequentially for all
  10 trainable objects. Accepts gin parameter overrides via a `GIN_PARAMS` array and
  configurable `BOP_ROOT` and `BACKGROUND_DIR` paths at the top of the file.

### Rationale
- Training all objects requires launching the same command 10 times with different
  `--class_type` and `--obj_id` flags. The script centralises that logic and makes
  full-dataset training a single invocation.

---

## df910c3 - Add Docker environment

### Added
- `docker/`: Dockerfiles and build/start scripts for two environments — a generic
  (`contourpose.Dockerfile`) and an RTX 4090-specific (`contourpose-4090.Dockerfile`)
  variant. Post-install scripts handle environment setup inside the container.
- `requirements-docker-py39.txt`, `requirements-docker-py39-4090.txt`: Pinned dependency
  lists for each environment. `bop_toolkit` is installed directly from upstream GitHub
  (`git+https://github.com/thodan/bop_toolkit.git`) rather than from a vendored local
  copy.

### Rationale
- Separate 4090 Dockerfile allows CUDA/driver version targeting without affecting the
  generic build.
- Direct GitHub install of `bop_toolkit` removes the need to vendor third-party source
  in this repository.

---

## 1d2f982 - Refactor network.py into network/ package

### Changed
- `network.py`: Deleted.
- `network/__init__.py`, `network/contourpose.py`, `network/resnet.py`: New package
  replacing the flat `network.py` module. All import sites updated accordingly.

### Rationale
- The original `network.py` combined the model definition, training interface, and ResNet
  backbone in a single file. Splitting into a package makes each concern independently
  navigable and easier to diff against the SpectraPose fork.

---

## 69e09de - Add BOP training stack with StepLR scheduling

### Added
- `train_bop.py`: BOP-enabled training script derived from upstream `main.py`. Adds DALI
  dataloader, wandb logging, StepLR scheduling, and gin config support. Original `main.py`
  preserved unchanged.
- `configs/contourpose_bop.gin`: Baseline hyperparameters matching the upstream ContourPose
  paper (batch_size=16, lr=0.1, step_size=20, gamma=0.5, epochs=150).
- `dataset/BOPDALIDataset.py`: DALI-based dataloader for BOP-format datasets. Handles
  background compositing, edge map loading, heatmap generation, and photometric augmentation.
- `dataset/DALIDataset.py`: DALI dataloader for the original LINEMOD-format dataset.
- `dataset/data_utils.py`: Data utilities including `create_bop_validation_setup`, which
  builds an 80/20 train/val split. Validation loaders intentionally receive no
  `background_dir` — background compositing is a training-only augmentation, and including
  it in validation wastes GPU memory (nvJPEG allocates per-pipeline) while adding noise to
  the validation signal.
- `utils/visualization.py`: Batch visualization utilities for wandb logging.
- `utils/utils.py`: Added `load_camera_intrinsics` and `get_K_override` — camera intrinsics
  helpers required by `network/contourpose.py`.

### Changed
- `network/contourpose.py`: Replaced `CosineAnnealingWarmRestarts` with
  `torch.optim.lr_scheduler.StepLR(step_size=lr_step_size, gamma=lr_gamma)`.
  Constructor params `T_0`, `T_mult`, `eta_min` removed; `lr_step_size=20` and
  `lr_gamma=0.5` added. Per-batch `lr_sched.step()` call removed from
  `optimize_params()` — StepLR steps once per epoch in `train_bop.py`.
- `train_bop.py`: `epochs_per_cycle` param removed; `lr_step_size` and `lr_gamma` added
  to `main()` and forwarded to the model. `model_module.lr_sched.step()` called after
  each epoch. Stale `compute_cosine_annealing_T0` call removed.
- `configs/contourpose_bop.gin`: `epochs_per_cycle` removed; `lr_step_size = 20` and
  `lr_gamma = 0.5` added.

### Rationale
- DALI replaces PyTorch DataLoader for significantly faster data throughput on GPU.
- gin config replaces hardcoded hyperparameters, making experiments reproducible and
  diff-able.
- wandb replaces print-based loss logging for persistent experiment tracking.
- StepLR matches the original `adjust_learning_rate()` math exactly:
  `lr = init_lr × gamma^(epoch // step_size)` with step_size=20, gamma=0.5.
  Cosine annealing was a SpectraPose-only addition; this baseline intentionally matches
  the upstream ContourPose paper's training schedule.

---

## c7d257b 239fa00 - Add .gitignore

### Added
- `.gitignore`: Excludes trained model checkpoints (`model/`), experiment logs (`wandb/`),
  Python bytecode (`__pycache__/`, `*.pyc`), editor swap files (`*.swp`), local Claude
  Code config (`.claude/`), diff/patch reference files (`*.diff`, `*.patch`),
  AI assistant instructions (`CLAUDE.md`), and original-file backups (`*_og.py`).

### Rationale
- None of the ignored paths contain source code — they are either generated artifacts,
  large binaries, or local-only configuration that would pollute the repository history.
