## Evaluator fixes, analysis scripts, and rotation error utilities
`fd28ae3`

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

## Legacy eval pipeline and LaTeX table notebook
`fd28ae3`

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

## Add train_all_objects.sh
`f4378f0`

### Added
- `train_all_objects.sh`: Shell script that runs `train_bop.py` sequentially for all
  10 trainable objects. Accepts gin parameter overrides via a `GIN_PARAMS` array and
  configurable `BOP_ROOT` and `BACKGROUND_DIR` paths at the top of the file.

### Rationale
- Training all objects requires launching the same command 10 times with different
  `--class_type` and `--obj_id` flags. The script centralises that logic and makes
  full-dataset training a single invocation.

---

## Add Docker environment
`df910c3`

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

## Refactor network.py into network/ package
`1d2f982`

### Changed
- `network.py`: Deleted.
- `network/__init__.py`, `network/contourpose.py`, `network/resnet.py`: New package
  replacing the flat `network.py` module. All import sites updated accordingly.

### Rationale
- The original `network.py` combined the model definition, training interface, and ResNet
  backbone in a single file. Splitting into a package makes each concern independently
  navigable and easier to diff against the SpectraPose fork.

---

## Add BOP training stack with StepLR scheduling
`69e09de`

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

## Add .gitignore
`c7d257b` `239fa00`

### Added
- `.gitignore`: Excludes trained model checkpoints (`model/`), experiment logs (`wandb/`),
  Python bytecode (`__pycache__/`, `*.pyc`), editor swap files (`*.swp`), local Claude
  Code config (`.claude/`), diff/patch reference files (`*.diff`, `*.patch`),
  AI assistant instructions (`CLAUDE.md`), and original-file backups (`*_og.py`).

### Rationale
- None of the ignored paths contain source code — they are either generated artifacts,
  large binaries, or local-only configuration that would pollute the repository history.
