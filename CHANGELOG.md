## April 27 — Add BOP training stack

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

### Rationale
- DALI replaces PyTorch DataLoader for significantly faster data throughput on GPU.
- gin config replaces hardcoded hyperparameters, making experiments reproducible and
  diff-able.
- wandb replaces print-based loss logging for persistent experiment tracking.

---

## April 27 — Replace cosine annealing with StepLR

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
- StepLR matches the original `adjust_learning_rate()` math exactly:
  `lr = init_lr × gamma^(epoch // step_size)` with step_size=20, gamma=0.5.
- Cosine annealing was a SpectraPose-only addition; this baseline intentionally matches
  the upstream ContourPose paper's training schedule.

---

## April 30 — Refactor network.py into network/ package

### Changed
- `network.py`: Deleted.
- `network/__init__.py`, `network/contourpose.py`, `network/resnet.py`: New package
  replacing the flat `network.py` module. All import sites updated accordingly.

### Rationale
- The original `network.py` combined the model definition, training interface, and ResNet
  backbone in a single file. Splitting into a package makes each concern independently
  navigable and easier to diff against the SpectraPose fork.

---

## April 30 — Add Docker environment

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

## April 30 — Add train_all_objects.sh

### Added
- `train_all_objects.sh`: Shell script that runs `train_bop.py` sequentially for all
  10 trainable objects. Accepts gin parameter overrides via a `GIN_PARAMS` array and
  configurable `BOP_ROOT` and `BACKGROUND_DIR` paths at the top of the file.

### Rationale
- Training all objects requires launching the same command 10 times with different
  `--class_type` and `--obj_id` flags. The script centralises that logic and makes
  full-dataset training a single invocation.
