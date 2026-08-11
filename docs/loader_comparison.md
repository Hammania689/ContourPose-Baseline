# Loader comparison: legacy `MyDataset` vs BOP DALI

Companion reference for the two training-time dataloaders in this repo. The
mechanical distribution comparison lives in `tests/loader_comparison/` (samples
both loaders, dumps CSVs + histograms, assigns per-quantity coverage verdicts);
this doc is the human-readable side-by-side of *what each loader does and why*.

Cross-references:
- `RT_Less.md` — authors' dataset structure and unit conventions.
- `tests/loader_comparison/README.md` — how to run the characterization sweep.
- `CLAUDE.md` — top-level architecture and augmentation-order gotchas.

---

## Why two loaders

| Loader | File | Purpose |
|---|---|---|
| `MyDataset` | `dataset/Dataset.py` | Original authors' CPU loader for the **RT-Less native** layout (real captures + Blender renders under `{root}/train/`). Used by `main.py` and `eval.py`. |
| `BOPDALIPipeline` | `dataset/BOPDALIDataset.py` | New DALI/GPU loader for the **BOP-format** layout. Used by `train_bop.py`, `test.py`, and the BOP evaluators. Enables PBR synthetic data (BlenderProc2) and shared-format datasets. |

Both feed `network.ContourPose` — same tensor contract (`img`, `heatmap`,
`edge`, `K`, `pose`). Only the front end and augmentation pipeline differ.

---

## Side-by-side

| Aspect | Legacy `MyDataset` | BOP DALI `BOPDALIPipeline` |
|---|---|---|
| **On-disk layout** | `{root}/train/{cls}/photo_cut,mask,gtEdge` (real) + `{root}/train/renders/{cls}/` (synth) | `{data_dir}/rgb, mask (or mask_visib), edges` + `scene_gt.json`, `scene_camera.json` (flat, one scene per obj) |
| **Metadata format** | YAML: `gt.yml`, `Intrinsic.yml` (lists-of-lists per frame; `index` = slot within frame) | JSON: BOP standard (`cam_R_m2c`, `cam_t_m2c`, `cam_K`) |
| **Camera intrinsic (real)** | Loaded from `Intrinsic.yml` (fx≈1172) | Loaded from `scene_camera.json` per-frame |
| **Camera intrinsic (synth)** | Hardcoded Blender K: `fx=fy=700, cx=320, cy=240` (`Dataset.py:17`) — never read from disk | Read from `scene_camera.json` — whatever BlenderProc2 wrote |
| **Translation units** | **metres** (matches PLY units; ×1000 → mm at eval time) | **millimetres** (BOP convention: `cam_t_m2c` is mm) |
| **Keypoint units** | Read as-is from `keypoints/{cls}.txt` — treated as metres in-code | Prefers `keypoints/{cls}_mm.txt`. Legacy metre files auto-multiplied ×1000 with a warning. Runtime assertion enforces max|xyz|>1.0 as a mm-sanity guard. |
| **Image size** | 480×640, resized only via augmentation | 480×640 via `fn.resize` in the pipeline graph |
| **Real+synth mix** | Yes — real captures + Blender renders combined in one epoch, distinguished by `path["type"] == "true"` for pipeline branching | No — one loader = one scene dir. Real+synth mix would require running two loaders and interleaving. Current use is PBR-only. |
| **Compute location** | CPU (Python + NumPy + OpenCV) | GPU (DALI graph); external-source callbacks for K, pose, and CPU-generated heatmaps |
| **Heatmap generation** | Regenerated from pose after each augment step (`get_heatmap`, calls `heatmap.generate_heatmap`) | Either precomputed on disk (PNG stacked or NPY) or generated CPU-side in `_generate_heatmap_cpu` and pushed to GPU |
| **Edge/contour source** | `gtEdge/{idx}.png` for real, `renders/gtEdge/{cls}/{idx}.png` for synth | Precomputed `edges/{frame:06d}.png` (preferred) → mask-Laplacian fallback → image-Laplacian fallback |
| **Background source** | `SUN2012pascalformat/JPEGImages`, path list cached to `dataset/bg_imgs.npy` (invalidated when paths no longer resolve or `sun_path` prefix changes) | Same SUN2012 dir; shuffled at pipeline init with the same seed used by DALI |
| **Batch source** | `torch.utils.data.DataLoader` wrapping `Dataset` | `DALIGenericIterator` yielding tensor dicts already on GPU |

---

## Augmentation pipelines

### Legacy `MyDataset.__getitem__` (order matters — pose is only updated in `augment`)

1. `augment` — in-plane rotation ±30° via `rotate_img`; **updates `pose[:3,:3]`** to keep pose consistent with the rotated image.
2. `get_heatmap` — regenerated from the post-`augment` pose.
3. `random_rotation_and_resize` — geometric aug on img/mask/edge/heatmap. **Pose is NOT updated.** Heatmap is warped along with the image so it stays visually consistent, but the pose-to-heatmap invariant is broken from this point on. (Same for step 4.)
4. `random_translation` — pixel shift of img/mask/edge/heatmap; pose again not updated.
5. `random_background` — composites `SUN2012` background using `mask`.
6. Photometric: `alpha ~ U(0.8, 1.2)`, `beta ~ U(-5, 5)` via `cv2.convertScaleAbs`.

**Consequence**: on legacy the `pose → project(keypoints) → heatmap-peak` invariant only holds through step 2. After step 3 the pose is stale relative to the image; the network learns from the *warped heatmap*, not from projecting keypoints with the returned pose. `tests/loader_comparison/` reports this as a per-loader invariant check and expects it to fail on legacy by several pixels.

### BOP DALI `define_graph` (all GPU, no pose mutation)

1. RGB decode → resize.
2. Mask decode → resize NN.
3. Edge: precomputed decode → resize NN. Fallback: Laplacian on mask (or on grayscale RGB if no mask).
4. **Background composite** (if `background_dir` set): `bg * (1 - mask) + img * mask`. Skipped entirely when `bg_files` is empty — see gotcha below.
5. **HSV jitter** (60% probability): `hue ~ U(-15, 15)`, `saturation ~ U(0.7, 1.3)`.
6. **Brightness / contrast** (always): both `~ U(0.8, 1.2)`. Identity is at 1.0, so with p=1.0 the distribution spans no-op.
7. **Gaussian blur** (40% probability): `sigma ~ U(0.5, 1.5)`.
8. Cast → normalize with `mean=[0.419, 0.427, 0.424]`, `std=[0.184, 0.206, 0.197]` (dataset-specific, not ImageNet).
9. Transpose → CHW.

**Consequence**: DALI never touches the pose or the projected keypoints, so `pose → project → heatmap-peak` remains sub-pixel accurate for the entire pipeline. `tests/loader_comparison/` expects this to pass.

---

## Determinism

| Aspect | Legacy | BOP DALI |
|---|---|---|
| Sample order | `DataLoader(shuffle=...)` at the wrapper level | `random.shuffle(self.samples)` in `__init__` iff `file_indices is None`, else input order preserved. `fn.readers.file(..., random_shuffle=False)` for rgb/mask/edge |
| Augmentation seeding | Python `random` — not centrally seeded | `Pipeline(seed=...)` seeds every `fn.random.*` node in the graph |
| Background selection | `random.choice(self.bg_imgs)` in `random_background` | `fn.readers.file(bg_files, random_shuffle=True)` — DALI-seeded |
| Val loader use | N/A | `fixed_val_loader` uses `seed=1111` (stable per epoch); `random_val_loader` uses time-based seed (varies per epoch, intended for wandb viz variety) |

---

## Val split, checkpointing, and viz (BOP path only)

`dataset/data_utils.py:create_bop_validation_setup` builds three loaders from
one per-object scene dir:

- `train_loader`  — 80% split, augs + bg on, shuffled.
- `fixed_val_loader`  — 20% split, `seed=1111`, deterministic. Used for the val metric.
- `random_val_loader` — same 20%, time-based seed. Used only to pull a fresh viz batch each epoch.

All three now forward `background_dir` (as of the `dataset/data_utils.py` fix
in this commit series). Previously the val loaders hardcoded
`background_dir=None`, which produced black-background thumbnails in wandb
even when the train loader had compositing enabled. See the fix commit and
`scripts/visualize_bop_train_batch.py` for the sanity check that catches
this class of drift.

`train_bop.py` saves two files per run under `model/{class_type}/{run_name}/`:
- `best_model.pkl` — overwritten whenever fixed-val `selection_score` improves.
- `model.pkl` — overwritten every 10 epochs (latest snapshot only). **No per-epoch history.** If per-epoch checkpoints are needed later, `save_checkpoint` in `network/contourpose.py` needs a filename change.

---

## Known gotchas

- **Legacy pose-heatmap invariant breaks after step 3.** Not a bug — the network trains against the warped heatmap. But any code that assumes `project(kp, pose_out, K_out) == heatmap_argmax` will fail on legacy samples. BOP DALI keeps the invariant.
- **BOP background compositing silently no-ops** when `background_dir` is `None`, points at a non-existent path, or points at a directory with 0 `.jpg`/`.png` files. Pipeline prints `[BOP DALI] Loaded N background images` at init — if that line is absent, `bg_files=[]` and no compositing runs.
- **BOP keypoint units.** `keypoints/*.txt` are metre-scale (legacy inherited from `MyDataset`). BOP DALI auto-converts with a warning; the mm-native path is `keypoints/*_mm.txt`. Prefer the mm files when converting new datasets to avoid the runtime multiply.
- **Legacy bg cache.** `dataset/bg_imgs.npy` is invalidated when `sun_path` changes or when cached paths stop resolving — safe across container mount changes. But if you delete SUN2012 and re-download to a new path, delete `bg_imgs.npy` too or trust the resolution guard.
- **DALI iterator order.** With `random_shuffle=False` and `file_indices` supplied, the DALI pipeline yields samples in `pipeline.samples` order — so `visualize_bop_train_batch.py` can zip `batch[i]` with `pipeline.samples[i]` safely. Do NOT assume this on the val loaders without `file_indices` set, since `random.shuffle(self.samples)` runs in `__init__`.

---

## When to use which

- **Reproducing paper numbers** on RT-Less native → legacy `MyDataset` via `main.py` / `eval.py`. This is what `eval_rtless_authors.sh` runs.
- **Training on BOP-format PBR data** (RT-Less BOP, T-LESS, ROBI, etc.) → `train_bop.py` + `configs/contourpose_bop.gin`. This is what `train_bop_4objs.sh` and `train_all_objects.sh` use.
- **BOP-format eval** (matched to the BOP training path) → `test.py` + `eval_rtless_bop.sh`.
- **New dataset** → convert to BOP format (see `scripts/rtless_test_to_bop.py` for an example), use the BOP loader. Don't add a new legacy-style branch.
