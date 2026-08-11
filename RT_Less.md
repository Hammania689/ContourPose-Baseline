# RT-Less Dataset Reference

This document describes the structure, conventions, and evaluation details of the
**RT-Less 10-parts** dataset, used in:

> *ContourPose: A monocular 6D pose estimation method for reflective texture-less metal parts*
> (IEEE Transactions on Robotics, Vol. 39, No. 5, October 2023)

Dataset root on this machine: `/data/Datasets/ContourPose/`
This path is passed to `MyDataset` and all eval scripts via `--data_path`.

---

## Overview

The dataset contains **10 reflective, texture-less metal parts** (referred to as objects)
collected for monocular 6D pose estimation. The 10 trainable objects are:
`obj1, obj2, obj3, obj6, obj7, obj13, obj16, obj18, obj21, obj32`.

Note: the object IDs in the code do not match the paper's numbering. The mapping is:

| Paper | obj1 | obj2 | obj3 | obj4 | obj5 | obj6 | obj7 | obj8 | obj9 | obj10 |
|-------|------|------|------|------|------|------|------|------|------|-------|
| Code  | obj1 | obj2 | obj3 | obj7 | obj13| obj16| obj18| obj6 | obj21| obj32 |

Per object: **660 real training images** + **6,600 OpenGL rendered synthetic images**.

The test set is organised into **16 scene directories**, each containing 416 frames with
3–5 objects co-present in the scene. Not every object appears in every scene — each object
has been placed in 1–3 test scenes by dataset design.

---

## Training Data

Training data lives under `{root}/train/` and is split into real captures and synthetic renders.

> For loader-side details (RT-Less native `MyDataset` vs BOP DALI
> `BOPDALIPipeline` — units, augmentation pipelines, determinism guarantees,
> known gotchas), see `docs/loader_comparison.md`.

### Real images — `{root}/train/{cls}/`

```
photo_cut/{idx}.png     — 640×480 RGB crop, object foreground on black background
mask/{idx}.png          — binary object mask
gtEdge/{idx}.png        — Canny-derived contour ground truth (binary, 640×480)
Intrinsic.yml           — per-frame 3×3 camera intrinsic matrix (fx~1172, real camera)
gt.yml                  — per-frame pose: {m2c_R: 3×3, m2c_T: 1×3}, translation in meters
```

### Synthetic renders — `{root}/train/renders/{cls}/` and `renders/gtEdge/{cls}/`

```
{idx}.jpg                        — RGB render
{idx}_depth.png                  — binary silhouette mask (used as object mask)
{idx}_RT.pkl                     — {'K': 3×3 float32, 'RT': 3×4 float64}, translation in meters
renders/gtEdge/{cls}/{idx}.png   — edge ground truth for the render
```

The Blender camera intrinsic is **hardcoded** in `Dataset.py:17` as
**fx=fy=700, cx=320, cy=240** and is constant across all synthetic images.
It is never read from disk. This differs from the real camera intrinsic (fx~1172),
so be careful when mixing real and synthetic data paths.

### Background augmentation

During training, object crops are composited onto random natural backgrounds drawn from:

```
{root}/SUN2012pascalformat/JPEGImages/
```

On the first run, all image paths from this directory are cached to `dataset/bg_imgs.npy`
to avoid repeated directory scans.

---

## Test Data

### File structure — `{root}/test/scene{N}/`

The 16 on-disk scene IDs are: **1, 3, 4, 5, 8, 10, 13, 18, 21, 24, 25, 26, 29, 30, 31, 32**.
Each scene contains 416 frames (hardcoded in `Dataset.py:99`) at 640×480.

```
photo_cut/{idx}_{objID}.png  — 640×480 RGB scene image (same file, duplicated per object in scene)
mask/{idx}_{objID}.png       — per-object mask (RGB, pixel values in {0, 64, 128, 191, 255})
edge/{idx}_{objID}.png       — edge images (must be pre-generated from CAD; not in raw download)
gt.yml                       — per-frame list of pose dicts: [{m2c_R: 3×3, m2c_T: 1×3}, ...]
Intrinsic.yml                — per-frame, per-slot, per-objID intrinsics (see below)
```

There are no depth maps in any test scene.

### The `index` slot — read this carefully

Each scene contains multiple objects simultaneously. Both `gt.yml` and `Intrinsic.yml` store
data as nested lists indexed first by frame, then by **slot position** within that scene:

```
gt.yml[frame_id][index]                → pose dict for the object at slot position `index`
Intrinsic.yml[frame_id][index][objID]  → 3×3 K matrix for that object
```

`index` is the **position of the target object within that scene's object list**, not the
object ID itself. Using the wrong index silently loads the wrong object's pose and intrinsics,
producing plausible-looking but incorrect results. The correct `(scene, index)` pairs for
each object are listed in the table below and are encoded in `eval_legacy_all_scenes.py`.

### The 4 background categories

The 16 scene directories are grouped into 4 background types (from the paper, Fig. 6 and Table V).
These 4 types are sometimes referred to as "four scenes" in the dataset README, but there are
4 *categories* of scenes, not 4 scene directories:

| Category | Background |
|----------|------------|
| Scene A  | Black background |
| Scene B  | Black background with texture |
| Scene C  | Simulated rust background |
| Scene D  | Reflective metal background |

Each category spans ~4 of the 16 scene directories. An object typically appears in one scene
from each of the background categories it was captured under.

### Object → scene mapping

`sceneObjs.yml` in the repo root lists which objects appear in which scenes. The full dataset
specification has 32 scene entries, but only 16 were included in the download. Scenes 7, 11,
and 23 are defined in `sceneObjs.yml` but are not present on disk.

The table below shows every on-disk scene for each trainable object, along with its `index`
slot, and any scene that is missing from the download:

| Object | On-disk scenes `(scene, index)` | Missing from disk |
|--------|----------------------------------|-------------------|
| obj1   | (3,3), (13,2), (24,0)           | —                 |
| obj2   | (13,1), (18,1), (25,3)          | —                 |
| obj3   | (13,0), (24,1), (32,2)          | scene 7           |
| obj6   | (3,0), (24,2), (31,0)           | scene 23          |
| obj7   | (1,3), (26,2), (31,1)           | —                 |
| obj13  | (10,2), (25,0), (32,0)          | —                 |
| obj16  | (4,0), (8,0)                    | —                 |
| obj18  | (21,2), (25,2)                  | scene 11          |
| obj21  | (5,2), (29,0)                   | —                 |
| obj32  | (30,1)                          | —                 |

The variation in scene count (1–3 per object) is **by dataset design**: obj21 was placed in
2 test scenarios, obj32 in just 1. No object reaches 4 on-disk scenes because the scenes that
would complete obj3, obj6, and obj18 to 4 are the ones missing from the download.

---

## CAD Models and Annotations

These files live in the repository root and are used by both training and evaluation.

```
cad/{cls}.ply        — 38 PLY files (ASCII); vertex positions, normals, RGBA colours; units: meters
keypoints/{cls}.txt  — 8–21 manually annotated 3D keypoints per object; units: meters
Valid3D/{cls}.txt    — 380–583 dense contour sample points per object; units: meters
```

Only the 10 trainable objects have entries in `keypoints/` and `Valid3D/`. The `cad/` directory
covers all 38 objects in the broader dataset. Object diameters are defined in `config.py` in
**millimeters**.

---

## Units

All 3D data in the dataset files is in **meters**. The evaluator converts to millimetres
internally for reporting:

| Data source | Units |
|-------------|-------|
| PLY vertices, keypoints, Valid3D | meters |
| `gt.yml` translation (`m2c_T`) — train and test | meters |
| `_RT.pkl` translation | meters |
| Translation errors reported by `eval.py` | mm |
| Diameters in `config.py` | mm |
| ADD correctness threshold | `diameter_mm × 0.1` |

`eval.py:28` applies the ×1000 conversion when loading PLY points for ADD distance computation.

---

## Evaluation

### Scripts

| Script | What it runs |
|--------|-------------|
| `_tmp_rt-less_paper_eval.sh` | Runs `eval_legacy_all_scenes.py --all_objects` twice — once without PECP, once with `--pecp` — using the authors' epoch-150 checkpoints from `model/paper_checkpoints/`. Covers every on-disk scene per object and writes `results.csv`, `run.log`, and `summary.txt` to `results/rtless/authors_checkpoints/{timestamp}_{pecp\|no_pecp}/`. |
| `eval_legacy_all_scenes.py` | Core multi-scene evaluation driver. Can run a single object (`--class_type`) or all 10 (`--all_objects`). Aggregates per-scene results into per-object and overall averages. |
| `eval_all_rtless_legacy.sh` | Runs `test_legacy_eval.py` with **one hardcoded scene per object** — the scene used in the original paper evaluation. Useful for reproducing the paper's exact numbers. |

### Metrics

- **2D projection**: mean reprojection distance of mesh points (predicted pose vs GT pose).
  Pose counted correct if distance < 5 px.
- **ADD(-S)**: mean point-cloud distance between predicted and GT pose transformations.
  Symmetric objects use ADD-S (nearest-neighbour); others use ADD.
  Correct if distance < 10% of object diameter.
- **R/t error**: rotation error in degrees (Euler angles α, β, γ) and translation error in mm,
  computed only over frames that pass the ADD(-S) threshold.

---

## Paper Results vs Multi-Scene Evaluation

The paper (Table IV) reports per-object ADD(-S) using ContourNet + PECP. The table below
compares those numbers against our multi-scene PECP run (`260609_0543_pecp`), which evaluates
every object across all its on-disk scenes:

| Paper Obj | Code Obj | Paper ADD | Multi-scene ADD | Δ |
|-----------|----------|-----------|-----------------|---|
| Obj 1     | obj1     | 100.00%   | 100.00%         |  0.0pp |
| Obj 2     | obj2     |  97.54%   |  90.22%         | −7.3pp |
| Obj 3     | obj3     |  95.35%   |  95.11%         | −0.2pp |
| Obj 4     | obj7     |  88.14%   |  91.27%         | +3.1pp |
| Obj 5     | obj13    |  90.70%   |  97.20%         | +6.5pp |
| Obj 6     | obj16    |  96.71%   |  92.07%         | −4.6pp |
| Obj 7     | obj18    |  91.82%   |  94.59%         | +2.8pp |
| Obj 8     | obj6     |  95.31%   |  89.34%         | −6.0pp |
| Obj 9     | obj21    |  93.50%   |  61.78%         | **−31.7pp** |
| Obj 10    | obj32    |  92.30%   |  92.31%         |  ~0pp |
| **Avg**   |          | **94.14%**| **91.14%**      | **−3.0pp** |

### Confirming the pipeline is correct

For obj32, which has only one test scene (scene 30), our PECP result is **92.31%** vs the
paper's **92.30%** — effectively identical. For obj21 on scene 5 alone, our PECP result is
**93.51%** vs the paper's **93.50%**. These exact matches confirm the evaluation pipeline
faithfully reproduces the paper's methodology when the scene set matches.

### Why obj21 drops so far

obj21 has two test scenes with very different outcomes:

| Scene | ADD (no PECP) | ADD (PECP) |
|-------|---------------|------------|
| 5     | 92.07%        | 93.51%     |
| 29    | 30.77%        | 30.05%     |

The paper's reported 93.50% matches scene 5 alone almost exactly, indicating the paper's
obj21 number reflects scene 5. The paper does not report a scene 29 result.

Scene 29 is genuinely hard for obj21: the 2D projection metric is only ~0.37 (vs ~0.98 for
scene 5), meaning the model fails at keypoint prediction for 63% of frames before PnP is
even reached. Scene 29 co-locates obj21 with objects 33, 34, and 30 — none of which are
in the trainable set — creating unlabeled clutter the model was not designed to handle.

**GT verification (scene 29).** Checked all 416 frames: slot 0 is obj_id 21 on every frame,
the GT CAD projects fully in-bounds, and mesh-to-mask overlap averages 100% with no frame
below 30%. The ground truth is sound. The 30% ADD is a model failure, not a data error.

### Interpreting the overall 3pp gap

The 3pp overall gap (paper 94.14% vs multi-scene 91.14%) is almost entirely explained by
scene 29 for obj21. Excluding that one scene, the multi-scene average sits close to the paper's
number. The multi-scene evaluation is the more complete and honest benchmark — it tests
generalization across all available conditions, including those the paper did not report on.
