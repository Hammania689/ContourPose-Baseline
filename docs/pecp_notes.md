# PECP Implementation Notes

**PECP** (Pose Estimation via Contour-confidence PnP) is the contour-aware refinement
described in the ContourPose paper. It is enabled via `--pecp` in `eval_legacy_all_scenes.py`
or `args.use_pecp = True` in the evaluator.

## Stochasticity

PECP is **non-deterministic by default**. The core loop in `eval.py:PECP()` runs 400
iterations of `random.choice(list_all)` (Python's `random` module, line ~218) to sample
4-keypoint subsets. No seed is set anywhere in the evaluation pipeline.

**Observed variance:** Re-running PECP evaluation on the same checkpoint and data can
shift the aggregate ADD by roughly ±1 pp. For obj21 scene 29 (416 frames), this corresponds
to ~3–4 frames flipping at the margin between runs.

Concrete example:
- Official 260609_0543_pecp run: **30.05%** ADD for obj21 scene 29
- A subsequent re-run with identical config: **29.33%** (−0.72 pp, 3 frames)

This is not a bug — it is expected behaviour from the unseeded random subset sampling.

## Making PECP reproducible

Seed Python's `random` module before calling `evaluator.evaluate()`:

```python
import random
random.seed(0)   # or any fixed integer
result = evaluator(args, model, loader, device).evaluate()
```

The seed must be set once per object/scene evaluation. Setting it inside `PECP()` directly
would also work but would require modifying `eval.py`.

**Important:** seeding will not reproduce the original 30.05% number unless you know the
exact seed used in the original run (which was not recorded). The practical approach is to
pick a fixed seed (e.g. 0), record whatever aggregate you get with that seed, and cite that
as your canonical number.

## What PECP does (brief)

For each frame where the predicted contour has >1000 foreground pixels:

1. Randomly samples 400 4-keypoint subsets from the predicted keypoints.
2. Solves PnP on each subset (EPNP).
3. Projects the `Valid3D/{obj}.txt` point cloud and scores how many projected points
   fall inside the predicted contour.
4. Iteratively refines: keeps the highest-scoring keypoint subset and tries adding more
   keypoints, dropping the worst-scoring ones when adding a keypoint hurts the score.
5. Falls back to standard RANSAC-PnP if the contour has ≤1000 foreground pixels (collapsed
   heatmap / no visible edge).

The contour-confidence scoring is what makes PECP better than vanilla PnP on good frames
— and what makes it stochastic.

## PECP is commented out in the default eval path

`eval.py:67–70` gates on `getattr(self.args, 'use_pecp', False)`, which defaults to
`False`. The official reproduction results in `results/rtless/authors_checkpoints/`
use `--pecp` enabled. Runs without that flag use `calculate_metric` (standard
RANSAC-PnP from GT heatmap keypoints) and give slightly different numbers.
