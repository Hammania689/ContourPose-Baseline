"""
Aggregate per-object BOP eval outputs into the legacy results.csv layout.

Reads   {run_dir}/{obj}/masked_detailed.csv  (one per object)
Writes  {run_dir}/results.csv
        {run_dir}/summary.txt

The results.csv schema matches results/rtless/authors_checkpoints/260609_0543_pecp/
so downstream table/plot tooling works unchanged:

    object_id,scene_id,n_frames,add,proj_2d,trans_error_mm,
    x_error_mm,y_error_mm,z_error_mm,
    rot_error_deg,alpha_error_deg,beta_error_deg,gamma_error_deg

Per-object "all_scenes" and final "overall_average, all_objects" rows are
aggregated from raw per-frame data (frame-weighted, not scene-weighted), and
translation/rotation Euclidean errors are derived from the mean per-axis
errors — same convention as eval_legacy_all_scenes.py so numbers stack up
directly against the legacy pipeline.

Usage:
    python scripts/aggregate_bop_results.py --run_dir results/rtless_bop/260806_1830_pecp
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np


CSV_COLUMNS = [
    "object_id", "scene_id", "n_frames",
    "add", "proj_2d", "trans_error_mm",
    "x_error_mm", "y_error_mm", "z_error_mm",
    "rot_error_deg", "alpha_error_deg", "beta_error_deg", "gamma_error_deg",
]

# Column names to pull from each object's masked_detailed.csv
FIELDS = ["add_pass", "proj_2d_pass",
          "x_error_mm", "y_error_mm", "z_error_mm",
          "alpha_error_deg", "beta_error_deg", "gamma_error_deg"]


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _nanmean(xs):
    """Mean over finite values; NaN if the list is empty or all NaN.
    Matches the legacy accounting in eval.py:calculate_tra_and_rot, which
    only records per-axis errors for frames that passed ADD — so NaN/absent
    values here just don't contribute to the aggregate."""
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr[np.isfinite(arr)]))


def _aggregate(raw):
    # add_pass / proj_2d_pass are 0/1 pass flags — always defined per frame,
    # so plain mean matches the legacy pass-ratio semantics.
    add = _mean(raw["add_pass"])
    proj = _mean(raw["proj_2d_pass"])

    # Per-axis errors: legacy calculate_tra_and_rot (eval.py:112-114) early-returns
    # for frames where ADD failed, so per-axis errors are ONLY averaged over
    # ADD-passing frames. The BOP eval records errors on every frame including
    # degenerate PnP blowups (z ~ 1e17 mm), which would poison a plain mean.
    # Mask by add_pass to match the legacy convention exactly.
    add_pass = np.asarray(raw["add_pass"], dtype=float)
    def _masked(field):
        arr = np.asarray(raw[field], dtype=float)
        return arr[add_pass == 1.0].tolist()

    x = _nanmean(_masked("x_error_mm"))
    y = _nanmean(_masked("y_error_mm"))
    z = _nanmean(_masked("z_error_mm"))
    a = _nanmean(_masked("alpha_error_deg"))
    b = _nanmean(_masked("beta_error_deg"))
    g = _nanmean(_masked("gamma_error_deg"))
    trans = math.sqrt(x*x + y*y + z*z) if not math.isnan(x) else float("nan")
    rot   = math.sqrt(a*a + b*b + g*g) if not math.isnan(a) else float("nan")
    return {
        "n_frames":        len(raw["add_pass"]),
        "add":             add,
        "proj_2d":         proj,
        "trans_error_mm":  trans,
        "x_error_mm":      x,
        "y_error_mm":      y,
        "z_error_mm":      z,
        "rot_error_deg":   rot,
        "alpha_error_deg": a,
        "beta_error_deg":  b,
        "gamma_error_deg": g,
    }


def _row(object_id, scene_id, metrics):
    return {"object_id": object_id, "scene_id": scene_id,
            **{k: metrics[k] for k in CSV_COLUMNS if k not in ("object_id", "scene_id")}}


def _load_detailed(csv_path):
    """Return dict of column-name → list[float], one entry per frame.
    Unparseable values become NaN rather than dropping the whole row —
    this keeps the frame count aligned with what test.py actually ran
    (matches legacy where every frame contributes to add/proj_2d ratios,
    and failed frames just don't contribute to per-axis means)."""
    raw = {f: [] for f in FIELDS + ["scene_id"]}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "scene_id" not in r or r["scene_id"] is None:
                continue
            raw["scene_id"].append(r["scene_id"])
            for f_ in FIELDS:
                try:
                    raw[f_].append(float(r.get(f_, "nan")))
                except (TypeError, ValueError):
                    raw[f_].append(float("nan"))
    return raw


def _split_by_scene(raw):
    """Group raw per-frame data by scene_id."""
    by_scene = {}
    for i, sid in enumerate(raw["scene_id"]):
        bucket = by_scene.setdefault(sid, {f: [] for f in FIELDS + ["scene_id"]})
        for f_ in FIELDS:
            bucket[f_].append(raw[f_][i])
        bucket["scene_id"].append(sid)
    return by_scene


def _pretty_scene(sid):
    """'000003' → '3' so results.csv matches legacy formatting."""
    try:
        return str(int(sid))
    except ValueError:
        return sid


def _resolve_detailed_name(od: Path, explicit: Optional[str]) -> Optional[Path]:
    """Pick the per-instance CSV. If explicit is set, use it; otherwise fall
    back to whichever of {rgb,masked}_detailed.csv exists (test.py names the
    output after the --use_masks variant it ran)."""
    if explicit:
        p = od / explicit
        return p if p.exists() else None
    for name in ("rgb_detailed.csv", "masked_detailed.csv"):
        p = od / name
        if p.exists():
            return p
    return None


def aggregate_run(run_dir: Path, detailed_name: Optional[str] = None):
    obj_dirs = sorted([d for d in run_dir.iterdir()
                       if d.is_dir() and d.name.startswith("obj")],
                      key=lambda d: int(d.name[3:]) if d.name[3:].isdigit() else 999)
    if not obj_dirs:
        raise FileNotFoundError(f"No obj*/ subdirs under {run_dir}")

    all_scene_rows = []      # list[list[row]]
    all_obj_rows   = []      # list[row]
    combined_all   = {f: [] for f in FIELDS + ["scene_id"]}
    scenes_by_obj  = {}      # obj_name → [scene_id, ...] for summary.txt

    for od in obj_dirs:
        det_path = _resolve_detailed_name(od, detailed_name)
        if det_path is None:
            print(f"[skip] {od.name}: no *_detailed.csv found "
                  f"(looked for {detailed_name or 'rgb_detailed.csv / masked_detailed.csv'})")
            continue

        raw = _load_detailed(det_path)
        if not raw["add_pass"]:
            print(f"[skip] {od.name}: no numeric rows in {detailed_name}")
            continue

        by_scene = _split_by_scene(raw)
        scene_rows = [_row(od.name, _pretty_scene(sid), _aggregate(sraw))
                      for sid, sraw in sorted(by_scene.items(),
                                              key=lambda kv: int(kv[0]))]
        obj_row = _row(od.name, "all_scenes", _aggregate(raw))

        all_scene_rows.append(scene_rows)
        all_obj_rows.append(obj_row)
        scenes_by_obj[od.name] = [_pretty_scene(s) for s in sorted(by_scene, key=int)]

        for f_ in FIELDS:
            combined_all[f_].extend(raw[f_])
        combined_all["scene_id"].extend(raw["scene_id"])

        print(f"[ok]  {od.name}: {len(raw['add_pass'])} frames across "
              f"{len(by_scene)} scene(s)  add={obj_row['add']:.4f}  "
              f"proj2d={obj_row['proj_2d']:.4f}")

    overall = _aggregate(combined_all)
    return all_scene_rows, all_obj_rows, overall, scenes_by_obj


def write_csv(path, all_scene_rows, all_obj_rows, overall_metrics):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for scene_rows, obj_row in zip(all_scene_rows, all_obj_rows):
            writer.writerows(scene_rows)
            writer.writerow(obj_row)
            writer.writerow({})   # blank separator between objects
        writer.writerow(_row("overall_average", "all_objects", overall_metrics))


def write_summary(path, run_dir, scenes_by_obj, overall):
    with open(path, "w") as f:
        f.write("ContourPose BOP/DALI Evaluation — Run Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Output dir  : {run_dir}\n")
        f.write(f"Pipeline    : test.py + eval_spectra_pose.py + BOPTestDALIDataset\n\n")
        f.write(f"Objects evaluated ({len(scenes_by_obj)}):\n")
        for obj, scenes in scenes_by_obj.items():
            f.write(f"  {obj}: scenes {', '.join(scenes)}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Overall (frame-weighted across all {overall['n_frames']} frames):\n")
        f.write(f"  ADD (pass ratio)  : {overall['add']:.4f}\n")
        f.write(f"  2D proj pass ratio: {overall['proj_2d']:.4f}\n")
        f.write(f"  Trans error       : {overall['trans_error_mm']:.3f} mm\n")
        f.write(f"  Rot error         : {overall['rot_error_deg']:.3f} deg\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Timestamped run dir containing obj*/masked_detailed.csv")
    p.add_argument("--detailed_name", default=None,
                   help="Detailed CSV filename inside each obj dir. If omitted, "
                        "auto-picks rgb_detailed.csv, then falls back to masked_detailed.csv.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[error] run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    scene_rows, obj_rows, overall, scenes_by_obj = aggregate_run(
        run_dir, detailed_name=args.detailed_name)

    csv_path = run_dir / "results.csv"
    sum_path = run_dir / "summary.txt"
    write_csv(csv_path, scene_rows, obj_rows, overall)
    write_summary(sum_path, run_dir, scenes_by_obj, overall)

    print(f"\nWrote:")
    print(f"  {csv_path}")
    print(f"  {sum_path}")


if __name__ == "__main__":
    main()
