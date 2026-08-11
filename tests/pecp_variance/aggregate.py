"""
Aggregate a PECP variance sweep produced by tests/pecp_variance/run_sweep.sh.

Inputs (per sweep):
    {sweep_root}/seed{k}/{obj_dir}/rgb_detailed.csv    (or masked_detailed.csv)
        obj_dir ∈ {obj1, obj2, obj3, obj6, obj7, obj13, obj16, obj18,
                   obj21, obj32, obj21_excl_scene29}

Outputs:
    {sweep_root}/variance_raw.csv
        object,code_index,scene,seed,n_frames,add,proj_2d,trans_error_mm,rot_error_deg
        One row per (obj, scene, seed). scene="all_scenes" for the object
        aggregate; specific numbers (e.g. "5", "29") for per-scene rows.

    {sweep_root}/variance_summary.csv
        object,code_index,scene,n_seeds,add_mean,add_std,add_min,add_max,
                                        proj_mean,proj_std,proj_min,proj_max
        One row per (obj, scene). Aggregates across seeds.

    {sweep_root}/variance_report.txt
        Human-readable object-level summary with sanity checks:
          - Do all seeds actually differ?
          - Do prior single-run reference numbers fall inside the range?
          - Rough obj/std ranking (variance-structure prediction check).

Aggregation uses the legacy accounting convention: per-axis errors are only
averaged over frames where ADD passed (matches eval.py:112-114). Frame count
and pass ratios use all frames.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reference single-run ADD numbers to sanity-check against (from prior runs).
# Format: (obj_name, scene_str) → expected ADD in [0, 1].
# obj21 scene 5 = 93.51% and obj32 = 92.31% quoted in the sweep spec.
REFERENCE_ADD = {
    ("obj21", "5"):         0.9351,
    ("obj32", "all_scenes"): 0.9231,
}

FIELDS = ["add_pass", "proj_2d_pass",
          "x_error_mm", "y_error_mm", "z_error_mm",
          "alpha_error_deg", "beta_error_deg", "gamma_error_deg"]


def _pretty_scene(sid):
    try:
        return str(int(sid))
    except (TypeError, ValueError):
        return str(sid)


def _load_detailed(csv_path: Path) -> Dict[str, list]:
    """One entry per frame; unparseable → NaN so frame count stays honest."""
    raw = {f: [] for f in FIELDS}
    raw["scene_id"] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r.get("scene_id")
            if sid is None:
                continue
            raw["scene_id"].append(sid)
            for f_ in FIELDS:
                try:
                    raw[f_].append(float(r.get(f_, "nan")))
                except (TypeError, ValueError):
                    raw[f_].append(float("nan"))
    return raw


def _mean(xs):
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _agg_frames(raw: Dict[str, list]) -> dict:
    """Legacy accounting: pass ratios over all frames, per-axis over ADD-pass."""
    add_pass = np.asarray(raw["add_pass"], dtype=float)
    n = len(raw["scene_id"])
    if n == 0:
        return {
            "n_frames": 0, "add": float("nan"), "proj_2d": float("nan"),
            "trans_error_mm": float("nan"), "rot_error_deg": float("nan"),
        }
    add = _mean(raw["add_pass"])
    proj = _mean(raw["proj_2d_pass"])

    def _masked_mean(field):
        arr = np.asarray(raw[field], dtype=float)
        m = arr[(add_pass == 1.0) & np.isfinite(arr)]
        return float(m.mean()) if m.size else float("nan")

    x = _masked_mean("x_error_mm")
    y = _masked_mean("y_error_mm")
    z = _masked_mean("z_error_mm")
    a = _masked_mean("alpha_error_deg")
    b = _masked_mean("beta_error_deg")
    g = _masked_mean("gamma_error_deg")
    trans = math.sqrt(x*x + y*y + z*z) if not math.isnan(x) else float("nan")
    rot   = math.sqrt(a*a + b*b + g*g) if not math.isnan(a) else float("nan")
    return {
        "n_frames": n, "add": add, "proj_2d": proj,
        "trans_error_mm": trans, "rot_error_deg": rot,
    }


def _split_by_scene(raw: Dict[str, list]) -> Dict[str, Dict[str, list]]:
    out: Dict[str, Dict[str, list]] = {}
    for i, sid in enumerate(raw["scene_id"]):
        bucket = out.setdefault(sid, {f: [] for f in FIELDS + ["scene_id"]})
        bucket["scene_id"].append(sid)
        for f_ in FIELDS:
            bucket[f_].append(raw[f_][i])
    return out


def _resolve_detailed(od: Path) -> Optional[Path]:
    for name in ("rgb_detailed.csv", "masked_detailed.csv"):
        p = od / name
        if p.exists():
            return p
    return None


def _code_index(obj_dir_name: str) -> Optional[int]:
    """obj21 → 21, obj21_excl_scene29 → 21."""
    name = obj_dir_name.split("_")[0]  # strip trailing _excl_scene29 etc.
    if name.startswith("obj") and name[3:].isdigit():
        return int(name[3:])
    return None


def collect_raw(sweep_root: Path) -> List[dict]:
    """Return list of rows: {object, code_index, scene, seed, n_frames, add, proj_2d, trans_error_mm, rot_error_deg}."""
    rows: List[dict] = []
    seed_dirs = sorted(sweep_root.glob("seed*"),
                       key=lambda p: int(p.name.replace("seed", "")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed*/ directories under {sweep_root}")

    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        for od in sorted(sd.iterdir()):
            if not od.is_dir():
                continue
            if not od.name.startswith("obj"):
                continue
            det = _resolve_detailed(od)
            if det is None:
                print(f"[warn] {sd.name}/{od.name}: no *_detailed.csv found")
                continue

            raw = _load_detailed(det)
            code_ix = _code_index(od.name)

            # Per-scene rows
            for sid, sraw in sorted(_split_by_scene(raw).items(), key=lambda kv: int(kv[0])):
                m = _agg_frames(sraw)
                rows.append({
                    "object": od.name, "code_index": code_ix,
                    "scene": _pretty_scene(sid), "seed": seed, **m,
                })
            # All-scenes aggregate
            m = _agg_frames(raw)
            rows.append({
                "object": od.name, "code_index": code_ix,
                "scene": "all_scenes", "seed": seed, **m,
            })
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    """Group by (object, scene), compute mean/std/min/max over seeds."""
    from collections import defaultdict
    grp: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        grp[(r["object"], r["scene"])].append(r)

    def _stats(vals: List[float]) -> Tuple[float, float, float, float]:
        arr = np.asarray([v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))],
                         dtype=float)
        if arr.size == 0:
            return (float("nan"),) * 4
        return (float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                float(arr.min()),  float(arr.max()))

    summary: List[dict] = []
    for (obj, scene), grp_rows in sorted(grp.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
        add_m, add_s, add_lo, add_hi = _stats([r["add"] for r in grp_rows])
        pj_m,  pj_s,  pj_lo,  pj_hi  = _stats([r["proj_2d"] for r in grp_rows])
        tr_m,  tr_s,  tr_lo,  tr_hi  = _stats([r["trans_error_mm"] for r in grp_rows])
        rt_m,  rt_s,  rt_lo,  rt_hi  = _stats([r["rot_error_deg"] for r in grp_rows])
        summary.append({
            "object": obj, "code_index": grp_rows[0]["code_index"],
            "scene": scene, "n_seeds": len(grp_rows),
            "add_mean": add_m, "add_std": add_s, "add_min": add_lo, "add_max": add_hi,
            "proj_mean": pj_m, "proj_std": pj_s, "proj_min": pj_lo, "proj_max": pj_hi,
            "trans_mean_mm": tr_m, "trans_std_mm": tr_s,
            "rot_mean_deg":  rt_m, "rot_std_deg":  rt_s,
        })
    return summary


def sanity_report(rows: List[dict], summary: List[dict]) -> str:
    """Human-readable checks: seed diversity, reference-hit, variance structure."""
    lines = ["PECP variance sweep — sanity report",
             "=" * 60, ""]

    # 1. Seeds actually differ?
    lines.append("1. Do seeds produce distinct ADD numbers?")
    from collections import defaultdict
    per_obj_all: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r["scene"] == "all_scenes":
            per_obj_all[r["object"]].append((r["seed"], r["add"]))
    any_collapse = False
    for obj in sorted(per_obj_all):
        pairs = sorted(per_obj_all[obj])
        adds = [a for _, a in pairs if not math.isnan(a)]
        distinct = len(set(round(a, 8) for a in adds))
        marker = "  OK" if distinct >= max(2, len(adds) // 2) else "  FLAG"
        if distinct < 2 and len(adds) >= 2:
            any_collapse = True
        lines.append(f"   {marker}  {obj:22s} distinct_adds={distinct}/{len(adds)}  "
                     f"range=[{min(adds):.4f}, {max(adds):.4f}]"
                     if adds else f"   {obj}: no data")
    lines.append("   → If any FLAG rows have distinct_adds=1, seeding did NOT take."
                 if any_collapse else "   → All objects show seed-driven variance. Seeding took.")
    lines.append("")

    # 2. Reference single-run numbers inside the 10-run range?
    lines.append("2. Reference single-run ADD numbers vs 10-run range:")
    inside = True
    idx = {(s["object"], s["scene"]): s for s in summary}
    for (obj, scene), ref_add in REFERENCE_ADD.items():
        s = idx.get((obj, scene))
        if s is None:
            lines.append(f"   [n/a]   {obj:22s} scene={scene:11s} reference={ref_add:.4f}  (no summary row)")
            continue
        lo, hi = s["add_min"], s["add_max"]
        ok = (lo - 1e-6) <= ref_add <= (hi + 1e-6)
        marker = "  OK" if ok else "  FLAG"
        lines.append(f"   {marker}  {obj:22s} scene={scene:11s} "
                     f"reference={ref_add:.4f}  observed_range=[{lo:.4f}, {hi:.4f}]  "
                     f"mean={s['add_mean']:.4f}±{s['add_std']:.4f}")
        inside = inside and ok
    lines.append("   → All references inside observed ranges."
                 if inside else "   → At least one reference outside range — flagged above.")
    lines.append("")

    # 3. Variance-structure prediction: tight on strong-heatmap objects, wide on hard cases.
    lines.append("3. Per-object ADD std ranking (tightest → widest, all_scenes):")
    rows_std = [(s["object"], s["add_mean"], s["add_std"])
                for s in summary if s["scene"] == "all_scenes"
                and not math.isnan(s["add_std"])]
    rows_std.sort(key=lambda t: t[2])
    for obj, mean, std in rows_std:
        lines.append(f"     {obj:22s} mean={mean:.4f}  std={std:.5f}")
    lines.append("")
    lines.append("   Prediction: strong-heatmap objects (obj1, obj7, obj13) tight;")
    lines.append("   hard cases (obj2, obj16, obj21 with scene 29) wider. Compare above.")
    return "\n".join(lines)


def write_csv(path: Path, rows: List[dict], cols: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="Sweep dir with seed*/obj*/ subdirs")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_dir():
        print(f"[error] sweep_root not found: {sweep_root}", file=sys.stderr)
        sys.exit(1)

    rows = collect_raw(sweep_root)
    summary = summarize(rows)

    raw_csv = sweep_root / "variance_raw.csv"
    sum_csv = sweep_root / "variance_summary.csv"
    report  = sweep_root / "variance_report.txt"

    write_csv(raw_csv, rows, [
        "object", "code_index", "scene", "seed",
        "n_frames", "add", "proj_2d", "trans_error_mm", "rot_error_deg",
    ])
    write_csv(sum_csv, summary, [
        "object", "code_index", "scene", "n_seeds",
        "add_mean", "add_std", "add_min", "add_max",
        "proj_mean", "proj_std", "proj_min", "proj_max",
        "trans_mean_mm", "trans_std_mm",
        "rot_mean_deg", "rot_std_deg",
    ])
    report.write_text(sanity_report(rows, summary))

    print(f"\nWrote:")
    print(f"  {raw_csv}")
    print(f"  {sum_csv}")
    print(f"  {report}")
    print()
    print(report.read_text())


if __name__ == "__main__":
    main()
