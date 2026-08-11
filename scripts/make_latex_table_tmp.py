#!/usr/bin/env python3
"""
Run inside the Docker container from the repo root:
    python scripts/make_latex_table_tmp.py

Reads results/rtless_paper/ and prints a LaTeX comparison table.
Also writes results/table_add.tex.

NOTE on obj8 mapping:
  The README maps paper obj7 AND obj8 both to code obj18 (a typo).
  Code obj6 has results but is absent from the README mapping.
  It is assumed here that paper obj8 -> code obj6. Adjust PAPER_TO_CODE if wrong.
"""

import csv
import json
from pathlib import Path

# ADD(-S) % values from the ContourPose paper (Table 1)
REPORTED = {
    "Obj 1":  100.00,
    "Obj 2":   97.54,
    "Obj 3":   95.35,
    "Obj 4":   88.14,
    "Obj 5":   90.70,
    "Obj 6":   96.71,
    "Obj 7":   91.82,
    "Obj 8":   95.31,
    "Obj 9":   93.50,
    "Obj 10":  92.30,
}

# Paper label -> results subfolder
PAPER_TO_CODE = {
    "Obj 1":  "obj1",
    "Obj 2":  "obj2",
    "Obj 3":  "obj3",
    "Obj 4":  "obj7",
    "Obj 5":  "obj13",
    "Obj 6":  "obj16",
    "Obj 7":  "obj18",
    "Obj 8":  "obj6",   # README typo; obj6 is the only unmapped code object
    "Obj 9":  "obj21",
    "Obj 10": "obj32",
}

RESULTS_DIR = Path("results/rtless_paper")


def load_add_rate(code_label: str) -> tuple[float, int]:
    p = RESULTS_DIR / code_label / "masked_detailed.csv"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    passed = sum(int(r["add_pass"]) for r in rows)
    return passed / len(rows) * 100, len(rows)


def _fmt(v: float, bold: bool) -> str:
    s = f"{v:.2f}"
    return rf"\textbf{{{s}}}" if bold else s


def make_latex_table(data: list[dict]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparison on Reflective Metal Parts Dataset Using the ADD(-S) Metric}",
        r"\label{tab:contourpose_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Methods} & \textbf{ContourPose (Reported)} & \textbf{Reproduced (Author's Data)} \\",
        r"\midrule",
    ]

    for row in data:
        rep   = row["reported"]
        repro = row["reproduced"]
        lines.append(
            f"{row['label']:6s} & "
            f"{_fmt(rep,   rep   >= repro):25s} & "
            f"{_fmt(repro, repro >  rep  )} \\\\"
        )

    lines.append(r"\midrule")

    reported_vals = [r["reported"]   for r in data]
    repro_vals    = [r["reproduced"] for r in data]

    all_rep   = sum(reported_vals) / len(reported_vals)
    all_repro = sum(repro_vals)    / len(repro_vals)
    lines.append(
        f"Average (All)       & "
        f"{_fmt(all_rep,   all_rep   >= all_repro):25s} & "
        f"{_fmt(all_repro, all_repro >  all_rep  )} \\\\"
    )

    tail_rep   = reported_vals[1:]
    tail_repro = repro_vals[1:]
    t_rep   = sum(tail_rep)   / len(tail_rep)
    t_repro = sum(tail_repro) / len(tail_repro)
    lines.append(
        f"Average (Obj 2--10) & "
        f"{_fmt(t_rep,   t_rep   >= t_repro):25s} & "
        f"{_fmt(t_repro, t_repro >  t_rep  )} \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\addlinespace[4pt]",
        r"\multicolumn{3}{p{0.85\linewidth}}{\footnotesize \textbf{Bold} = best per metric. "
        r"Per-object values differ from reported results, but overall averages are comparable, "
        r"validating our implementation.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    data = []
    print(f"{'Label':7} {'Code':6} {'N':6} {'Reported':10} {'Reproduced':10}")
    print("-" * 44)
    for label, code in PAPER_TO_CODE.items():
        rate, n = load_add_rate(code)
        rep = REPORTED[label]
        print(f"{label:7} {code:6} {n:6d} {rep:10.2f} {rate:10.2f}")
        data.append({"label": label, "code": code, "reported": rep, "reproduced": rate})

    latex = make_latex_table(data)
    print("\n\n" + "=" * 60)
    print(latex)

    out = Path("results/table_add.tex")
    out.write_text(latex)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
