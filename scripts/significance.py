#!/usr/bin/env python3
"""Statistical significance over multiple eval runs.

Reads a set of ``corr_summary_*.csv`` files (one per random split) produced by
``ceshi/corr_eval.py`` and prints, for every metric:

    * mean and std of Spearman / Pearson correlation across splits
    * paired Wilcoxon p-value vs. GAVEL (or any chosen reference)
    * whether the gap is significant at p<0.05 (``†`` in the table footer)

This implements the rebuttal commitment to reviewer tUiR-W2 ("we will integrate
mean ± std and significance into the main table").

Typical usage
-------------
    python scripts/significance.py \
        --summary_glob "./outputs/corr/corr_summary_split_*.csv" \
        --reference out_gavel_split \
        --out_csv ./outputs/significance_table.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


# Strip trailing ``_<digits>`` so ``out_refclip_3`` -> ``out_refclip``.
_RUN_SUFFIX = re.compile(r"_\d+$")


def canonical_metric(name: str) -> str:
    return _RUN_SUFFIX.sub("", name)


def pick_score_col(group: pd.DataFrame, prefer: List[str]) -> str:
    """For metrics with multiple score columns (e.g. siglip has sim+dist) pick
    the one ranked highest by ``prefer`` ordering.  Falls back to the column
    with the largest mean Spearman value."""
    cols = group["score_col"].unique().tolist()
    for p in prefer:
        for c in cols:
            if c.lower() == p.lower():
                return c
    # fallback
    return (
        group.groupby("score_col")["spearman_rho"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )


def build_matrix(summary_files: List[str],
                 prefer_score_col: List[str]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (spearman_matrix, pearson_matrix); rows are metrics, cols are
    split indices (0..K-1, sorted by file path)."""
    per_split: Dict[int, pd.DataFrame] = {}
    for k, path in enumerate(sorted(summary_files)):
        df = pd.read_csv(path)
        df["canonical"] = df["metric_name"].map(canonical_metric)
        per_split[k] = df

    if not per_split:
        raise FileNotFoundError("no summary CSVs matched the glob")

    # Find the union of metrics across splits
    all_metrics = sorted(set().union(*[set(d["canonical"]) for d in per_split.values()]))

    s_rows = []
    p_rows = []
    for m in all_metrics:
        s_row = {"metric": m}
        p_row = {"metric": m}
        for k, df in per_split.items():
            sub = df[df["canonical"] == m]
            if sub.empty:
                s_row[f"split{k}"] = np.nan
                p_row[f"split{k}"] = np.nan
                continue
            col = pick_score_col(sub, prefer_score_col)
            sub2 = sub[sub["score_col"] == col]
            s_row[f"split{k}"] = float(sub2["spearman_rho"].mean())
            p_row[f"split{k}"] = float(sub2["pearson_r"].mean())
        s_rows.append(s_row)
        p_rows.append(p_row)

    s_mat = pd.DataFrame(s_rows).set_index("metric")
    p_mat = pd.DataFrame(p_rows).set_index("metric")
    return s_mat, p_mat


def paired_pvalues(mat: pd.DataFrame, ref_metric: str
                   ) -> pd.DataFrame:
    """For every metric != ref, run a paired Wilcoxon (and t-test) across
    splits.  Drops splits where either side is NaN."""
    if ref_metric not in mat.index:
        raise KeyError(
            f"reference metric {ref_metric!r} not found.  Available: "
            f"{sorted(mat.index.tolist())}"
        )

    ref = mat.loc[ref_metric].to_numpy()
    rows = []
    for m in mat.index:
        if m == ref_metric:
            rows.append({"metric": m, "p_wilcoxon": np.nan, "p_ttest": np.nan,
                         "n_paired_splits": int(np.isfinite(ref).sum())})
            continue
        other = mat.loc[m].to_numpy()
        ok = np.isfinite(ref) & np.isfinite(other)
        if ok.sum() < 3:
            rows.append({"metric": m, "p_wilcoxon": np.nan, "p_ttest": np.nan,
                         "n_paired_splits": int(ok.sum())})
            continue
        try:
            _, pw = wilcoxon(ref[ok], other[ok], zero_method="zsplit")
        except ValueError:
            pw = np.nan
        try:
            _, pt = ttest_rel(ref[ok], other[ok])
        except Exception:
            pt = np.nan
        rows.append({"metric": m, "p_wilcoxon": float(pw),
                     "p_ttest": float(pt),
                     "n_paired_splits": int(ok.sum())})
    return pd.DataFrame(rows).set_index("metric")


def format_table(mat: pd.DataFrame,
                 pvals: pd.DataFrame,
                 ref_metric: str) -> pd.DataFrame:
    n_splits = mat.shape[1]
    out = pd.DataFrame(index=mat.index)
    out["mean"] = mat.mean(axis=1)
    out["std"] = mat.std(axis=1)
    out["n"] = mat.notna().sum(axis=1)
    out = out.join(pvals[["p_wilcoxon", "p_ttest"]], how="left")
    out["sig_vs_ref"] = out["p_wilcoxon"].apply(
        lambda p: "" if (np.isnan(p) or p >= 0.05) else "†"
    )

    def _fmt(row):
        s = f"{row['mean']:.4f} ± {row['std']:.4f}"
        if row.name == ref_metric:
            return s + "  ⟵ ref"
        if row["sig_vs_ref"] == "†":
            return s + f"  † (p={row['p_wilcoxon']:.3g})"
        if not np.isnan(row["p_wilcoxon"]):
            return s + f"     (p={row['p_wilcoxon']:.3g})"
        return s

    out["pretty"] = out.apply(_fmt, axis=1)
    return out.sort_values("mean", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary_glob", required=True,
                    help='Glob, e.g. "ceshi/corr_summary_*.csv"')
    ap.add_argument("--reference", default="out_refclip",
                    help="Reference metric (canonical name without _<split>). "
                         "Other metrics are paired-tested against this.")
    ap.add_argument("--prefer_score_col", nargs="+",
                    default=["refclip", "gavel", "sim", "similarity",
                             "lpips_alex", "dists", "ms_ssim"],
                    help="Preferred order when a metric has multiple score "
                         "columns (e.g. siglip has both sim and dist).")
    ap.add_argument("--out_csv", default="",
                    help="Optional output: detailed mean/std/p-value table.")
    ap.add_argument("--out_matrix_csv", default="",
                    help="Optional output: raw NxK Spearman matrix.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.summary_glob))
    if not files:
        print(f"[err] no files matched {args.summary_glob!r}", file=sys.stderr)
        sys.exit(2)
    print(f"[load] {len(files)} summary files:")
    for f in files:
        print(f"       {f}")

    s_mat, p_mat = build_matrix(files, args.prefer_score_col)
    print("\n[Spearman matrix]")
    print(s_mat.round(4).to_string())

    if args.out_matrix_csv:
        os.makedirs(os.path.dirname(args.out_matrix_csv) or ".", exist_ok=True)
        s_mat.to_csv(args.out_matrix_csv)
        print(f"[save] raw matrix -> {args.out_matrix_csv}")

    pvals = paired_pvalues(s_mat, args.reference)
    table = format_table(s_mat, pvals, args.reference)

    print("\n" + "=" * 80)
    print(f"Spearman ρ across {s_mat.shape[1]} splits  "
          f"(reference = {args.reference!r}; † = significant at p<0.05 "
          f"by paired Wilcoxon)")
    print("=" * 80)
    print(table[["pretty"]].to_string())
    print("=" * 80)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        table.to_csv(args.out_csv)
        print(f"[save] detailed table -> {args.out_csv}")


if __name__ == "__main__":
    main()
