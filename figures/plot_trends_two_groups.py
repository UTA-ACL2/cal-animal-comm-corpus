#!/usr/bin/env python3
"""
Publication trends as a stacked bar plot -- Biological vs. Computational.
Years are bucketed (coarse pre-1980, then by decade) rather than plotted
year-by-year: the pre-1980 range is near-empty, so a continuous year axis
wastes most of its width on a flat line. Bucketing keeps the figure compact
while still showing the late acceleration of computational work.

Hybrid-tradition papers (~1.2% of the corpus) are folded into "Computational",
matching the original (reviewed) manuscript's two-group legend; the exact
hybrid count is disclosed in the stats sidecar only, not in the figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _plot_utils import LoadSpec, load_df, write_stats_txt
from _plot_style import apply_style, save_both, PALETTE

# (bucket start, exclusive end, label). Pre-1980 is one wide bucket since it
# is near-empty; buckets from 1980 on are by decade.
BUCKET_EDGES = [1980, 1990, 2000, 2010, 2020]

# Latest publication_date among relevant papers in the corpus snapshot used
# for this figure (data/processed_nemotron/final_relevant.jsonl); the final
# bucket is a partial period ending here, not a full decade, so it is marked
# with "*" and this date is disclosed alongside it rather than left implicit.
LAST_SNAPSHOT_DATE = "2026-06-25"


def _make_buckets(year_min: int, year_max: int) -> list[tuple[int, int, str]]:
    edges = [year_min] + [e for e in BUCKET_EDGES if year_min < e <= year_max] + [year_max + 1]
    edges = sorted(set(edges))
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == year_min and lo < BUCKET_EDGES[0]:
            label = f"≤{hi - 1}"
        elif hi - lo <= 1:
            label = f"{lo}"
        elif hi - 1 == lo + 9:
            label = f"{lo}s"
        else:
            label = f"{lo}–{str(hi - 1)[-2:]}"
        buckets.append((lo, hi, label))
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=False, default=None)
    ap.add_argument("--jsonl",    required=False, default="../dataset/final_relevant.parquet")
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--out_dir",  default="output/trends")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, jsonl=args.jsonl, year_min=args.year_min, year_max=args.year_max))
    df = df.copy()
    df["year"] = df["year"].astype(int)

    buckets = _make_buckets(args.year_min, args.year_max)
    labels  = [b[2] for b in buckets]
    # Mark the final bucket if it's a partial/open period (ends at year_max,
    # i.e. "now", not at a real decade boundary) so the reader doesn't read
    # "2020-26" as a complete decade.
    last_lo, last_hi, _ = buckets[-1]
    open_final_bucket = (last_hi - 1) == args.year_max
    if open_final_bucket:
        labels[-1] = labels[-1] + "*"
    y_bio   = np.zeros(len(buckets))
    y_comp  = np.zeros(len(buckets))
    for i, (lo, hi, _) in enumerate(buckets):
        sub = df[(df["year"] >= lo) & (df["year"] < hi)]
        y_bio[i]  = int((sub["group2"] == "biological").sum())
        y_comp[i] = int((sub["group2"] == "comp_or_hybrid").sum())

    col_bio  = PALETTE["biological"]
    col_comp = PALETTE["comp_or_hybrid"]

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(buckets))
    ax.bar(x, y_bio,  color=col_bio,  label="Biological",    width=0.62)
    ax.bar(x, y_comp, color=col_comp, label="Computational", width=0.62, bottom=y_bio)

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("Papers")
    ax.set_xlabel("Year")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
    ax.set_title("Publication trends in animal communication research",
                 fontsize=10.5, fontweight="bold")
    if open_final_bucket:
        fig.text(0.98, 0.005, f"* through {LAST_SNAPSHOT_DATE}",
                  ha="right", va="bottom", fontsize=6.6, color="#555555")

    out_pdf = out_dir / "publication_trends_two_groups.pdf"
    out_png = out_dir / "publication_trends_two_groups.png"
    save_both(fig, out_pdf, out_png, tight=True)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")

    n_bio, n_comp = int(y_bio.sum()), int(y_comp.sum())
    n_total = n_bio + n_comp
    n_hybrid = int((df["trad3"] == "hybrid").sum())
    bucket_lines = [
        f"  {label:>8s}: biological={int(b):6d}  computational(incl. hybrid)={int(c):6d}"
        for (_, _, label), b, c in zip(buckets, y_bio, y_comp)
    ]
    write_stats_txt(
        out_dir / "publication_trends_two_groups_stats.txt",
        "Publication trends in animal communication research -- exact counts",
        [
            f"Year range: {args.year_min}-{args.year_max}",
            f"Total papers in range: {n_total:,}",
            f"  biological:              {n_bio:,} ({n_bio/n_total:.1%})",
            f"  computational+hybrid:    {n_comp:,} ({n_comp/n_total:.1%})",
            f"    of which hybrid-tradition: {n_hybrid:,} ({n_hybrid/n_total:.1%} of total)",
            "NOTE: 'Computational' in the figure legend includes hybrid-tradition",
            "papers (both biological and computational methods), matching the",
            "original manuscript's grouping; exact hybrid count disclosed here.",
            "",
            "By bucket (pre-1980 merged into one bucket; decades thereafter):",
            *bucket_lines,
        ],
    )


if __name__ == "__main__":
    main()
