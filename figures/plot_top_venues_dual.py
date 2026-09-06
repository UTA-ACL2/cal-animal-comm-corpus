#!/usr/bin/env python3
"""Top publication venues per research tradition — stacked lollipop chart
(biological over computational, sharing one log-scale x-axis)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_utils import (
    LoadSpec,
    load_df,
    lollipop_row,
    relative_rank_status_for_labels,
    short_venue_label,
    value_counts_scalar,
)
from _plot_style import apply_style, save_both, PALETTE


def _topk(df, group: str, k: int):
    dfg = df[df["group2"] == group]
    assert len(dfg) > 0, f"No rows for group2={group}"
    vc = value_counts_scalar(dfg, "venue", normalizer=short_venue_label)
    assert len(vc) > 0, f"No non-empty venues for group2={group}"
    return vc.head(k).sort_values(ascending=True)


def _trend_statuses(df, group: str, labels: list[str]) -> dict[str, str]:
    focal = value_counts_scalar(df[df["group2"] == group], "venue", normalizer=short_venue_label)
    other_group = "comp_or_hybrid" if group == "biological" else "biological"
    other = value_counts_scalar(df[df["group2"] == other_group], "venue", normalizer=short_venue_label)
    return relative_rank_status_for_labels(
        focal.index.tolist(),
        other.index.tolist(),
        labels,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=False, default=None)
    ap.add_argument("--jsonl",    required=False, default="../dataset/final_relevant.parquet")
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--k",        type=int, default=10)
    ap.add_argument("--out_dir",  default="output/venues")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, jsonl=args.jsonl, year_min=args.year_min, year_max=args.year_max))
    assert "venue" in df.columns

    bio  = _topk(df, "biological",    args.k)
    comp = _topk(df, "comp_or_hybrid",args.k)

    bio_status  = _trend_statuses(df, "biological", bio.index.tolist())
    comp_status = _trend_statuses(df, "comp_or_hybrid", comp.index.tolist())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.26, 7.7), sharex=True)

    # xmin=1 because log(0) is undefined; all venue counts are >= 1.
    # font_scale/row_spacing/tick_scale are oversized because this figure is
    # placed at a narrower \linewidth than its native size in the paper.
    lollipop_row(ax1, bio, "Biological", PALETTE["biological"], bio_status,
                 show_xlabel=False, xlabel="Papers (log scale)", xmin=1, log_scale=True, font_scale=1.5, row_spacing=1.5, tick_scale=1.1)
    lollipop_row(ax2, comp, "Computational", PALETTE["comp_or_hybrid"], comp_status,
                 show_xlabel=True, xlabel="Papers (log scale)", xmin=1, log_scale=True, font_scale=1.5, row_spacing=1.5, tick_scale=1.1)

    fig.suptitle("Most frequent publication venues by research tradition",
                 fontsize=16, fontweight="bold", fontfamily="sans-serif", y=1.02)
    # Marker legend (arrow meanings) is given once in the running text, not repeated here.
    fig.subplots_adjust(left=0.4, right=0.97, top=0.87, bottom=0.075, hspace=0.3)

    out_pdf = out_dir / "top_venues_dual.pdf"
    out_png = out_dir / "top_venues_dual.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
