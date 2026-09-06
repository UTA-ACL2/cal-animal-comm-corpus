#!/usr/bin/env python3
"""
Top species by research tradition — stacked lollipop chart (biological over
computational, sharing one log-scale x-axis).

Species labels are normalized via _plot_utils.norm_species_label (authority/
year suffix stripping, synonym canonicalization, common-name singularization)
and then coarsened via _plot_utils.taxon_group_label, which collapses
individual whale/porpoise/bat species into one "Whale"/"Porpoise"/"Bat" row
each -- these are long-tail near-duplicates with no independent discussion in
the text, unlike well-established model species (Zebra Finch, Bottlenose
Dolphin, ...) which are deliberately left un-grouped.

The x-axis is log-scale: after taxon grouping, "Whale" is an order of
magnitude larger than most other rows and would otherwise compress everything
else against the axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_utils import (
    LoadSpec,
    load_df,
    lollipop_row,
    relative_rank_status_for_labels,
    taxon_group_label,
    value_counts_list,
    write_stats_txt,
)
from _plot_style import apply_style, save_both, PALETTE


def _counts_for_group(df, group: str):
    dfg = df[df["group2"] == group]
    assert len(dfg) > 0, f"No rows for group2={group}"
    vc = value_counts_list(
        dfg, "species", normalizer=taxon_group_label,
        drop_labels={"Other", "Others", "other species"},
    )
    assert len(vc) > 0, f"No species values for group2={group}"
    return vc


def _topk(vc, k: int):
    return vc.head(k).sort_values(ascending=True)


def _trend_statuses(df, group: str, labels: list[str]) -> dict[str, str]:
    focal = _counts_for_group(df, group)
    other_group = "comp_or_hybrid" if group == "biological" else "biological"
    other = _counts_for_group(df, other_group)
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
    ap.add_argument("--out_dir",  default="output/species")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, jsonl=args.jsonl, year_min=args.year_min, year_max=args.year_max))
    assert "species" in df.columns

    n_bio  = int((df["group2"] == "biological").sum())
    n_comp = int((df["group2"] == "comp_or_hybrid").sum())

    vc_bio  = _counts_for_group(df, "biological")
    vc_comp = _counts_for_group(df, "comp_or_hybrid")
    bio  = _topk(vc_bio, args.k)
    comp = _topk(vc_comp, args.k)

    bio_status  = _trend_statuses(df, "biological", bio.index.tolist())
    comp_status = _trend_statuses(df, "comp_or_hybrid", comp.index.tolist())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 7.0), sharex=True)

    # xmin=100: both panels' minimum values are well above 100, so no data
    # are clipped. font_scale/row_spacing are oversized because this figure
    # is placed at a narrower \linewidth than its native size in the paper.
    lollipop_row(ax1, bio, f"Biological (n = {n_bio:,})", PALETTE["biological"], bio_status,
                 show_xlabel=False, xlabel="Papers (log scale)", xmin=100, log_scale=True, font_scale=1.5, row_spacing=1.5)
    lollipop_row(ax2, comp, f"Computational (n = {n_comp:,})", PALETTE["comp_or_hybrid"], comp_status,
                 show_xlabel=True, xlabel="Papers (log scale)", xmin=100, log_scale=True, font_scale=1.5, row_spacing=1.5)

    fig.suptitle("Most studied species by research tradition",
                 fontsize=17, fontweight="bold", fontfamily="sans-serif", y=1.03)
    # Marker legend (arrow meanings) is given once in the running text, not repeated here.
    fig.subplots_adjust(left=0.3, right=0.98, top=0.86, bottom=0.09, hspace=0.36)

    out_pdf = out_dir / "top_species_dual.pdf"
    out_png = out_dir / "top_species_dual.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")

    lines = [f"n biological = {n_bio:,} papers, n computational = {n_comp:,} papers", ""]
    lines.append("Biological panel (top {}):".format(args.k))
    for l, v in bio.sort_values(ascending=False).items():
        lines.append(f"  {l:30s} {int(v):8,d}  [{bio_status[l]}]")
    lines.append("")
    lines.append("Computational panel (top {}):".format(args.k))
    for l, v in comp.sort_values(ascending=False).items():
        lines.append(f"  {l:30s} {int(v):8,d}  [{comp_status[l]}]")
    write_stats_txt(out_dir / "top_species_dual_stats.txt",
                     "Most studied species by research tradition -- exact counts", lines)


if __name__ == "__main__":
    main()
