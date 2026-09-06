#!/usr/bin/env python3
"""
Application domains -- compact pie chart with a left-side wrapped legend
(matching colors, no leader lines).

Research settings and communication modalities are not plotted here: settings
has too few categories to justify a figure, and modalities is so
acoustic-dominated that a pie/donut just wastes space on a near-solid disk.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _plot_utils import LoadSpec, load_df, explode_list_col, assert_listlike, bucket_small_slices, write_stats_txt
from _plot_style import apply_style, save_both, COLORS_CYCLE


def _norm_label(x: str) -> str:
    assert isinstance(x, str)
    s  = x.strip()
    if not s:
        return "Other"
    sl = s.lower()
    nullish = {
        "na", "n/a", "none", "null", "unknown", "unk", "undefined",
        "not_applicable", "not applicable", "not-applicable", "n.a.",
        "notapplicable",
    }
    if sl in nullish or sl == "other" or sl.startswith("other ") \
            or sl in {"misc", "miscellaneous"}:
        return "Other"
    return s


def _value_counts_normalized(s: pd.Series) -> pd.Series:
    assert isinstance(s, pd.Series)
    if len(s) == 0:
        return pd.Series([], dtype=int)
    s2 = s.map(lambda v: _norm_label(v) if isinstance(v, str) else "Other")
    return s2.value_counts()


def pie_panel_left_legend(fig, ax, counts: pd.Series, title: str, top_n: int = 10, subtitle: str | None = None) -> pd.Series:
    counts = counts[counts > 0]
    assert len(counts) > 0, f"No counts for {title}"

    top  = counts.head(top_n).copy()
    rest = int(counts.iloc[top_n:].sum())
    if rest > 0:
        if "Other" in top.index:
            top.loc["Other"] += rest
        else:
            top = pd.concat([top, pd.Series({"Other": rest})])

    # Fold anything under 1% of the shown total into "Other" too -- avoids
    # illegible slivers that round to 0.0% in the label.
    top = bucket_small_slices(top, min_pct=1.0)

    # Put "Other" last
    if "Other" in top.index:
        other_val = top.pop("Other")
        top["Other"] = other_val

    total = int(top.sum())
    n_slices = len(top)

    colors = [COLORS_CYCLE[i % len(COLORS_CYCLE)] for i in range(n_slices)]
    if "Other" in top.index:
        colors[list(top.index).index("Other")] = "#aaaaaa"

    wedges, _ = ax.pie(
        top.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=1.0,
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
    )
    ax.set_title(title, fontsize=12.6, fontweight="bold", pad=16)
    if subtitle:
        # Drawn separately (not bold) so the sample-size annotation doesn't
        # visually compete with the bold panel title above it.
        ax.text(0.5, 1.09, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5, fontweight="normal")
    ax.set(aspect="equal")

    # Left-side legend with matching color swatches, wrapped to 2 lines for
    # long labels -- keeps text large/readable without a below-plot legend
    # eating vertical space or in-slice text getting illegibly small.
    labels = []
    for name, val in top.items():
        pct = val / total * 100.0
        wrapped = "\n".join(textwrap.wrap(str(name), width=22, max_lines=3))
        labels.append(f"{wrapped}  ({pct:.1f}%)")

    ax.legend(
        wedges, labels,
        loc="center right",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=10.5,
        handlelength=1.1,
        handletextpad=0.6,
        labelspacing=0.85,
        borderaxespad=0.0,
    )

    return top


def make_domains(df: pd.DataFrame, col: str, title: str, out_dir: Path, top_n: int) -> None:
    assert_listlike(df, col)
    raw = explode_list_col(df, col)
    vc  = _value_counts_normalized(raw)
    assert len(vc) > 0, f"No values for {col}"

    n_total = len(df)
    # Fixed-size axes (not tight_layout) -- tight_layout fights the pie's
    # fixed aspect="equal" when the legend extends outside the axes box.
    # bbox_inches="tight" on save still crops the figure to content.
    fig, ax = plt.subplots(figsize=(5.8, 3.15))
    fig.subplots_adjust(left=0.46, right=0.99, top=0.80, bottom=0.03)
    top = pie_panel_left_legend(fig, ax, vc, title, top_n=top_n, subtitle=f"(n = {n_total:,})")

    out_pdf = out_dir / f"{col}_donut.pdf"
    out_png = out_dir / f"{col}_donut.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")

    total = int(top.sum())
    lines = [f"n papers (facet mentions) = {total:,}  [documents = {n_total:,}]", ""]
    for name, val in top.items():
        lines.append(f"  {name:32s} {int(val):7,d}  ({val/total*100:5.1f}%)")
    write_stats_txt(out_dir / f"{col}_donut_stats.txt", f"{title} -- exact counts", lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=False, default=None)
    ap.add_argument("--jsonl",    required=False, default="../dataset/final_relevant.parquet")
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--top_n",    type=int, default=10)
    ap.add_argument("--out_dir",  default="output/donuts")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, jsonl=args.jsonl, year_min=args.year_min, year_max=args.year_max))

    make_domains(df, "domains", "Application domains", out_dir, args.top_n)


if __name__ == "__main__":
    main()
