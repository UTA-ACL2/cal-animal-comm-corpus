from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# ── Colorblind-safe palette ────────────────────────────────────────────────────
# biological / traditional  →  teal
# computational / hybrid    →  coral-orange
# pre-split                 →  slate-blue
# post-split                →  amber
PALETTE = {
    "biological":     "#2E8B8B",   # teal
    "comp_or_hybrid": "#D4622A",   # coral-orange (computational papers; hybrid is broken out separately where the 3-way split is shown)
    "computational":  "#D4622A",   # coral-orange, same as comp_or_hybrid -- used only where hybrid is shown as its own 3rd stream
    "hybrid":         "#8B4C8B",   # muted purple -- distinct 3rd stream, ~1.2% of the corpus
    "pre":            "#4C6E8C",   # slate-blue
    "post":           "#C4872A",   # amber
    "neutral":        "#555555",
}

# Ordered list useful for cycling (donut wedges, etc.)
COLORS_CYCLE = [
    "#2E8B8B", "#D4622A", "#4C6E8C", "#C4872A",
    "#6A9E5B", "#8B4C8B", "#B5773A", "#3A7AB5",
    "#C45A5A", "#5A8C6E",
]


def apply_style() -> None:
    matplotlib.rcParams.update({
        # Use a serif font family for publication quality; falls back gracefully
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          9.5,
        "axes.titlesize":     10.5,
        "axes.titleweight":   "bold",
        "axes.labelsize":     9.5,
        "xtick.labelsize":    8.8,
        "ytick.labelsize":    8.8,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "pdf.fonttype":       42,   # embed fonts as TrueType in PDF
        "ps.fonttype":        42,
        "axes.prop_cycle":    matplotlib.cycler(color=COLORS_CYCLE),
    })


def save_both(fig: plt.Figure, out_pdf, out_png, tight: bool = True) -> None:
    if tight:
        fig.tight_layout(pad=0.6)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=600)
