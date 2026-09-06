from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

from _plot_style import PALETTE

TRAD_CATS = {"biological", "computational", "hybrid"}
TREND_SAME_EPS_PCT = 1.0
TREND_SYMBOLS = {
    "up": "↑",
    "down": "↓",
    "same": "-",
    "new": "*",
}
TREND_COLORS = {
    "up": "#2E8B57",
    "down": "#C43C39",
    "same": "#C4872A",
    "new": "#6A4C93",
}

# CANON_MAP: hard overrides for inflected domain terms post-lemmatization
CANON_MAP: dict[str, str] = {
    "bats": "bat", "whales": "whale", "dolphins": "dolphin", "birds": "bird",
    "frogs": "frog", "mice": "mouse", "rats": "rat", "seals": "seal",
    "finches": "finch", "sparrows": "sparrow", "monkeys": "monkey",
    "crickets": "cricket", "insects": "insect", "fishes": "fish",
    "songbirds": "songbird", "warblers": "warbler", "porpoises": "porpoise",
    "primates": "primate", "cetaceans": "cetacean", "elephants": "elephant",
    "orcas": "orca", "bees": "bee", "grasshoppers": "grasshopper",
    "gerbils": "gerbil", "canaries": "canary", "wolves": "wolf",
    "crows": "crow", "ravens": "raven", "parrots": "parrot",
    "chimpanzees": "chimpanzee", "gorillas": "gorilla",
    "males": "male", "females": "female",
    "responses": "response", "neurons": "neuron", "pulses": "pulse",
    "clicks": "click", "whistles": "whistle", "chirps": "chirp",
    "repertoires": "repertoire", "syllables": "syllable",
    "echolocate": "echolocation",
    "focusing": "focus", "producing": "produce", "produced": "produce",
    "measuring": "measure", "measured": "measure",
    "detecting": "detect", "detected": "detect",
    "recognizing": "recognize", "recognised": "recognize",
    "monitoring": "monitor",
}


def canonicalize(token: str) -> str:
    """Map a lowercased token to its canonical domain form."""
    return CANON_MAP.get(token, token)


# ── Authority / year suffix patterns in taxonomic names ───────────────────────
# e.g. "Taeniopygia guttata (Vieillot, 1817)"  →  "taeniopygia guttata"
_AUTHORITY_RE = re.compile(r"\s*\([^)]*\d{4}[^)]*\)\s*$")
_YEAR_ONLY_RE = re.compile(r",?\s*\d{4}\s*$")

# Known common-name / synonym mappings (lowercase binomial → preferred label)
_SPECIES_SYNONYMS: dict[str, str] = {
    "taeniopygia guttata":          "Zebra finch",
    "poephila guttata":             "Zebra finch",        # older synonym
    "mus musculus":                 "House mouse",
    "rattus norvegicus":            "Norway rat",
    "pan troglodytes":              "Chimpanzee",
    "tursiops truncatus":           "Bottlenose dolphin",
    "physeter macrocephalus":       "Sperm whale",
    "megaptera novaeangliae":       "Humpback whale",
    "orcinus orca":                 "Killer whale",
    "apis mellifera":               "Honeybee",
    "drosophila melanogaster":      "Fruit fly",
    "columba livia":                "Rock pigeon",
    "melospiza melodia":            "Song sparrow",
    "serinus canaria":              "Domestic canary",
    "canis lupus familiaris":       "Dog",
    "canis familiaris":             "Dog",
    "felis catus":                  "Domestic cat",
    "xenopus laevis":               "African clawed frog",
    "homo sapiens":                 "Human",
}

# Simple English plural → singular rules (applied after lower-casing)
_PLURAL_RULES = [
    (re.compile(r"idae$"),    "idae"),      # family endings: keep as-is
    (re.compile(r"inae$"),    "inae"),      # subfamily: keep as-is
    (re.compile(r"ves$"),     "f"),         # wolves → wolf
    (re.compile(r"ies$"),     "y"),         # butterflies → butterfly
    (re.compile(r"ses$"),     "s"),         # grasses → grass
    (re.compile(r"(?<!s)s$"), ""),          # generic trailing -s
]


def norm_species_label(x: object) -> str:
    """Normalize a species label to a canonical, display-ready string."""
    if x is None:
        return "Other"
    s = str(x).strip()
    if not s:
        return "Other"

    sl = s.lower()

    # Null-ish values
    _nullish = {
        "na", "n/a", "none", "null", "unknown", "unk", "undefined",
        "not_applicable", "not applicable", "not-applicable", "n.a.",
        "notapplicable", "other", "others", "other species", "misc",
        "miscellaneous", "unspecified", "multiple", "various", "several",
    }
    if sl in _nullish or sl.startswith("other "):
        return "Other"

    # Strip taxonomic authority + year: "Taeniopygia guttata (Vieillot, 1817)"
    cleaned = _AUTHORITY_RE.sub("", s).strip()
    cleaned = _YEAR_ONLY_RE.sub("", cleaned).strip()
    key = cleaned.lower()

    # Check synonym table
    if key in _SPECIES_SYNONYMS:
        return _SPECIES_SYNONYMS[key]

    # Singularize common name (e.g. "zebra finches" → "zebra finch")
    words = key.split()
    if len(words) >= 1:
        last = words[-1]
        for pattern, replacement in _PLURAL_RULES:
            if pattern.search(last) and last not in {"species", "series", "lens"}:
                # Don't singularize taxonomic family/subfamily endings
                if not (last.endswith("idae") or last.endswith("inae")):
                    last = pattern.sub(replacement, last)
                    break
        words[-1] = last
        key = " ".join(words)

    # Re-check synonyms after singularization
    if key in _SPECIES_SYNONYMS:
        return _SPECIES_SYNONYMS[key]

    return key.title()


# Broad common-name taxa where individual species aren't worth distinguishing
# in a top-N chart (e.g. "fin whale" and "blue whale" both collapse to "Whale").
_TAXON_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("whale", "Whale"),
    ("porpoise", "Porpoise"),
    ("bat", "Bat"),
]


def taxon_group_label(x: object) -> str:
    """Coarser grouping on top of norm_species_label: collapses individual
    whale/porpoise/bat species into one genus-level bucket each."""
    label = norm_species_label(x)
    if label == "Other":
        return label
    ll = label.lower()
    for pattern, group in _TAXON_GROUP_PATTERNS:
        if pattern in ll:
            return group
    return label


# ── Venue label shortening ─────────────────────────────────────────────────
# OpenAlex source display_name sometimes appends the hosting
# publisher/organization in parentheses (e.g. "bioRxiv (Cold Spring Harbor
# Laboratory)", "Zenodo (CERN European Organization for Nuclear Research)").
# That suffix is redundant for a bar-chart y-axis label, so strip it generally.
_VENUE_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")

# A few venue titles are long enough that even after the parenthetical strip
# they still dominate the label column; abbreviate those specifically (the
# expansion is given once in the figure caption / surrounding text).
_VENUE_ABBREVIATIONS: dict[str, str] = {
    "the journal of the acoustical society of america": "J. Acoust. Soc. Am.",
    "proceedings of the royal society b biological sciences": "Proc. R. Soc. B",
}


def short_venue_label(x: object) -> str:
    """Normalize a venue/source display name for compact display: drop a
    trailing '(Publisher/Org Name)' suffix, then apply a small abbreviation
    table for a few titles that are long even without it."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = _VENUE_PAREN_SUFFIX_RE.sub("", s).strip()
    abbrev = _VENUE_ABBREVIATIONS.get(s.lower())
    return abbrev if abbrev is not None else s


@dataclass(frozen=True)
class LoadSpec:
    config: str | None = None
    jsonl: str | None = None
    year_min: int = 1950
    year_max: int = 2026
    col_year: str = "year"
    col_trad: str = "research_tradition"


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}. Present: {list(df.columns)}"


def _normalize_tradition(x: object) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    mapping = {
        "biological": "biological",
        "bio": "biological",
        "biology": "biological",
        "computational": "computational",
        "comp": "computational",
        "ml": "computational",
        "nlp": "computational",
        "hybrid": "hybrid",
        "bio+comp": "hybrid",
        "computational+biological": "hybrid",
        "biological+computational": "hybrid",
    }
    out = mapping.get(s, s)
    return out if out in TRAD_CATS else "unknown"


def load_df(spec: LoadSpec) -> pd.DataFrame:
    assert spec.jsonl, "LoadSpec.jsonl must point at final_relevant.parquet"
    assert str(spec.jsonl).endswith(".parquet"), "Only the released Parquet dataset is supported"
    df = pd.read_parquet(spec.jsonl)
    assert not df.empty, f"No rows loaded from {spec.jsonl}"

    _require_cols(df, [spec.col_year, spec.col_trad])

    df = df.copy()
    df[spec.col_year] = pd.to_numeric(df[spec.col_year], errors="coerce")
    df = df.dropna(subset=[spec.col_year])
    df[spec.col_year] = df[spec.col_year].astype(int)
    df = df[(df[spec.col_year] >= spec.year_min) & (df[spec.col_year] <= spec.year_max)]
    assert len(df) > 0, f"No rows in range {spec.year_min} to {spec.year_max}."

    df["trad3"] = df[spec.col_trad].map(_normalize_tradition)
    assert df["trad3"].isin(list(TRAD_CATS)).any(), (
        "No rows mapped to a known tradition. "
        f"Top values: {df['trad3'].value_counts().head(10).to_dict()}"
    )

    df["group2"] = df["trad3"].map(
        lambda t: "biological" if t == "biological"
        else ("comp_or_hybrid" if t in {"computational", "hybrid"} else "drop")
    )
    df = df[df["group2"].isin(["biological", "comp_or_hybrid"])].copy()
    assert len(df) > 0, "All rows were dropped during group2 filtering."
    return df


def assert_listlike(df: pd.DataFrame, col: str) -> None:
    assert col in df.columns, f"Missing column: {col}"
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return

    def ok_type(x: object) -> bool:
        return (
            x is None
            or isinstance(x, str)
            or isinstance(x, list)
            or isinstance(x, tuple)
            or isinstance(x, np.ndarray)
        )

    ok = non_null.map(ok_type).all()
    assert bool(ok), f"Column {col} contains unsupported values. Example: {non_null.iloc[0]!r}"


def explode_list_col(df: pd.DataFrame, col: str) -> pd.Series:
    assert col in df.columns, f"Missing column: {col}"
    s = df[col]
    out: list[str] = []

    for v in s:
        if v is None:
            continue

        if isinstance(v, str):
            vv = v.strip()
            if vv:
                out.append(vv)
            continue

        if isinstance(v, (list, tuple, np.ndarray)):
            for x in v:
                if isinstance(x, str):
                    xx = x.strip()
                    if xx:
                        out.append(xx)
            continue

        assert False, f"Unexpected type in column {col}: {type(v)} value={v!r}"

    return pd.Series(out, dtype=str)


def split_early_late(df: pd.DataFrame, year_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame]:
    assert year_col in df.columns, f"Missing year column: {year_col}"
    years = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
    assert len(years) >= 2, f"Need at least two unique years to compute a trend for {year_col}"

    split_idx = len(years) // 2
    early_years = set(years[:split_idx])
    late_years = set(years[split_idx:])
    early = df[df[year_col].isin(early_years)].copy()
    late = df[df[year_col].isin(late_years)].copy()
    assert len(early) > 0 and len(late) > 0, "Early/late split produced an empty partition."
    return early, late


def value_counts_scalar(
    df: pd.DataFrame,
    col: str,
    normalizer=None,
    drop_labels: set[str] | None = None,
) -> pd.Series:
    assert col in df.columns, f"Missing column: {col}"
    s = df[col].dropna()
    if normalizer is not None:
        s = s.map(normalizer)
    s = s.map(lambda x: str(x).strip() if x is not None else "")
    s = s[s != ""]
    if drop_labels:
        s = s[~s.isin(drop_labels)]
    return s.value_counts()


def value_counts_list(
    df: pd.DataFrame,
    col: str,
    normalizer=None,
    drop_labels: set[str] | None = None,
) -> pd.Series:
    assert_listlike(df, col)
    s = explode_list_col(df, col)
    if normalizer is not None:
        s = s.map(normalizer)
    s = s.map(lambda x: str(x).strip() if x is not None else "")
    s = s[s != ""]
    if drop_labels:
        s = s[~s.isin(drop_labels)]
    return s.value_counts()


def relative_rank_status_for_labels(
    focal_labels: list[str],
    other_labels: list[str],
    labels: list[str],
) -> dict[str, str]:
    focal_rank = {label: idx for idx, label in enumerate(focal_labels)}
    other_rank = {label: idx for idx, label in enumerate(other_labels)}

    out: dict[str, str] = {}
    for label in labels:
        f_rank = focal_rank.get(label)
        o_rank = other_rank.get(label)
        assert f_rank is not None, f"Label {label!r} missing from focal rank set."

        if o_rank is None:
            out[label] = "new"
        elif f_rank == o_rank:
            out[label] = "same"
        elif f_rank < o_rank:
            out[label] = "up"
        else:
            out[label] = "down"
    return out


def write_stats_txt(out_path, title: str, lines: list[str]) -> None:
    """Plain-text sidecar with the exact numbers behind a figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title.strip() + "\n")
        f.write("=" * len(title.strip()) + "\n\n")
        for line in lines:
            f.write(line.rstrip("\n") + "\n")
    print(f"Wrote: {out_path}")


def dumbbell_dual(
    ax,
    bio_full: pd.Series,
    comp_full: pd.Series,
    top_n: int = 10,
    xlabel: str = "Papers",
    bio_color: str = PALETTE["biological"],
    comp_color: str = PALETTE["comp_or_hybrid"],
    value_fmt: str = "{:,.0f}",
) -> tuple[list[str], list[float], list[float]]:
    """Single-panel biological-vs-computational comparison: one row per label
    (union of each group's top_n, de-duplicated), two dots (bio/comp) joined by
    a connector line, rows sorted by combined value."""
    bio_top_labels = bio_full.head(top_n).index.tolist()
    comp_top_labels = comp_full.head(top_n).index.tolist()

    labels: list[str] = []
    for l in bio_top_labels + comp_top_labels:
        if l not in labels:
            labels.append(l)

    def _combined(l: str) -> float:
        return float(bio_full.get(l, 0.0)) + float(comp_full.get(l, 0.0))

    labels.sort(key=_combined, reverse=True)

    bio_vals = [float(bio_full.get(l, 0.0)) for l in labels]
    comp_vals = [float(comp_full.get(l, 0.0)) for l in labels]
    y = list(range(len(labels)))

    # Nudge near-tied dots apart vertically so both markers stay visible.
    max_val = max(bio_vals + comp_vals) if (bio_vals or comp_vals) else 0.0
    tie_eps = max_val * 0.015
    dy = 0.11

    bio_y: list[float] = []
    comp_y: list[float] = []
    for yi, bv, cv in zip(y, bio_vals, comp_vals):
        if abs(bv - cv) <= tie_eps:
            bio_y.append(yi - dy)
            comp_y.append(yi + dy)
        else:
            bio_y.append(yi)
            comp_y.append(yi)

    for yi, bv, cv, yb, yc in zip(y, bio_vals, comp_vals, bio_y, comp_y):
        ax.plot([bv, cv], [yb, yc], color="#B5B5B5", linewidth=1.1, zorder=1, solid_capstyle="round")

    ax.scatter(bio_vals, bio_y, color=bio_color, s=40, zorder=3,
               edgecolor="white", linewidth=0.6, label="Biological")
    ax.scatter(comp_vals, comp_y, color=comp_color, s=40, zorder=3,
               edgecolor="white", linewidth=0.6, label="Computational")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8, handletextpad=0.3)

    return labels, bio_vals, comp_vals


def bucket_small_slices(vc: pd.Series, min_pct: float = 1.0, other_label: str = "Other") -> pd.Series:
    """Fold slices below min_pct of the total into `other_label`."""
    total = float(vc.sum())
    assert total > 0, "bucket_small_slices got an empty/zero-sum series"
    keep = vc[vc / total * 100.0 >= min_pct]
    dropped = vc[vc / total * 100.0 < min_pct]
    if len(dropped) == 0:
        return keep
    out = keep.copy()
    extra = int(dropped.sum())
    if other_label in out.index:
        out[other_label] = out[other_label] + extra
    else:
        out[other_label] = extra
    return out.sort_values(ascending=False)


def lollipop_row(
    ax,
    series: pd.Series,
    row_label: str,
    color: str,
    statuses: dict[str, str],
    show_xlabel: bool,
    xlabel: str,
    xmin: float,
    log_scale: bool,
    font_scale: float = 1.0,
    row_spacing: float = 1.0,
    tick_scale: float = 1.0,
) -> None:
    """One row of a two-row (biological over computational) stacked lollipop
    figure sharing a single x-axis. font_scale scales all text in the row;
    row_spacing scales the vertical gap between rows; tick_scale additionally
    scales only the tick labels, independent of font_scale."""
    y_pos = [i * row_spacing for i in range(len(series))]
    ax.hlines(y_pos, xmin=xmin, xmax=series.values, color=color, linewidth=1.6, alpha=0.7)
    ax.scatter(series.values, y_pos, color=color, s=48, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(series.index.tolist(), fontsize=9.0 * font_scale * tick_scale)
    ax.set_title(row_label, loc="left", fontsize=11.5 * font_scale, fontweight="bold", color=color)
    if log_scale:
        ax.set_xscale("log")
        # Leave a margin before xmin so the rank-arrow marker isn't flush
        # against the axis spine.
        ax.set_xlim(left=xmin * 0.75)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--", which="both" if log_scale else "major")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if log_scale:
        ax.tick_params(axis="x", which="minor", labelsize=0)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10 * font_scale)
        ax.tick_params(axis="x", labelsize=9 * font_scale * tick_scale)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    add_yaxis_status_markers(ax, statuses, gap_points=4.0 * font_scale, fontsize=9.5 * font_scale)


def add_yaxis_status_markers(ax, status_by_label: dict[str, str], gap_points: float = 4.0, fontsize: float = 9.5) -> None:
    fig = ax.figure
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        status = status_by_label.get(label)
        if status is None:
            continue
        x, y = tick.get_position()
        ax.text(
            x,
            y,
            TREND_SYMBOLS[status],
            transform=offset_copy(tick.get_transform(), fig=fig, x=gap_points, y=0.0, units="points"),
            ha="left",
            va="center",
            color=TREND_COLORS[status],
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
        )
