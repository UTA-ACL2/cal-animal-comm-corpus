#!/usr/bin/env python3
"""
text_features.py — builds the classifier input text for a paper record.

Shared by filter_bert.py (inference) so that the exact same field-selection
and formatting logic the SciBERT classifier was trained against is used at
scoring time. Config-driven via configs/bert_classifier.yaml's input_fields
section.
"""

from __future__ import annotations


def reconstruct_abstract(inv) -> str:
    if not inv or not isinstance(inv, dict):
        return ""
    try:
        pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
        pairs.sort()
        return " ".join(w for _, w in pairs)
    except Exception:
        return ""


def _safe_str(x):
    """Robust string extractor for noisy OpenAlex fields."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("display_name", "name", "label", "title"):
            v = x.get(k)
            if isinstance(v, str):
                return v
    if isinstance(x, (list, tuple)):
        parts = [p for p in x if isinstance(p, str)]
        if parts:
            return "; ".join(parts)
    return ""


def build_text(paper: dict, cfg: dict) -> str:
    f = cfg.get("input_fields", {})
    parts = []

    if f.get("use_title", True):
        t = (paper.get("title") or paper.get("display_name") or "").strip()
        if t:
            parts.append(f"[TITLE] {t}")

    if f.get("use_venue", True):
        v = (paper.get("venue") or "").strip()
        if v:
            parts.append(f"[VENUE] {v}")

    if f.get("use_primary_topic_field", True):
        pt = paper.get("primary_topic")
        fn = ""
        if isinstance(pt, dict):
            # pt may look like {"field": "Bioacoustics", ...} or
            # {"field": {"display_name": "Bioacoustics", ...}, ...}
            raw = pt.get("field") or pt.get("subfield") or ""
            fn = _safe_str(raw).strip()
        elif isinstance(pt, str):
            fn = pt.strip()
        if fn:
            parts.append(f"[FIELD] {fn}")

    if f.get("use_keywords", True):
        ks = []
        for k in (paper.get("keywords") or [])[: f.get("max_keywords", 8)]:
            val = _safe_str(k) if isinstance(k, dict) else str(k)
            val = val.strip()
            if val:
                ks.append(val)
        if ks:
            parts.append(f"[KEYWORDS] {'; '.join(ks)}")

    if f.get("use_abstract", True):
        ab = (paper.get("abstract") or "").strip()
        if not ab:
            ab = reconstruct_abstract(paper.get("abstract_inverted_index"))
        if ab:
            parts.append(f"[ABSTRACT] {ab}")

    return " ".join(parts)
