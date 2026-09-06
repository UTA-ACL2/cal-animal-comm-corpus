#!/usr/bin/env python3
"""
filter_bert.py — SciBERT relevance classification, producer-consumer
streaming pipeline with multi-process/multi-GPU parallelism.

Key properties:
  - Multi-GPU support with per-GPU config (batch_size, chunk_size, queue_max, num_readers)
  - File-level resume: completed .gz files are skipped entirely on restart
  - Completed-files manifest keyed by filename (not worker_id) — safe to change
    gpu_configs between runs without losing progress
  - Seed-stable file partitioning: same seed + same file list = same assignment
  - Writer has a 5-min timeout so it never hangs if a GPU worker is hard-killed
  - On --resume, pass1_bert.jsonl is backed up automatically before appending
  - Backup/merge utility modes baked in (--backup-only, --merge-only)

RUN (two GPUs, different memory budgets):
    CUDA_VISIBLE_DEVICES=0,1 python -u filter_bert.py \\
      --snapshot_dir /path/to/openalex-works \\
      --output_dir   /path/to/corpus_output \\
      --model        /path/to/hf_model \\
      --config       bert_classifier.yaml \\
      --gpu_configs  "0:batch=1024,chunk=16384,queue=3,readers=2" \\
                     "1:batch=2048,chunk=16384,queue=4,readers=3" \\
      --threshold    0.45 \\
      --year_start   2000 \\
      --year_end     2026 \\
      --seed         42 \\
      --resume

RUN (single GPU):
    CUDA_VISIBLE_DEVICES=0 python -u filter_bert.py \\
      --snapshot_dir /path/to/openalex-works \\
      --output_dir   /path/to/corpus_output \\
      --model        /path/to/hf_model \\
      --config       bert_classifier.yaml \\
      --gpu_configs  "0:batch=2048,chunk=16384,queue=4,readers=2" \\
      --threshold    0.45 \\
      --resume

BACKUP + MERGE ONLY (run before killing a live pipeline):
    python -u filter_bert.py \\
      --output_dir /path/to/corpus_output \\
      --backup-only

MERGE DEDUP ONLY:
    python -u filter_bert.py \\
      --output_dir /path/to/corpus_output \\
      --merge-only

NOTE ON GPU CONFIG CHANGES BETWEEN RUNS:
    You can freely change --gpu_configs and --num_readers between runs.
    The completed-files manifest is keyed by the .gz FILENAME (not worker index),
    so file-level skipping survives any config change. The dedup.tsv also survives
    any config change. Only ensure --seed stays the same so file assignment is
    reproducible (prevents two workers being assigned the same file if you add GPUs).
"""

import argparse
import hashlib
import os
import sys
import time
import shutil
import signal
import ctypes
import traceback
import subprocess
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from datetime import datetime
from glob import glob
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict

import torch
import numpy as np
import yaml
import orjson
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_features import build_text


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

KEEP_FIELDS = frozenset({
    "id", "doi", "display_name", "title",
    "publication_year", "publication_date",
    "type", "language",
    "primary_location", "locations", "best_oa_location",
    "open_access", "authorships",
    "topics", "concepts", "keywords",
    "abstract_inverted_index",
    "biblio", "cited_by_count", "counts_by_year",
    "is_retracted", "is_paratext",
    "referenced_works", "related_works",
    "grants", "mesh", "primary_topic",
    "indexed_in", "updated_date", "created_date",
})

ALLOWED_TYPES = frozenset({
    "article",              # standard peer-reviewed journal article
    "preprint",             # arXiv / bioRxiv / ESSOAr / etc.
    "book",                 # full monographs & textbooks
    "book-chapter",         # chapters in edited volumes
    "dataset",              # published data papers / repositories
    "dissertation",         # PhD / master's theses
    "review",               # systematic / narrative review articles
    "proceedings-article",  # ACL, NeurIPS, ICML, etc.
    "report",               # technical reports
    "standard",             # ISO / IEEE standards
    "other",                # catch-all — BERT will filter irrelevant ones
})

DEDUP_SYNC_INTERVAL    = 100_000
COUNTER_FLUSH_INTERVAL = 1_000

# Writer will give up waiting for GPU output after this many seconds.
# Protects against the writer hanging forever if a GPU worker is hard-killed.
WRITER_QUEUE_TIMEOUT_S = 1500

# Single shared completed-files manifest in output_dir (basename-keyed)
COMPLETED_MANIFEST_NAME = "completed_files.txt"


# ─────────────────────────────────────────────────────────────────────────────
# PER-GPU CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuConfig:
    """Configuration for one GPU worker and its dedicated reader pool."""
    device_id:   int = 0
    batch_size:  int = 2048
    chunk_size:  int = 16384
    queue_max:   int = 4
    num_readers: int = 2

    @property
    def device(self) -> str:
        return f"cuda:{self.device_id}"

    def __str__(self):
        return (f"cuda:{self.device_id} | batch={self.batch_size} "
                f"chunk={self.chunk_size} queue={self.queue_max} "
                f"readers={self.num_readers}")


def parse_gpu_config(spec: str) -> GpuConfig:
    """
    Parse a GPU config string: "DEVICE_ID:batch=N,chunk=N,queue=N,readers=N"
    All key=value pairs are optional; GpuConfig defaults fill missing ones.
    Examples:
        "0"                                          -> cuda:0 all defaults
        "0:batch=1024"                               -> cuda:0, batch=1024
        "1:batch=2048,chunk=16384,queue=4,readers=3"
    """
    if ":" not in spec:
        return GpuConfig(device_id=int(spec.strip()))
    dev_str, rest = spec.split(":", 1)
    cfg = GpuConfig(device_id=int(dev_str.strip()))
    for kv in rest.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Malformed key=value in GPU spec: {kv!r} (from {spec!r})")
        k, v = kv.split("=", 1)
        k, v = k.strip(), int(v.strip())
        if   k == "batch":   cfg.batch_size  = v
        elif k == "chunk":   cfg.chunk_size  = v
        elif k == "queue":   cfg.queue_max   = v
        elif k == "readers": cfg.num_readers = v
        else:
            raise ValueError(f"Unknown GPU config key: {k!r} in {spec!r}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# RESUME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def repair_torn_tail(path: str, window: int = 8 * 1024 * 1024) -> int:
    """
    Append-only output files can only ever be corrupted at the very end (the
    line being written when a hard kill/crash hit). Detect that by checking
    whether the file ends with a newline; if not, find the start of the last
    (torn) line within the last `window` bytes and truncate it off. Returns
    the number of bytes removed (0 if the file already ended cleanly).
    """
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0
        start = max(0, size - window)
        f.seek(start)
        buf = f.read(size - start)
        idx = buf.rfind(b"\n")
        if idx == -1:
            raise RuntimeError(
                f"{path}: no newline found in last {window} bytes — "
                f"cannot safely repair torn tail (file may be non-JSONL "
                f"or window too small)."
            )
        new_size = start + idx + 1
        f.truncate(new_size)
        return size - new_size


def load_dedup_tsv(tsv_path: str) -> dict:
    """
    Return {key: (year, updated_date)}. `year` is kept only as an informational
    column; `updated_date` alone is the priority signal. Each OpenAlex
    updated_date=... partition is a full record snapshot, not a diff, so any
    real content change (preprint -> published, a corrected publication_year,
    an enriched abstract, etc.) necessarily produces a newer updated_date.
    Comparing by year first would actually be wrong: it could keep a stale
    row over a later scrape that *lowers* publication_year (e.g. fixing a bad
    value) — the latest scrape should always win unconditionally.
    Backward-compatible with the old 2-column (key\\tyear) format: missing
    updated_date defaults to "" which sorts before any real ISO date, so any
    record actually encountered again (which only happens if it was really
    re-touched, since OpenAlex partitions only contain touched records) is
    correctly treated as newer and reprocessed.
    """
    dedup = {}
    if not os.path.exists(tsv_path):
        return dedup
    with open(tsv_path, "r", buffering=64 * 1024 * 1024) as f:
        next(f, None)  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                key, year_s = parts
                udate = ""
            elif len(parts) == 3:
                key, year_s, udate = parts
            else:
                continue
            try:
                dedup[key] = (int(year_s), udate)
            except ValueError:
                continue
    return dedup


def save_dedup_tsv(dedup: dict, tsv_path: str):
    """Write directly to destination — avoids NFS/Lustre tmp-rename failures."""
    with open(tsv_path, "w", buffering=64 * 1024 * 1024) as f:
        f.write("key\tyear\tupdated_date\n")
        for key, (year, udate) in dedup.items():
            f.write(f"{key}\t{year}\t{udate}\n")


def dedup_pass1_jsonl_by_key(jsonl_path: str) -> tuple:
    """
    Rewrite pass1_bert.jsonl keeping only the highest-updated_date row per
    key (doi if present else oa_id). year is kept alongside for reference
    but is not part of the priority comparison — see load_dedup_tsv.

    Because the dedup fix above lets a re-touched record (same doi/oa_id,
    same publication_year, later updated_date — e.g. an abstract backfilled
    onto an old paper) pass through and get reprocessed instead of being
    silently skipped, pass1_bert.jsonl can now legitimately contain more than
    one row for the same paper across separate runs (the old sparse version
    from an earlier run, plus a newer richer version from a later run that
    reprocessed it). Downstream (Stage 2 / annotate_nemotron.py) dedupes by
    oa_id using "first occurrence wins" (INSERT OR IGNORE), so without this
    merge step it would keep whichever version happens to be EARLIER in the
    file — almost always the older, less-complete one — silently defeating
    the whole point of reprocessing it. Run this after every BERT stage run.

    Returns (total_rows_before, unique_rows_after).
    """
    if not os.path.exists(jsonl_path):
        return (0, 0)

    best: dict = {}   # key -> (priority_tuple, raw_line)
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = orjson.loads(line)
            except Exception:
                continue
            oaid = (rec.get("id")  or "").strip()
            doi  = (rec.get("doi") or "").strip().lower()
            key  = doi if doi else oaid
            if not key:
                continue
            year  = rec.get("publication_year") or 0
            udate = (rec.get("updated_date") or "").strip()
            prev = best.get(key)
            if prev is None or udate >= prev[0][1]:
                best[key] = ((year, udate), line)

    if total == len(best):
        return (total, len(best))  # nothing to dedup, skip the rewrite

    tmp = jsonl_path + ".dedup_tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for _, raw_line in best.values():
            f.write(raw_line + "\n")
    os.replace(tmp, jsonl_path)
    return (total, len(best))


def merge_dedup_tsvs(worker_paths: list, merged_path: str) -> int:
    """
    Merge all per-worker dedup TSVs into one canonical TSV.
    For duplicate keys, keeps the row with the highest updated_date (see
    load_dedup_tsv for why year is not part of the priority comparison).
    Safe to include TSVs from runs with different reader counts.
    """
    merged = {}
    for wp in worker_paths:
        if not os.path.exists(wp):
            continue
        for key, val in load_dedup_tsv(wp).items():
            if key not in merged or val[1] > merged[key][1]:
                merged[key] = val
    save_dedup_tsv(merged, merged_path)
    return len(merged)


def _bucket_split_worker(args) -> int:
    """Stage 1 of merge_dedup_tsvs_bucketed(): stream one input TSV once and
    fan its rows out into per-(file,bucket) shard files by hash(key). No dict
    is built here, so memory stays O(1) regardless of file size — this
    replaces the full dict-per-file build that makes merge_dedup_tsvs() slow
    (90M+ tuple+dict-slot allocations, single-threaded; observed <10% CPU
    while running, i.e. allocation-bound, not disk-bound)."""
    file_idx, src_path, tmp_dir, num_buckets = args
    if not os.path.exists(src_path):
        return 0
    handles = [
        open(os.path.join(tmp_dir, f"b{b}_f{file_idx}.tsv"), "w", buffering=4 * 1024 * 1024)
        for b in range(num_buckets)
    ]
    n = 0
    try:
        with open(src_path, "r", buffering=64 * 1024 * 1024) as f:
            next(f, None)  # header
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                key = line.split("\t", 1)[0]
                if not key:
                    continue
                b = int(hashlib.md5(key.encode()).hexdigest(), 16) % num_buckets
                handles[b].write(line)
                handles[b].write("\n")
                n += 1
    finally:
        for h in handles:
            h.close()
    return n


def _bucket_resolve_worker(args):
    """Stage 2: resolve winners within one bucket (keyspace / num_buckets, so
    small and cheap) across every input file's shard for that bucket. Runs in
    parallel across buckets — this is where merge_dedup_tsvs_bucketed() gets
    its multi-core win, since this is the CPU-heavy part."""
    bucket_id, tmp_dir, num_files = args
    best: dict = {}
    for file_idx in range(num_files):
        shard = os.path.join(tmp_dir, f"b{bucket_id}_f{file_idx}.tsv")
        if not os.path.exists(shard):
            continue
        with open(shard, "r", buffering=16 * 1024 * 1024) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    key, year_s = parts
                    udate = ""
                elif len(parts) == 3:
                    key, year_s, udate = parts
                else:
                    continue
                try:
                    year = int(year_s)
                except ValueError:
                    continue
                prev = best.get(key)
                if prev is None or udate >= prev[1]:
                    best[key] = (year, udate)
        os.remove(shard)
    out_path = os.path.join(tmp_dir, f"resolved_{bucket_id}.tsv")
    with open(out_path, "w", buffering=4 * 1024 * 1024) as f:
        for key, (year, udate) in best.items():
            f.write(f"{key}\t{year}\t{udate}\n")
    return bucket_id, len(best)


def merge_dedup_tsvs_bucketed(
    worker_paths: list, merged_path: str, num_buckets: int = 256, workers: int | None = None,
) -> int:
    """
    Parallel, memory-bounded drop-in replacement for merge_dedup_tsvs() (same
    result: highest updated_date per key wins). merge_dedup_tsvs() is slow
    because it builds one full-corpus dict per input file, single-threaded
    (90M+ Python tuple+dict-slot allocations each) — that allocation cost,
    not disk I/O, is the bottleneck (observed <10% CPU on a real run).

    Two-stage shuffle instead: (1) stream each input file once, fan its rows
    out into `num_buckets` shard files by hash(key) — O(1) memory, pure I/O,
    parallel across input files; (2) resolve winners within each bucket
    (keyspace / num_buckets, so cheap) in parallel across buckets. Both
    stages use every available core instead of one, and peak memory is
    bounded by a single bucket's worth of keys rather than the whole corpus.
    """
    worker_paths = [p for p in worker_paths if os.path.exists(p)]
    if not worker_paths:
        save_dedup_tsv({}, merged_path)
        return 0

    cpu = os.cpu_count() or 8
    workers = workers or cpu
    tmp_dir = merged_path + ".bucket_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        split_args = [(i, p, tmp_dir, num_buckets) for i, p in enumerate(worker_paths)]
        with mp.Pool(min(len(worker_paths), cpu)) as pool:
            list(tqdm(
                pool.imap_unordered(_bucket_split_worker, split_args),
                total=len(split_args), desc="[merge] bucket split",
            ))

        resolve_args = [(b, tmp_dir, len(worker_paths)) for b in range(num_buckets)]
        total = 0
        with mp.Pool(min(workers, num_buckets)) as pool:
            for _, n in tqdm(
                pool.imap_unordered(_bucket_resolve_worker, resolve_args),
                total=num_buckets, desc="[merge] bucket resolve",
            ):
                total += n

        with open(merged_path, "wb") as out:
            out.write(b"key\tyear\tupdated_date\n")
            for b in range(num_buckets):
                part = os.path.join(tmp_dir, f"resolved_{b}.tsv")
                if not os.path.exists(part):
                    continue
                with open(part, "rb") as pf:
                    shutil.copyfileobj(pf, out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return total


def slim(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k in KEEP_FIELDS}


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETED-FILES MANIFEST
# Keyed by gz BASENAME (not worker index) so it survives gpu_config changes.
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_files(manifest_path: str) -> set:
    """Return set of dir-qualified .gz keys fully processed in a previous run."""
    completed = set()
    if not os.path.exists(manifest_path):
        return completed
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(line)
    return completed


def gz_manifest_key(gz_path: str) -> str:
    """
    Dir-qualified key for the completed-files manifest, e.g.
    "updated_date=2026-01-16/part_0000.gz". OpenAlex restarts part numbering
    at 0 in every updated_date=... directory, so a bare basename (the old
    scheme) collides across directories — e.g. "part_0000.gz" would match a
    file in a brand-new directory just because some OTHER directory's
    part_0000.gz was processed previously, silently skipping unprocessed
    data. Two path components (parent dir + basename) disambiguate while
    staying config-independent (doesn't embed snapshot_dir's absolute path,
    so the manifest still survives moving snapshot_dir).
    """
    return os.path.join(os.path.basename(os.path.dirname(gz_path)), os.path.basename(gz_path))


def mark_file_completed(manifest_path: str, gz_key: str):
    """Append a fully-processed dir-qualified .gz key to the shared manifest (append-safe)."""
    with open(manifest_path, "a") as f:
        f.write(gz_key + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION STATUS
# ─────────────────────────────────────────────────────────────────────────────

def is_published_or_accepted(raw: dict) -> bool:
    """
    Returns True if ANY location reports is_published or is_accepted.
    Records with no location data pass through (let BERT decide).
    """
    locations = raw.get("locations")
    if not locations:
        return True
    for loc in locations:
        if loc.get("is_published") or loc.get("is_accepted"):
            return True
    primary = raw.get("primary_location") or {}
    if primary.get("is_published") or primary.get("is_accepted"):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL FILTER
# ─────────────────────────────────────────────────────────────────────────────

def structural_filter(raw: dict, year_start: int, year_end: int) -> tuple:
    """Returns (passes: bool, rejection_reason: str). Ordered cheapest-first."""
    y = raw.get("publication_year")
    if isinstance(y, int) and not (year_start <= y <= year_end):
        return False, "year"
    if raw.get("is_retracted") or raw.get("is_paratext"):
        return False, "retracted"
    lang = (raw.get("language") or "").strip().lower()
    if lang and lang != "en":
        return False, "language"
    work_type = (raw.get("type") or "").strip().lower()
    if work_type and work_type not in ALLOWED_TYPES:
        return False, "type"
    if not is_published_or_accepted(raw):
        return False, "status"
    title = (raw.get("display_name") or raw.get("title") or "").strip()
    if not title:
        return False, "title"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# READER WORKER
# ─────────────────────────────────────────────────────────────────────────────

def reader_worker(
    worker_id, file_list, in_queue, cfg, chunk_size,
    year_start, year_end,
    checked_val, kept_val, deduped_val,
    filtered_type_val, filtered_status_val,
    worker_tsv, initial_dedup,
    manifest_path,       # shared completed-files manifest (basename-keyed)
    skipped_files_val,   # shared int64: total .gz files skipped at file level
):
    try:
        # Private dedup copy per worker — merged back to dedup.tsv at pipeline end
        dedup = dict(initial_dedup)

        chunk_papers = []
        chunk_texts  = []

        local_checked  = 0
        local_kept     = 0
        local_deduped  = 0
        local_ftype    = 0
        local_fstatus  = 0

        records_since_dedup_sync = 0

        # Load completed-files manifest — keyed by dir-qualified path, config-independent
        completed_basenames = load_completed_files(manifest_path)
        n_skipped = sum(1 for gz in file_list
                        if gz_manifest_key(gz) in completed_basenames)
        if n_skipped:
            with skipped_files_val.get_lock():
                skipped_files_val.value += n_skipped
            print(f"[reader-{worker_id}] Skipping {n_skipped}/{len(file_list)} "
                  f"already-completed files", flush=True)

        def flush_counters():
            nonlocal local_checked, local_kept, local_deduped, local_ftype, local_fstatus
            with checked_val.get_lock():         checked_val.value         += local_checked
            with kept_val.get_lock():            kept_val.value            += local_kept
            with deduped_val.get_lock():         deduped_val.value         += local_deduped
            with filtered_type_val.get_lock():   filtered_type_val.value   += local_ftype
            with filtered_status_val.get_lock(): filtered_status_val.value += local_fstatus
            local_checked = local_kept = local_deduped = local_ftype = local_fstatus = 0

        def try_save_dedup():
            try:
                save_dedup_tsv(dedup, worker_tsv)
            except Exception as e:
                print(f"[reader-{worker_id}] WARNING: dedup save failed (non-fatal): {e}",
                      flush=True)

        def flush_chunk():
            """Push current chunk to GPU queue. Blocks indefinitely — never drops records."""
            nonlocal chunk_papers, chunk_texts
            if not chunk_papers:
                return
            while True:
                try:
                    in_queue.put((chunk_papers, chunk_texts), timeout=30)
                    chunk_papers = []
                    chunk_texts  = []
                    return
                except Exception:
                    print(f"[reader-{worker_id}] Queue full, waiting for GPU...", flush=True)
                    time.sleep(5)

        for gz_path in file_list:
            gz_base = gz_manifest_key(gz_path)

            # File-level skip: dir-qualified key, survives gpu_config changes
            if gz_base in completed_basenames:
                continue

            file_ok = True
            try:
                with subprocess.Popen(
                    ["pigz", "-dc", gz_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                ) as proc:
                    for raw_line in proc.stdout:
                        if not raw_line:
                            continue
                        try:
                            rec = orjson.loads(raw_line)
                        except Exception:
                            continue

                        local_checked            += 1
                        records_since_dedup_sync += 1

                        passes, reason = structural_filter(rec, year_start, year_end)
                        if not passes:
                            if reason == "type":     local_ftype   += 1
                            elif reason == "status": local_fstatus += 1
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                                try_save_dedup()
                                records_since_dedup_sync = 0
                            continue

                        oaid  = (rec.get("id")  or "").strip()
                        doi   = (rec.get("doi") or "").strip().lower()
                        year  = rec.get("publication_year") or 0
                        udate = (rec.get("updated_date") or "").strip()
                        key   = doi if doi else oaid

                        if key:
                            if udate <= dedup.get(key, (-1, ""))[1]:
                                local_deduped += 1
                                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                    flush_counters()
                                if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                                    try_save_dedup()
                                    records_since_dedup_sync = 0
                                continue
                            dedup[key] = (year, udate)

                        text = (
                            build_text(rec, cfg)
                            or rec.get("display_name")
                            or rec.get("title")
                            or "unknown"
                        )
                        chunk_papers.append(slim(rec))
                        chunk_texts.append(text)
                        local_kept += 1

                        if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                            flush_counters()
                        if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                            try_save_dedup()
                            records_since_dedup_sync = 0
                        if len(chunk_papers) >= chunk_size:
                            flush_chunk()   # blocks until GPU has space — no records lost

            except Exception as e:
                print(f"[reader-{worker_id}] ERROR on {gz_path}: {e}", flush=True)
                traceback.print_exc()
                file_ok = False

            # Only mark complete if the file was processed without exception
            if file_ok:
                mark_file_completed(manifest_path, gz_base)
                completed_basenames.add(gz_base)

        flush_chunk()
        flush_counters()

    except Exception as e:
        print(f"[reader-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            save_dedup_tsv(dedup, worker_tsv)
        except Exception:
            pass
        in_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# READER WORKER — dedup-index mode (--from_index)
# Reads pre-deduped/pre-filtered Parquet chunks written by
# build_dedup_index.py instead of raw .gz partitions. No structural_filter
# or per-record dedup-key resolution needed here (build_dedup_index.py
# already did it globally) — but the dedup.tsv check is kept as cheap
# insurance in case dedup.tsv was updated by another run between when the
# index was built and when this runs.
# ─────────────────────────────────────────────────────────────────────────────

def reader_worker_from_index(
    worker_id, file_list, in_queue, cfg, chunk_size,
    checked_val, kept_val, deduped_val,
    worker_tsv, initial_dedup,
    manifest_path,
    skipped_files_val,
    model_path, max_len, batch_size,
):
    try:
        # Tokenizing here (in the reader's own process) instead of in gpu_worker
        # gives real parallelism across readers with no GIL/thread-pool
        # contention, and avoids calling the fast tokenizer's internal Rayon
        # thread pool from a background Python thread while the GPU worker's
        # main thread is doing CUDA calls concurrently — that combination
        # deadlocked (observed: gpu_worker process stuck on futex_wait_queue
        # with 246 threads, 0% GPU util, full queue).
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        dedup = dict(initial_dedup)

        chunk_papers = []
        chunk_texts  = []

        local_checked = 0
        local_kept    = 0
        local_deduped = 0

        records_since_dedup_sync = 0

        completed_basenames = load_completed_files(manifest_path)
        n_skipped = sum(1 for fp in file_list
                        if gz_manifest_key(fp) in completed_basenames)
        if n_skipped:
            with skipped_files_val.get_lock():
                skipped_files_val.value += n_skipped
            print(f"[reader-{worker_id}] Skipping {n_skipped}/{len(file_list)} "
                  f"already-completed index files", flush=True)

        def flush_counters():
            nonlocal local_checked, local_kept, local_deduped
            with checked_val.get_lock(): checked_val.value += local_checked
            with kept_val.get_lock():    kept_val.value    += local_kept
            with deduped_val.get_lock(): deduped_val.value += local_deduped
            local_checked = local_kept = local_deduped = 0

        def try_save_dedup():
            try:
                save_dedup_tsv(dedup, worker_tsv)
            except Exception as e:
                print(f"[reader-{worker_id}] WARNING: dedup save failed (non-fatal): {e}",
                      flush=True)

        def flush_chunk():
            # Tokenize into GPU-batch-sized pieces here (CPU-only) and hand
            # gpu_worker ready-to-run numpy arrays — it just pins + copies to
            # device + runs model(). This is what lets multiple readers keep
            # several batches staged ahead of the GPU at all times, instead
            # of the GPU worker tokenizing (and stalling) between batches.
            nonlocal chunk_papers, chunk_texts
            if not chunk_papers:
                return
            for s in range(0, len(chunk_papers), batch_size):
                e   = min(s + batch_size, len(chunk_papers))
                enc = tokenizer(
                    chunk_texts[s:e],
                    padding=True, truncation=True,
                    max_length=max_len, return_tensors="np",
                )
                encoded = {
                    k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids")
                    if k in enc
                }
                sub_papers = chunk_papers[s:e]
                while True:
                    try:
                        in_queue.put((sub_papers, encoded), timeout=30)
                        break
                    except Exception:
                        print(f"[reader-{worker_id}] Queue full, waiting for GPU...", flush=True)
                        time.sleep(5)
            chunk_papers = []
            chunk_texts  = []

        for pq_path in file_list:
            pq_key = gz_manifest_key(pq_path)
            if pq_key in completed_basenames:
                continue

            file_ok = True
            try:
                pf = pq.ParquetFile(pq_path)
                for batch in pf.iter_batches(batch_size=8192, columns=["payload"]):
                    for payload in batch.column("payload").to_pylist():
                        local_checked            += 1
                        records_since_dedup_sync += 1
                        try:
                            rec = orjson.loads(payload)
                        except Exception:
                            continue

                        oaid  = (rec.get("id")  or "").strip()
                        doi   = (rec.get("doi") or "").strip().lower()
                        year  = rec.get("publication_year") or 0
                        udate = (rec.get("updated_date") or "").strip()
                        key   = doi if doi else oaid

                        if key:
                            if udate <= dedup.get(key, (-1, ""))[1]:
                                local_deduped += 1
                                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                    flush_counters()
                                if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                                    try_save_dedup()
                                    records_since_dedup_sync = 0
                                continue
                            dedup[key] = (year, udate)

                        text = (
                            build_text(rec, cfg)
                            or rec.get("display_name")
                            or rec.get("title")
                            or "unknown"
                        )
                        chunk_papers.append(rec)
                        chunk_texts.append(text)
                        local_kept += 1

                        if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                            flush_counters()
                        if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                            try_save_dedup()
                            records_since_dedup_sync = 0
                        if len(chunk_papers) >= chunk_size:
                            flush_chunk()

            except Exception as e:
                print(f"[reader-{worker_id}] ERROR on {pq_path}: {e}", flush=True)
                traceback.print_exc()
                file_ok = False

            if file_ok:
                mark_file_completed(manifest_path, pq_key)
                completed_basenames.add(pq_key)

        flush_chunk()
        flush_counters()

    except Exception as e:
        print(f"[reader-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            save_dedup_tsv(dedup, worker_tsv)
        except Exception:
            pass
        in_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# GPU INFERENCE WORKER
# ─────────────────────────────────────────────────────────────────────────────

def gpu_worker(in_queue, out_queue, model_path, batch_size, device,
               max_len, num_readers, threshold):
    assert batch_size > 0,          "Batch size must be a positive integer."
    assert 0.0 <= threshold <= 1.0, "Threshold must be between 0.0 and 1.0."

    try:
        model = (AutoModelForSequenceClassification
                 .from_pretrained(model_path)
                 .to(device)
                 .eval())

        # Tokenization happens in the reader processes now (see
        # reader_worker_from_index) — queue items arrive pre-tokenized as
        # numpy arrays, one GPU-batch-sized item at a time. gpu_worker just
        # pins + copies to device + runs the forward pass, so the main thread
        # is never blocked on CPU tokenization and readers can stage several
        # batches ahead in the queue independently.
        def infer_gpu(papers: list, encoded: dict) -> np.ndarray:
            n        = len(papers)
            use_cuda = "cuda" in device
            amp_ctx  = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_cuda else nullcontext())

            inputs = {}
            for k, v in encoded.items():
                t = torch.from_numpy(v)
                if use_cuda:
                    t = t.pin_memory()
                inputs[k] = t.to(device, non_blocking=use_cuda)

            with torch.inference_mode(), amp_ctx:
                try:
                    logits    = model(**inputs).logits.squeeze(-1)
                    probs_cpu = torch.sigmoid(logits).float().cpu().numpy()
                    del inputs, logits

                except torch.cuda.OutOfMemoryError:
                    print(f"[gpu:{device}] OOM at batch of {n}, "
                          f"halving to micro-batches...", flush=True)
                    del inputs
                    torch.cuda.empty_cache()
                    probs_cpu = np.zeros(n, dtype=np.float32)
                    micro = max(n // 4, 1)
                    for ss in range(0, n, micro):
                        se  = min(ss + micro, n)
                        inp = {k: torch.from_numpy(v[ss:se]).to(device)
                               for k, v in encoded.items()}
                        logits = model(**inp).logits.squeeze(-1)
                        probs_cpu[ss:se] = torch.sigmoid(logits).float().cpu().numpy()
                        del inp, logits
                    torch.cuda.empty_cache()

            return probs_cpu

        done_count = 0
        while done_count < num_readers:
            item = in_queue.get()
            if item is None:
                done_count += 1
                continue
            papers, encoded = item
            probs   = infer_gpu(papers, encoded)
            # Send EVERY scored record (not just BERT-positives) so the writer
            # can archive full metadata for every deduped/structurally-valid
            # paper — not only the ones that pass the relevance threshold.
            # Without this, a BERT-rejected paper's metadata is never written
            # anywhere, so once its raw .gz is deleted it's gone for good
            # except by re-downloading from OpenAlex.
            scored = [
                (paper, round(float(prob), 4))
                for paper, prob in zip(papers, probs.tolist())
            ]
            if scored:
                out_queue.put(scored)
            del papers, encoded, probs, scored

        out_queue.put(None)   # signal writer that this GPU is done

    except Exception as e:
        print(f"\n[gpu:{device}] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        out_queue.put(None)   # always send sentinel so writer can exit cleanly
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# WRITER WORKER
# ─────────────────────────────────────────────────────────────────────────────

def writer_worker(out_queue, all_path, pos_path, threshold, pos_val, num_gpu_workers):
    """
    Receives (paper, bert_score) results from all GPU workers via out_queue.
    Writes EVERY scored record to all_path (the full deduped/structurally-
    filtered archive — every paper OpenAlex has that survived the cheap
    filters, regardless of BERT relevance), and additionally to pos_path
    (pass1_bert.jsonl, the BERT-positive subset Stage 2/Nemotron consumes)
    when bert_score >= threshold. all_path is what makes it safe to later
    delete the raw .gz snapshot files and what lets a future run re-score
    with a different BERT threshold/model without re-downloading anything.

    Expects exactly `num_gpu_workers` None sentinels before exiting.
    Has a per-get timeout so it never hangs if a GPU worker is hard-killed.
    """
    assert threshold >= 0.0, "Threshold must be explicitly provided."
    try:
        local_pos     = 0
        unflushed     = 0
        SYNC_INTERVAL = 200
        done_count    = 0

        with open(all_path, "a", encoding="utf-8", buffering=8 * 1024 * 1024) as f_all, \
             open(pos_path, "a", encoding="utf-8", buffering=8 * 1024 * 1024) as f_pos:
            while done_count < num_gpu_workers:
                try:
                    item = out_queue.get(timeout=WRITER_QUEUE_TIMEOUT_S)
                except Exception:
                    # Timeout — GPU worker(s) likely hard-killed
                    print(
                        f"[writer] {WRITER_QUEUE_TIMEOUT_S}s timeout waiting for GPU output "
                        f"({done_count}/{num_gpu_workers} sentinels received). "
                        f"GPU workers may have crashed. Flushing and exiting.",
                        flush=True,
                    )
                    break

                if item is None:
                    done_count += 1
                    continue

                for paper, prob in item:
                    unflushed += 1
                    is_pos = prob >= threshold
                    out = dict(paper)
                    out.update({
                        "bert_score":     prob,
                        "bert_label":     1 if is_pos else 0,
                        "bert_threshold": threshold,
                    })
                    line = orjson.dumps(out).decode("utf-8") + "\n"
                    f_all.write(line)
                    if is_pos:
                        f_pos.write(line)
                        local_pos += 1

                if local_pos > 0:
                    with pos_val.get_lock():
                        pos_val.value += local_pos
                    local_pos = 0

                if unflushed >= SYNC_INTERVAL:
                    f_all.flush(); os.fsync(f_all.fileno())
                    f_pos.flush(); os.fsync(f_pos.fileno())
                    unflushed = 0

            # Final flush regardless of how we exited the loop
            if unflushed > 0:
                f_all.flush(); os.fsync(f_all.fileno())
                f_pos.flush(); os.fsync(f_pos.fileno())

            if local_pos > 0:
                with pos_val.get_lock():
                    pos_val.value += local_pos

    except Exception as e:
        print(f"\n[writer] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SEED-STABLE FILE PARTITIONING
# ─────────────────────────────────────────────────────────────────────────────

def partition_files(gz_files: list, gpu_configs: List[GpuConfig], seed: int) -> Dict:
    """
    Assign .gz files to (gpu_idx, reader_idx) slots deterministically.

    Guarantees:
      - Same seed + same file list = identical assignment on every restart
      - No file assigned to more than one slot (strictly non-overlapping)
      - Changing gpu_configs reshuffles assignments but remains non-overlapping

    Returns dict: (gpu_idx, reader_idx) -> list of gz paths
    """
    import random
    rng   = random.Random(seed)
    files = list(gz_files)
    rng.shuffle(files)

    slots = []
    for gi, gcfg in enumerate(gpu_configs):
        for ri in range(gcfg.num_readers):
            slots.append((gi, ri))

    partitions: Dict = {slot: [] for slot in slots}
    for i, fpath in enumerate(files):
        partitions[slots[i % len(slots)]].append(fpath)

    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# BACKUP UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def do_backup_and_merge(output_dir: str):
    """
    1. Auto-detect all dedup_worker_*.tsv files in output_dir
    2. Merge them into dedup.tsv
    3. Backup dedup.tsv and pass1_bert.jsonl with timestamp suffixes

    Safe to run while the pipeline is live (reads TSVs; does not modify them).
    Run this BEFORE killing a live pipeline to capture full dedup state.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    worker_paths = sorted(glob(os.path.join(output_dir, "dedup_worker_*.tsv")))
    merged_tsv   = os.path.join(output_dir, "dedup.tsv")

    if worker_paths:
        print(f"[backup] Found {len(worker_paths)} worker TSVs, merging → {merged_tsv}")
        n = merge_dedup_tsvs_bucketed(worker_paths, merged_tsv)
        print(f"[backup] Merged {n:,} unique dedup keys.")
    else:
        print("[backup] No worker TSVs found — skipping merge.")

    for fname in ["dedup.tsv", "pass1_bert.jsonl", "all_papers.jsonl"]:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(output_dir, f"{fname}.backup.{ts}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f"[backup] {fname}  →  {os.path.basename(dst)}  ({size_mb:.1f} MB)")
        else:
            print(f"[backup] WARNING: {src} not found — skipping backup.")

    print("[backup] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--snapshot_dir", default=None)
    ap.add_argument("--from_index",   default=None,
                    help="Read pre-deduped/pre-filtered Parquet chunks from this dir "
                         "(built by build_dedup_index.py) instead of raw .gz partitions "
                         "under --snapshot_dir. structural_filter/dedup-by-key are skipped "
                         "here since the index already resolved them globally.")
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--model",        default=None)
    ap.add_argument("--config",       default="bert_classifier.yaml")
    ap.add_argument("--threshold",    type=float, default=None)
    ap.add_argument("--year_start",   type=int,   default=2000)
    ap.add_argument("--year_end",     type=int,   default=2026)
    ap.add_argument("--pattern",      default="**/*.gz")
    ap.add_argument("--seed",         type=int,   default=42,
                    help="RNG seed for file->worker assignment. "
                         "Keep constant across runs for reproducible partitioning.")
    ap.add_argument("--resume",       action="store_true",
                    help="Resume: load dedup.tsv, skip completed files, "
                         "auto-backup pass1_bert.jsonl before appending.")
    ap.add_argument("--delete_after_process", action="store_true",
                    help="After a successful run, delete raw .gz files that "
                         "are marked complete in the manifest. Safe because "
                         "by this point every record they contained has been "
                         "written+deduped into all_papers.jsonl (the full "
                         "archive, independent of BERT threshold) and backed "
                         "up to parquet. Re-fetch anytime with "
                         "backfill_historical_snapshot.py. Only runs at the "
                         "very end of main() — never mid-run, since a file "
                         "marked 'complete' by the reader has only been "
                         "queued for GPU scoring, not necessarily flushed to "
                         "disk yet.")

    # Per-GPU config — repeat once per GPU.
    # Format: "DEVICE_ID:batch=N,chunk=N,queue=N,readers=N"
    ap.add_argument(
        "--gpu_configs", nargs="+",
        default=["0:batch=2048,chunk=16384,queue=4,readers=2"],
        metavar="GPU_SPEC",
        help=(
            "Per-GPU config string(s). Safe to change between runs — "
            "resume state is preserved via basename-keyed manifest. "
            "Example: \"0:batch=1024,chunk=16384,queue=3,readers=2\" "
            "         \"1:batch=2048,chunk=16384,queue=4,readers=3\""
        ),
    )

    # Utility modes (no pipeline run)
    ap.add_argument("--backup-only", action="store_true",
                    help="Merge worker TSVs + backup outputs, then exit. "
                         "Safe to run against a live pipeline.")
    ap.add_argument("--merge-only",  action="store_true",
                    help="Merge worker TSVs into dedup.tsv only, then exit.")

    args = ap.parse_args()

    # ── Utility modes ─────────────────────────────────────────────────────────
    if args.backup_only:
        do_backup_and_merge(args.output_dir)
        return
    if args.merge_only:
        worker_paths = sorted(glob(os.path.join(args.output_dir, "dedup_worker_*.tsv")))
        merged_tsv   = os.path.join(args.output_dir, "dedup.tsv")
        print(f"[merge] Found {len(worker_paths)} worker TSVs")
        n = merge_dedup_tsvs_bucketed(worker_paths, merged_tsv)
        print(f"[merge] Wrote {n:,} keys -> {merged_tsv}")
        return

    # ── Validate pipeline args ────────────────────────────────────────────────
    if not args.snapshot_dir and not args.from_index:
        ap.error("--snapshot_dir or --from_index is required for pipeline runs.")
    if not args.model:
        ap.error("--model is required for pipeline runs.")

    # ── Parse and validate GPU configs ───────────────────────────────────────
    gpu_configs: List[GpuConfig] = [parse_gpu_config(s) for s in args.gpu_configs]
    total_readers = sum(g.num_readers for g in gpu_configs)

    if torch.cuda.is_available():
        n_avail = torch.cuda.device_count()
        print(f"\n[gpu] {n_avail} CUDA device(s) visible:")
        for gc in gpu_configs:
            if gc.device_id >= n_avail:
                raise ValueError(
                    f"Requested cuda:{gc.device_id} but only {n_avail} GPU(s) visible. "
                    f"Check CUDA_VISIBLE_DEVICES."
                )
            free_mem, total_mem = torch.cuda.mem_get_info(gc.device_id)
            print(f"  {gc}  |  free={free_mem/1e9:.1f}/{total_mem/1e9:.1f} GB")
    else:
        print("  WARNING: No CUDA GPUs found — running on CPU (very slow).")
        gpu_configs = [GpuConfig(device_id=-1, batch_size=64, chunk_size=512,
                                 queue_max=2, num_readers=1)]

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = (args.threshold if args.threshold is not None
                 else float(cfg["inference"]["threshold"]))
    max_len   = int(cfg["model"]["max_len"])

    out_jsonl     = os.path.join(args.output_dir, "pass1_bert.jsonl")
    all_jsonl     = os.path.join(args.output_dir, "all_papers.jsonl")
    dedup_tsv     = os.path.join(args.output_dir, "dedup.tsv")
    manifest_path = os.path.join(args.output_dir, COMPLETED_MANIFEST_NAME)

    # Worker dedup TSVs — one per reader, globally indexed
    worker_tsvs = [
        os.path.join(args.output_dir, f"dedup_worker_{i}.tsv")
        for i in range(total_readers)
    ]

    if args.from_index:
        gz_files = sorted(
            glob(os.path.join(args.from_index, "*.parquet")),
            key=os.path.getsize, reverse=True,
        )
        if not gz_files:
            raise FileNotFoundError(f"No .parquet chunks found under {args.from_index}.")
    else:
        gz_files = sorted(
            glob(os.path.join(args.snapshot_dir, args.pattern), recursive=True),
            key=os.path.getsize, reverse=True,
        )
        gz_files = [f for f in gz_files if f.endswith(".gz")]
        if not gz_files:
            raise FileNotFoundError(f"No .gz files found under {args.snapshot_dir}.")

    # ── Resume or clean start ─────────────────────────────────────────────────
    if not args.resume:
        for path in [out_jsonl, all_jsonl, dedup_tsv, manifest_path] + worker_tsvs:
            if os.path.exists(path):
                os.remove(path)
        initial_dedup = {}
        print("[clean start] All prior state cleared.")
    else:
        initial_dedup = load_dedup_tsv(dedup_tsv)
        print(f"[resume] Loaded {len(initial_dedup):,} dedup entries from {dedup_tsv}")

        # pass1_bert.jsonl/all_papers.jsonl are only ever opened "a" (append)
        # by writer_worker(), never rewritten in place, and completed_files.txt
        # + dedup.tsv already guarantee no chunk is ever double-processed. So
        # the only thing a hard kill/crash can corrupt is a torn last line
        # (the in-flight write when it died) — never anything already flushed
        # before it. No need to copy hundreds of GB "just in case": just
        # check/repair a torn tail, an O(1) operation regardless of file size.
        for fname, path in [("pass1_bert.jsonl", out_jsonl), ("all_papers.jsonl", all_jsonl)]:
            if os.path.exists(path):
                removed = repair_torn_tail(path)
                if removed:
                    print(f"[resume] {fname}: repaired a torn trailing line "
                          f"({removed:,} bytes truncated).")
                else:
                    print(f"[resume] {fname}: tail OK, no repair needed.")

        # Report how many files will be skipped at file level
        completed  = load_completed_files(manifest_path)
        total_gz   = len(gz_files)
        skippable  = sum(1 for f in gz_files if gz_manifest_key(f) in completed)
        remaining  = total_gz - skippable
        print(f"[resume] {skippable}/{total_gz} .gz files already completed "
              f"({remaining} remaining).")
        if remaining == 0:
            print("[resume] Nothing to do — all files already processed. Exiting cleanly.")
            sys.exit(0)

    # ── Seed-stable partitioning ──────────────────────────────────────────────
    partitions = partition_files(gz_files, gpu_configs, seed=args.seed)

    # ── Shared counters ───────────────────────────────────────────────────────
    checked_val         = Value(ctypes.c_int64, 0)
    kept_val            = Value(ctypes.c_int64, 0)
    deduped_val         = Value(ctypes.c_int64, 0)
    pos_val             = Value(ctypes.c_int64, 0)
    filtered_type_val   = Value(ctypes.c_int64, 0)
    filtered_status_val = Value(ctypes.c_int64, 0)
    skipped_files_val   = Value(ctypes.c_int64, 0)

    # ── Build queues, GPU workers, readers ───────────────────────────────────
    out_queue        = Queue(maxsize=8)
    in_queues: List  = []
    gpu_procs: List  = []
    readers:   List  = []
    global_reader_id = 0

    for gi, gcfg in enumerate(gpu_configs):
        dev = gcfg.device if torch.cuda.is_available() else "cpu"
        iq  = Queue(maxsize=max(gcfg.queue_max, 1))
        in_queues.append(iq)

        gp = Process(
            target=gpu_worker,
            args=(iq, out_queue, args.model, gcfg.batch_size,
                  dev, max_len, gcfg.num_readers, threshold),
            daemon=True,
        )
        gp.start()
        gpu_procs.append(gp)
        print(f"[main] GPU worker started on {dev} "
              f"(batch={gcfg.batch_size}, readers={gcfg.num_readers})", flush=True)

        for ri in range(gcfg.num_readers):
            wid       = global_reader_id
            file_list = partitions.get((gi, ri), [])
            if args.from_index:
                p = Process(
                    target=reader_worker_from_index,
                    args=(
                        wid, file_list, iq, cfg, gcfg.chunk_size,
                        checked_val, kept_val, deduped_val,
                        worker_tsvs[wid], initial_dedup,
                        manifest_path,
                        skipped_files_val,
                        args.model, max_len, gcfg.batch_size,
                    ),
                    daemon=True,
                )
            else:
                p = Process(
                    target=reader_worker,
                    args=(
                        wid, file_list, iq, cfg, gcfg.chunk_size,
                        args.year_start, args.year_end,
                        checked_val, kept_val, deduped_val,
                        filtered_type_val, filtered_status_val,
                        worker_tsvs[wid], initial_dedup,
                        manifest_path,
                        skipped_files_val,
                    ),
                    daemon=True,
                )
            p.start()
            readers.append(p)
            print(f"[main] reader-{wid} -> {dev}  "
                  f"({len(file_list)} files, chunk={gcfg.chunk_size})", flush=True)
            global_reader_id += 1

    del initial_dedup   # free parent-process memory

    writer_proc = Process(
        target=writer_worker,
        args=(out_queue, all_jsonl, out_jsonl, threshold, pos_val, len(gpu_configs)),
        daemon=True,
    )
    writer_proc.start()

    # ── Signal handlers ───────────────────────────────────────────────────────
    def kill_all():
        try:
            for iq in in_queues:
                while not iq.empty():
                    iq.get_nowait()
            while not out_queue.empty():
                out_queue.get_nowait()
        except Exception:
            pass
        for iq in in_queues:
            iq.cancel_join_thread()
        out_queue.cancel_join_thread()
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive():
                p.terminate()
        time.sleep(1)
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive():
                p.kill()

    def _shutdown(sig, frame):
        print(f"\n[signal {sig}] Shutting down cleanly...", flush=True)
        kill_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # ── Startup summary ───────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Pipeline started")
    print(f"  Total .gz files : {len(gz_files):,}")
    print(f"  Total readers   : {total_readers}")
    print(f"  GPU workers     : {len(gpu_configs)}")
    print(f"  BERT threshold  : {threshold}")
    print(f"  Year range      : {args.year_start}-{args.year_end}")
    print(f"  Allowed types   : {', '.join(sorted(ALLOWED_TYPES))}")
    print(f"  Dedup sync      : every {DEDUP_SYNC_INTERVAL:,} records/worker")
    print(f"  Partition seed  : {args.seed}  (keep constant for reproducible resume)")
    print(f"  Resume mode     : {args.resume}")
    for gi, gc in enumerate(gpu_configs):
        dev = gc.device if torch.cuda.is_available() else "cpu"
        print(f"    GPU {gi}: {dev}  batch={gc.batch_size}  chunk={gc.chunk_size}  "
              f"queue={gc.queue_max}  readers={gc.num_readers}")
    print()

    # ── Monitoring loop ───────────────────────────────────────────────────────
    t0, last_checked = time.time(), 0
    pbar = tqdm(
        desc="Scanning",
        unit="rec",
        dynamic_ncols=True,
        mininterval=2.0,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    while True:
        time.sleep(5)

        checked  = checked_val.value
        kept     = kept_val.value
        deduped  = deduped_val.value
        pos      = pos_val.value
        ftype    = filtered_type_val.value
        fstatus  = filtered_status_val.value
        skipped  = skipped_files_val.value
        elapsed  = max(time.time() - t0, 1)
        rate     = checked / elapsed

        pbar.update(checked - last_checked)
        last_checked = checked
        pass_rate = 100.0 * kept / max(checked, 1)
        iq_sizes  = "+".join(str(iq.qsize()) for iq in in_queues)

        pbar.set_postfix({
            "checked":  f"{checked/1e6:.2f}M",
            "pass%":    f"{pass_rate:.1f}%",
            "Δtype":    f"{ftype/1e3:.1f}k",
            "Δstatus":  f"{fstatus/1e3:.1f}k",
            "deduped":  f"{deduped/1e3:.1f}k",
            "BERT+":    f"{pos:,}",
            "rec/s":    f"{rate:,.0f}",
            "iq":       iq_sizes,
            "skip_f":   skipped,
        }, refresh=True)

        gpu_alive     = any(p.is_alive() for p in gpu_procs)
        writer_alive  = writer_proc.is_alive()
        readers_alive = any(p.is_alive() for p in readers)

        if not readers_alive and not gpu_alive and not writer_alive:
            break
        if not gpu_alive and readers_alive:
            print("\n[FATAL] All GPU workers crashed. Shutting down.", flush=True)
            kill_all()
            sys.exit(1)
        if not writer_alive and (gpu_alive or readers_alive):
            print("\n[FATAL] Writer worker crashed. Shutting down.", flush=True)
            kill_all()
            sys.exit(1)

    pbar.close()
    for p in readers:   p.join(timeout=60)
    for p in gpu_procs: p.join(timeout=60)
    writer_proc.join(timeout=60)

    # ── Final dedup merge ─────────────────────────────────────────────────────
    print(f"\n[merge] Merging worker TSVs -> {dedup_tsv} ...")
    all_worker_tsvs = sorted(glob(os.path.join(args.output_dir, "dedup_worker_*.tsv")))
    n_merged = merge_dedup_tsvs_bucketed(all_worker_tsvs, dedup_tsv)
    print(f"[merge] {n_merged:,} unique keys written.")
    for wp in worker_tsvs:
        if os.path.exists(wp):
            os.remove(wp)

    # ── Dedup pass1_bert.jsonl / all_papers.jsonl by key (keep highest updated_date) ──
    # Needed because a re-touched paper can now legitimately appear more than
    # once across runs — see dedup_pass1_jsonl_by_key() docstring.
    print(f"\n[merge] Deduping {out_jsonl} by key (keep latest per paper) ...")
    p1_total, p1_unique = dedup_pass1_jsonl_by_key(out_jsonl)
    print(f"[merge] {p1_total:,} rows -> {p1_unique:,} unique papers "
          f"({p1_total - p1_unique:,} superseded duplicates removed).")

    print(f"\n[merge] Deduping {all_jsonl} by key (keep latest per paper) ...")
    all_total, all_unique = dedup_pass1_jsonl_by_key(all_jsonl)
    print(f"[merge] {all_total:,} rows -> {all_unique:,} unique papers "
          f"({all_total - all_unique:,} superseded duplicates removed).")

    # ── Parquet backup of the full archive (corruption-safety copy, not the
    #    canonical format — see README for the jsonl-vs-parquet rationale) ────
    all_parquet = all_jsonl[:-len(".jsonl")] + ".parquet" if all_jsonl.endswith(".jsonl") else all_jsonl + ".parquet"
    try:
        convert_script = os.path.join(os.path.dirname(__file__), "..", "scripts", "convert_dataset.py")
        subprocess.run(
            [sys.executable, convert_script, all_jsonl, all_parquet],
            check=True, timeout=1800,
        )
        print(f"[backup] Wrote parquet backup -> {all_parquet}")
    except Exception as e:
        print(f"[backup] WARNING: parquet backup failed (non-fatal): {e}")

    # ── Optional: delete raw .gz files now that every record they contained
    #    is safely archived in all_papers.jsonl (+ parquet backup). Only safe
    #    to do here, at the very end of a successful run — see the
    #    --delete_after_process help text for why doing this per-file mid-run
    #    would risk data loss on a crash.
    if args.delete_after_process and args.from_index:
        print("[cleanup] --delete_after_process is not supported with --from_index "
              "(manifest keys reference index chunks, not raw .gz files) — skipping.",
              flush=True)
    elif args.delete_after_process:
        completed = load_completed_files(manifest_path)
        n_deleted, freed_bytes = 0, 0
        for key in completed:
            gz_path = os.path.join(args.snapshot_dir, key)
            if os.path.exists(gz_path):
                freed_bytes += os.path.getsize(gz_path)
                os.remove(gz_path)
                n_deleted += 1
        # Clean up now-empty updated_date=... directories
        for d in glob(os.path.join(args.snapshot_dir, "updated_date=*")):
            try:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass
        print(f"[cleanup] Deleted {n_deleted:,} processed .gz files "
              f"({freed_bytes/1e9:.1f} GB freed). Re-fetch anytime with "
              f"backfill_historical_snapshot.py if needed.")

    # ── Final summary ─────────────────────────────────────────────────────────
    dt      = time.time() - t0
    checked = checked_val.value
    kept    = kept_val.value
    pos     = pos_val.value
    ftype   = filtered_type_val.value
    fstatus = filtered_status_val.value
    deduped = deduped_val.value
    skipped = skipped_files_val.value
    other   = max(checked - kept - ftype - fstatus - deduped, 0)

    print("=" * 65)
    print(f"[{datetime.now():%H:%M:%S}] PIPELINE COMPLETE")
    print(f"  Files skipped (resume)    : {skipped:>12,}")
    print(f"  Records checked           : {checked:>12,}")
    print(f"  |-  year/lang/retracted   : {other:>12,}")
    print(f"  |-  filtered by type      : {ftype:>12,}")
    print(f"  |-  filtered by status    : {fstatus:>12,}")
    print(f"  |-  deduplicated          : {deduped:>12,}")
    print(f"  +-  sent to BERT          : {kept:>12,}  ({100*kept/max(checked,1):.2f}%)")
    print(f"  BERT positives            : {pos:>12,}  (threshold={threshold})")
    print(f"  Time                      : {dt/60:>11.1f} min")
    print(f"  Throughput                : {checked/max(dt,1):>10,.0f} rec/s")
    print("=" * 65)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
