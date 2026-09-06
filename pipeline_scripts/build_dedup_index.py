#!/usr/bin/env python3
"""
build_dedup_index.py
=====================
Phase A of BERT Stage 1: a GPU-free pre-pass over the full OpenAlex snapshot
that resolves "latest version per paper" globally, BEFORE any BERT scoring
happens.

Why this exists: filter_bert.py processes .gz partitions in a fixed-seed
random shuffle, not chronological order, so its inline dedup.tsv check can
only catch a record if an equal-or-newer version was *already* seen earlier
in the *same* run — it can't know a newer version is coming later, so a
stale record can still get fully BERT-scored (real GPU time spent) before
dedup_pass1_jsonl_by_key() cleans it up after the fact in the output file.
Since GPU inference (not I/O/decompression) is the actual bottleneck,
resolving "latest version per key" globally before any GPU work starts
eliminates that wasted GPU time, and the resulting index is reusable for
any future full-corpus rerun.

Two full passes over the snapshot, CPU-only (no torch model loaded, no GPU):

  Pass 1 (winner selection): stream every record in every .gz file, apply
    the same structural_filter() filter_bert.py uses, track the best
    (year, updated_date) per key (doi or oa_id) seen ANYWHERE in the
    corpus. Output: dedup_index/global_winners.tsv.

  Pass 2 (candidate emission): stream every record again; keep only
    records that ARE the global winner for their key AND are not already
    resolved in the existing dedup.tsv (i.e. not already BERT-scored in a
    prior/just-stopped run — this is what makes it safe to run without
    losing any prior progress). Survivors are written as rolling chunked
    Parquet files: dedup_index/workerNN_partNNNN.parquet, each row storing
    key/updated_date/payload (payload = orjson-serialized slim() record,
    identical to what filter_bert.py would have built for BERT scoring).

filter_bert.py --from_index dedup_index then reads these chunk files
instead of raw .gz partitions for Stage 1 GPU scoring.

RUN:
    python -u build_dedup_index.py \\
        --snapshot_dir /path/to/openalex \\
        --output_dir   /path/to/data/processed \\
        --year_start 1950 --year_end 2026 \\
        --workers 16
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
import traceback
import subprocess
import multiprocessing as mp
from multiprocessing import Process, Value
from glob import glob
from datetime import datetime

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_bert import (
    structural_filter,
    slim,
    load_dedup_tsv,
    save_dedup_tsv,
    merge_dedup_tsvs_bucketed,
)

COUNTER_FLUSH_INTERVAL = 1_000
ROWS_PER_FILE_DEFAULT  = 500_000
PARQUET_SCHEMA = pa.schema([
    ("key",          pa.string()),
    ("updated_date", pa.string()),
    ("payload",      pa.string()),
])


def split_files(gz_files: list, n_workers: int) -> list:
    """Balanced round-robin split over size-sorted files (no shuffle needed —
    both passes read the entire corpus regardless of order)."""
    files = sorted(gz_files, key=os.path.getsize, reverse=True)
    buckets = [[] for _ in range(n_workers)]
    for i, f in enumerate(files):
        buckets[i % n_workers].append(f)
    return buckets


def _iter_records(gz_path: str):
    with subprocess.Popen(
        ["pigz", "-dc", gz_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as proc:
        for raw_line in proc.stdout:
            if not raw_line:
                continue
            try:
                yield orjson.loads(raw_line)
            except Exception:
                continue


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — winner selection
# ─────────────────────────────────────────────────────────────────────────────

def winner_worker(worker_id, file_list, year_start, year_end, checked_val, out_tsv):
    local_dedup = {}
    local_checked = 0
    try:
        for gz_path in file_list:
            for rec in _iter_records(gz_path):
                local_checked += 1
                passes, _ = structural_filter(rec, year_start, year_end)
                if not passes:
                    if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                        with checked_val.get_lock():
                            checked_val.value += COUNTER_FLUSH_INTERVAL
                        local_checked = 0
                    continue

                oaid  = (rec.get("id")  or "").strip()
                doi   = (rec.get("doi") or "").strip().lower()
                key   = doi if doi else oaid
                if key:
                    year  = rec.get("publication_year") or 0
                    udate = (rec.get("updated_date") or "").strip()
                    prev = local_dedup.get(key)
                    if prev is None or udate >= prev[1]:
                        local_dedup[key] = (year, udate)

                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                    with checked_val.get_lock():
                        checked_val.value += COUNTER_FLUSH_INTERVAL
                    local_checked = 0
    except Exception as e:
        print(f"[winner-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        with checked_val.get_lock():
            checked_val.value += local_checked
        save_dedup_tsv(local_dedup, out_tsv)


def run_pass1(gz_files, year_start, year_end, n_workers, index_dir):
    buckets = split_files(gz_files, n_workers)
    checked_val = Value("q", 0)
    worker_tsvs = [os.path.join(index_dir, f"pass1_worker_{i}.tsv") for i in range(n_workers)]

    procs = []
    for i in range(n_workers):
        p = Process(target=winner_worker,
                     args=(i, buckets[i], year_start, year_end, checked_val, worker_tsvs[i]))
        p.start()
        procs.append(p)

    pbar = tqdm(desc="[Pass 1] winner selection", unit="rec", dynamic_ncols=True, mininterval=2.0)
    last = 0
    while any(p.is_alive() for p in procs):
        time.sleep(3)
        cur = checked_val.value
        pbar.update(cur - last)
        last = cur
    pbar.update(checked_val.value - last)
    pbar.close()
    for p in procs:
        p.join()

    global_winners_tsv = os.path.join(index_dir, "global_winners.tsv")
    n = merge_dedup_tsvs_bucketed(worker_tsvs, global_winners_tsv)
    for wp in worker_tsvs:
        if os.path.exists(wp):
            os.remove(wp)
    print(f"[Pass 1] {n:,} unique keys -> {global_winners_tsv}", flush=True)
    return global_winners_tsv


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — candidate emission
# ─────────────────────────────────────────────────────────────────────────────

class RollingParquetWriter:
    def __init__(self, out_dir, worker_id, rows_per_file):
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.rows_per_file = rows_per_file
        self.part = 0
        self.buf_key, self.buf_udate, self.buf_payload = [], [], []
        self.rows_in_file = 0
        self._writer = None

    def add(self, key, udate, payload):
        self.buf_key.append(key)
        self.buf_udate.append(udate)
        self.buf_payload.append(payload)
        self.rows_in_file += 1
        if len(self.buf_key) >= 16384:
            self._flush_batch()
        if self.rows_in_file >= self.rows_per_file:
            self._roll()

    def _flush_batch(self):
        if not self.buf_key:
            return
        table = pa.table(
            {"key": self.buf_key, "updated_date": self.buf_udate, "payload": self.buf_payload},
            schema=PARQUET_SCHEMA,
        )
        if self._writer is None:
            path = os.path.join(self.out_dir, f"worker{self.worker_id:02d}_part{self.part:04d}.parquet")
            self._writer = pq.ParquetWriter(path, PARQUET_SCHEMA)
        self._writer.write_table(table)
        self.buf_key, self.buf_udate, self.buf_payload = [], [], []

    def _roll(self):
        self._flush_batch()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.part += 1
        self.rows_in_file = 0

    def close(self):
        self._flush_batch()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def emit_worker(worker_id, file_list, year_start, year_end, still_needed,
                 out_dir, rows_per_file, checked_val, kept_val):
    local_checked = 0
    local_kept = 0
    writer = RollingParquetWriter(out_dir, worker_id, rows_per_file)
    try:
        for gz_path in file_list:
            for rec in _iter_records(gz_path):
                local_checked += 1
                passes, _ = structural_filter(rec, year_start, year_end)
                if passes:
                    oaid  = (rec.get("id")  or "").strip()
                    doi   = (rec.get("doi") or "").strip().lower()
                    key   = doi if doi else oaid
                    udate = (rec.get("updated_date") or "").strip()
                    want  = still_needed.get(key)
                    if key and want is not None and udate == want[1]:
                        payload = orjson.dumps(slim(rec)).decode("utf-8")
                        writer.add(key, udate, payload)
                        local_kept += 1

                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                    with checked_val.get_lock():
                        checked_val.value += COUNTER_FLUSH_INTERVAL
                    local_checked = 0
                    if local_kept:
                        with kept_val.get_lock():
                            kept_val.value += local_kept
                        local_kept = 0
    except Exception as e:
        print(f"[emit-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        writer.close()
        with checked_val.get_lock():
            checked_val.value += local_checked
        with kept_val.get_lock():
            kept_val.value += local_kept


def run_pass2(gz_files, year_start, year_end, n_workers, still_needed, index_dir, rows_per_file):
    buckets = split_files(gz_files, n_workers)
    checked_val = Value("q", 0)
    kept_val    = Value("q", 0)

    procs = []
    for i in range(n_workers):
        p = Process(target=emit_worker,
                     args=(i, buckets[i], year_start, year_end, still_needed,
                           index_dir, rows_per_file, checked_val, kept_val))
        p.start()
        procs.append(p)

    pbar = tqdm(desc="[Pass 2] candidate emission", unit="rec", dynamic_ncols=True, mininterval=2.0)
    last = 0
    while any(p.is_alive() for p in procs):
        time.sleep(3)
        cur = checked_val.value
        pbar.update(cur - last)
        last = cur
        pbar.set_postfix({"kept": f"{kept_val.value:,}"}, refresh=True)
    pbar.update(checked_val.value - last)
    pbar.close()
    for p in procs:
        p.join()

    print(f"[Pass 2] {kept_val.value:,} candidate records written to {index_dir}", flush=True)
    return kept_val.value


# ─────────────────────────────────────────────────────────────────────────────
# FUSED SINGLE-PASS (Pass 1 + Pass 2 combined via hash bucketing)
#
# Pass 1/Pass 2 above each do a full raw-.gz decompress+parse pass over the
# corpus — decompression/parsing is the expensive part, so doing it twice
# roughly doubles wall time for no algorithmic reason. This fuses them into
# ONE raw-.gz read: every structurally-passing record is routed by
# hash(key) into one of `num_buckets` shard files (append-only, so memory
# per worker is just I/O buffers, not a growing dict — same principle as
# merge_dedup_tsvs_bucketed's split stage). A second stage then resolves
# winners bucket-by-bucket (parallel across buckets, memory-bounded to one
# bucket's share of the keyspace at a time) and writes survivors straight to
# the candidate Parquet chunks filter_bert.py --from_index reads — no
# separate re-read of the raw corpus needed for "Pass 2".
# ─────────────────────────────────────────────────────────────────────────────

FUSED_BUCKETS_DEFAULT = 256


def _bucket_of(key, num_buckets):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % num_buckets


def _fused_split_worker(worker_id, file_list, year_start, year_end, shard_dir, num_buckets, checked_val):
    local_checked = 0
    handles = [
        open(os.path.join(shard_dir, f"b{b}_w{worker_id}.jsonl"), "wb", buffering=4 * 1024 * 1024)
        for b in range(num_buckets)
    ]
    try:
        for gz_path in file_list:
            for rec in _iter_records(gz_path):
                local_checked += 1
                passes, _ = structural_filter(rec, year_start, year_end)
                if passes:
                    oaid = (rec.get("id") or "").strip()
                    doi  = (rec.get("doi") or "").strip().lower()
                    key  = doi if doi else oaid
                    if key:
                        udate = (rec.get("updated_date") or "").strip()
                        b = _bucket_of(key, num_buckets)
                        handles[b].write(orjson.dumps({"k": key, "u": udate, "p": slim(rec)}))
                        handles[b].write(b"\n")

                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                    with checked_val.get_lock():
                        checked_val.value += COUNTER_FLUSH_INTERVAL
                    local_checked = 0
    except Exception as e:
        print(f"[fused-split-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        for h in handles:
            h.close()
        with checked_val.get_lock():
            checked_val.value += local_checked


def run_fused_split(gz_files, year_start, year_end, n_workers, shard_dir, num_buckets):
    buckets = split_files(gz_files, n_workers)
    checked_val = Value("q", 0)

    procs = []
    for i in range(n_workers):
        p = Process(target=_fused_split_worker,
                     args=(i, buckets[i], year_start, year_end, shard_dir, num_buckets, checked_val))
        p.start()
        procs.append(p)

    pbar = tqdm(desc="[fused] shard split", unit="rec", dynamic_ncols=True, mininterval=2.0)
    last = 0
    while any(p.is_alive() for p in procs):
        time.sleep(3)
        cur = checked_val.value
        pbar.update(cur - last)
        last = cur
    pbar.update(checked_val.value - last)
    pbar.close()
    for p in procs:
        p.join()


def _fused_resolve_worker(args):
    bucket_id, shard_dir, n_shard_workers, out_dir, rows_per_file, existing = args
    best = {}  # key -> (udate, payload_dict)
    for w in range(n_shard_workers):
        shard_path = os.path.join(shard_dir, f"b{bucket_id}_w{w}.jsonl")
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = orjson.loads(line)
                except Exception:
                    continue
                key, udate, payload = row["k"], row["u"], row["p"]
                prev = best.get(key)
                if prev is None or udate >= prev[0]:
                    best[key] = (udate, payload)
        os.remove(shard_path)

    writer = RollingParquetWriter(out_dir, bucket_id, rows_per_file)
    kept = 0
    try:
        for key, (udate, payload) in best.items():
            if udate > existing.get(key, (-1, ""))[1]:
                writer.add(key, udate, orjson.dumps(payload).decode("utf-8"))
                kept += 1
    finally:
        writer.close()
    return len(best), kept


def _partition_existing_dedup(existing_dedup, num_buckets):
    """Split the {key: (year, updated_date)} dict from dedup.tsv into
    per-bucket slices, using the same key->bucket hash as the shard split, so
    each resolve worker gets ONLY the ~1/num_buckets share it can ever look
    up. Passing these as plain (pickled) task args — rather than sharing the
    full multi-hundred-million-object dict as a fork-COW global — is
    deliberate: every dict.get() on a shared dict bumps the refcount of the
    Python objects it touches, and refcounts live in the object header, so
    "read-only" lookups still dirty the page and defeat copy-on-write. At
    118M+ keys that turned near-total sharing into near-total per-worker
    duplication (~24-30GB RSS x 16 workers), causing swap thrash. Small
    private per-bucket dicts have no such sharing to defeat.
    """
    parts = [{} for _ in range(num_buckets)]
    for key, val in existing_dedup.items():
        parts[_bucket_of(key, num_buckets)][key] = val
    return parts


def run_fused_resolve(shard_dir, n_shard_workers, existing_dedup, out_dir, rows_per_file,
                       num_buckets, pool_workers):
    existing_parts = _partition_existing_dedup(existing_dedup, num_buckets)
    del existing_dedup

    resolve_args = [(b, shard_dir, n_shard_workers, out_dir, rows_per_file, existing_parts[b])
                     for b in range(num_buckets)]
    del existing_parts
    total_unique = 0
    total_kept = 0
    with mp.Pool(min(pool_workers, num_buckets)) as pool:
        for n_unique, n_kept in tqdm(
            pool.imap_unordered(_fused_resolve_worker, resolve_args),
            total=num_buckets, desc="[fused] bucket resolve",
        ):
            total_unique += n_unique
            total_kept += n_kept

    print(f"[fused] {total_unique:,} unique keys resolved corpus-wide, "
          f"{total_kept:,} candidates written to {out_dir}", flush=True)
    return total_unique, total_kept


def run_fused(gz_files, year_start, year_end, n_workers, index_dir, existing_dedup,
              rows_per_file, num_buckets):
    shard_dir = os.path.join(index_dir, "fused_shards")
    os.makedirs(shard_dir, exist_ok=True)
    try:
        run_fused_split(gz_files, year_start, year_end, n_workers, shard_dir, num_buckets)
        return run_fused_resolve(shard_dir, n_workers, existing_dedup, index_dir,
                                  rows_per_file, num_buckets, n_workers)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--snapshot_dir", required=True)
    ap.add_argument("--output_dir",   required=True,
                     help="Same dir filter_bert.py uses (reads dedup.tsv from here); "
                          "the index is written to <output_dir>/dedup_index/")
    ap.add_argument("--year_start",   type=int, default=2000)
    ap.add_argument("--year_end",     type=int, default=2026)
    ap.add_argument("--pattern",      default="**/*.gz")
    ap.add_argument("--workers",      type=int, default=16)
    ap.add_argument("--rows_per_file", type=int, default=ROWS_PER_FILE_DEFAULT)
    ap.add_argument("--fused_buckets", type=int, default=FUSED_BUCKETS_DEFAULT,
                     help="Hash buckets for the fused single-pass split/resolve "
                          "(memory per bucket during resolve scales as 1/this).")
    args = ap.parse_args()

    index_dir = os.path.join(args.output_dir, "dedup_index")
    os.makedirs(index_dir, exist_ok=True)

    gz_files = [f for f in glob(os.path.join(args.snapshot_dir, args.pattern), recursive=True)
                if f.endswith(".gz")]
    if not gz_files:
        raise FileNotFoundError(f"No .gz files found under {args.snapshot_dir}")
    print(f"[build_dedup_index] {len(gz_files):,} .gz files, {args.workers} workers, "
          f"year range {args.year_start}-{args.year_end}", flush=True)

    t0 = time.time()

    # ── Two-pass: Pass 1 tracks only {key: (year, updated_date)} in memory
    # (no payload, no per-record disk writes) to find the global winner per
    # key; Pass 2 rereads the corpus once more and writes the full slim()
    # payload ONLY for confirmed survivors. The fused single-pass variant
    # (still below, dormant) writes the full payload for every structurally-
    # passing record into hash-bucket shard files before dedup narrows it
    # down — at full-corpus scale that wrote ~700GB of mostly-discarded
    # payload data and made shard-split the bottleneck (~10-15K rec/s,
    # far below pure decompress+filter throughput). Two full raw-.gz passes
    # here cost CPU time but bound disk writes to the actual candidate count.
    dedup_tsv = os.path.join(args.output_dir, "dedup.tsv")
    existing  = load_dedup_tsv(dedup_tsv)
    print(f"[Pass 1] {len(existing):,} keys already resolved in dedup.tsv", flush=True)

    global_winners_tsv = run_pass1(gz_files, args.year_start, args.year_end, args.workers, index_dir)
    winners = load_dedup_tsv(global_winners_tsv)
    total_unique = len(winners)
    still_needed = {k: v for k, v in winners.items() if v[1] > existing.get(k, (-1, ""))[1]}
    print(f"[Pass 1] {total_unique:,} global winners, {len(still_needed):,} still needed", flush=True)
    del winners, existing

    if not still_needed:
        print("[build_dedup_index] Nothing new to score — index is empty. Done.", flush=True)
        return

    # `still_needed` is inherited by every fork()'d Pass-2 worker as a plain
    # Python dict. This is the SAME fork-COW-vs-refcounting trap already hit
    # (and fixed) in the fused resolve step: emit_worker() calls
    # still_needed.get(key) for every one of the ~500M raw records, and each
    # hit increfs the returned tuple, dirtying its page and defeating COW —
    # at 118M entries that caused a 6+ hour stall from ~16x memory
    # duplication. Empirically that worked out to ~215-260 bytes/key of
    # per-worker duplicated RAM. Cap Pass 2's worker count so
    # n_workers * len(still_needed) * 260 bytes stays under a safe budget,
    # rather than trusting COW sharing to hold at whatever size this run's
    # delta happens to be.
    MEM_BUDGET_BYTES = 120 * 1024**3  # leave headroom under 251GB total RAM
    BYTES_PER_KEY    = 260
    safe_workers = max(1, MEM_BUDGET_BYTES // (len(still_needed) * BYTES_PER_KEY + 1))
    pass2_workers = max(1, min(args.workers, safe_workers))
    if pass2_workers < args.workers:
        print(f"[Pass 2] capping workers {args.workers} -> {pass2_workers} "
              f"({len(still_needed):,} still-needed keys x ~{BYTES_PER_KEY}B x {args.workers} "
              f"workers would risk exceeding the {MEM_BUDGET_BYTES/1024**3:.0f}GB memory budget)",
              flush=True)

    total_kept = run_pass2(gz_files, args.year_start, args.year_end, pass2_workers,
                            still_needed, index_dir, args.rows_per_file)

    dt = time.time() - t0
    print("=" * 65)
    print(f"[build_dedup_index] DONE in {dt/60:.1f} min")
    print(f"  Unique keys corpus-wide : {total_unique:,}")
    print(f"  Candidates written      : {total_kept:,}")
    print(f"  Index dir               : {index_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
