#!/usr/bin/env python3
"""
download_snapshot.py
====================
Parallel OpenAlex works snapshot downloader with:
  - Multi-threaded concurrent downloads (default: 8 threads)
  - Per-file tqdm progress bars
  - Overall progress bar
  - Skip already-completed files (resume-safe)
  - Integrity check via file size comparison with S3

Usage:
    pip install boto3 tqdm

    python download_snapshot.py \
        --output_dir ./openalex-works \
        --workers 8

    # More aggressive (fast network / high bandwidth):
    python download_snapshot.py \
        --output_dir ./openalex-works \
        --workers 16
"""

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install boto3 tqdm")
    sys.exit(1)

BUCKET   = "openalex"
# OpenAlex moved works snapshot data from "data/works/" to "data/jsonl/works/"
# (adding a parquet-format sibling at "data/parquet/works/") since the last run.
PREFIX   = "data/jsonl/works/"
REGION   = "us-east-1"

# ── thread-safe print lock ────────────────────────────────────────────────────
_print_lock = threading.Lock()

def log(msg):
    with _print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_s3():
    """Create a no-auth S3 client (public bucket)."""
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=50,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def list_all_gz_files(s3):
    """List every .gz file under s3://openalex/<PREFIX> using pagination."""
    log(f"Listing all .gz files in s3://openalex/{PREFIX} ...")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".gz"):
                files.append({"key": key, "size": obj["Size"]})
    log(f"Found {len(files):,} .gz files  "
        f"(~{sum(f['size'] for f in files)/1e9:.1f} GB compressed)")
    return files


def local_path(key, output_dir):
    """Map S3 key → local file path, preserving folder structure."""
    rel = key[len(PREFIX):]          # strip PREFIX, e.g. "data/jsonl/works/"
    return os.path.join(output_dir, rel)


def is_complete(local, expected_size):
    """Return True if file exists and matches expected S3 size."""
    if not os.path.exists(local):
        return False
    return os.path.getsize(local) == expected_size


def download_one(s3, key, expected_size, local, overall_bar):
    """
    Download a single .gz file with a per-file tqdm bar.
    Returns (key, bytes_downloaded, skipped: bool, error: str|None)
    """
    if is_complete(local, expected_size):
        overall_bar.update(1)
        overall_bar.set_postfix_str(f"skip {os.path.basename(local)}", refresh=False)
        return key, 0, True, None

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"

    # Resume partial download
    resume_byte = 0
    if os.path.exists(tmp):
        resume_byte = os.path.getsize(tmp)

    try:
        range_header = f"bytes={resume_byte}-" if resume_byte > 0 else None
        kwargs = {"Bucket": BUCKET, "Key": key}
        if range_header:
            kwargs["Range"] = range_header

        resp      = s3.get_object(**kwargs)
        body      = resp["Body"]
        remaining = expected_size - resume_byte

        desc = os.path.basename(local)[:28].ljust(28)

        with tqdm(
            total=expected_size,
            initial=resume_byte,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
            position=None,     # tqdm auto-assigns position
            miniters=1,
        ) as pbar:
            mode = "ab" if resume_byte > 0 else "wb"
            with open(tmp, mode) as f:
                downloaded = resume_byte
                for chunk in body.iter_chunks(chunk_size=1024 * 1024):  # 1 MB chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
                    downloaded += len(chunk)

        # Atomic rename only if complete
        if os.path.getsize(tmp) == expected_size:
            os.replace(tmp, local)
            overall_bar.update(1)
            overall_bar.set_postfix_str(f"done {os.path.basename(local)}", refresh=False)
            return key, expected_size - resume_byte, False, None
        else:
            return key, 0, False, f"Size mismatch: got {os.path.getsize(tmp)} expected {expected_size}"

    except Exception as e:
        return key, 0, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./openalex-works",
                        help="Local directory to save files")
    parser.add_argument("--workers",    type=int, default=8,
                        help="Concurrent download threads (default: 8)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Only download first N files (for testing)")
    parser.add_argument("--test",       action="store_true",
                        help="Download only 3 files (quick smoke-test)")
    parser.add_argument("--after_date", default=None, metavar="YYYY-MM-DD",
                        help=(
                            "Skip partitions with updated_date < this date. OpenAlex's "
                            "works snapshot is a cumulative changelog going back to 2016 "
                            "(a work last touched in 2018 exists ONLY in a 2018-dated "
                            "partition), so this does NOT give you 'only recent papers' — "
                            "it gives you 'only partitions not already covered by a prior "
                            "full download+process run'. Only use this when you already "
                            "have pass1_bert.jsonl + dedup.tsv from a prior run covering "
                            "everything before this date; otherwise you will silently miss "
                            "older works that were never downloaded/processed at all."
                        ))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    s3 = make_s3()

    # ── List all files ────────────────────────────────────────────────────────
    all_files = list_all_gz_files(s3)
    if args.after_date:
        import re
        before = len(all_files)
        all_files = [
            f for f in all_files
            if (m := re.search(r"updated_date=(\d{4}-\d{2}-\d{2})", f["key"]))
            and m.group(1) >= args.after_date
        ]
        log(f"--after_date {args.after_date}: {before:,} -> {len(all_files):,} files "
            f"({sum(f['size'] for f in all_files)/1e9:.1f} GB)")
    limit = 3 if args.test else args.limit
    if limit:
        all_files = all_files[:limit]
        log(f"TEST MODE — limited to first {limit} files")

    # ── Classify: done vs pending ─────────────────────────────────────────────
    done_files    = []
    pending_files = []
    for f in all_files:
        lp = local_path(f["key"], args.output_dir)
        if is_complete(lp, f["size"]):
            done_files.append(f)
        else:
            pending_files.append(f)

    total_pending_gb = sum(f["size"] for f in pending_files) / 1e9
    log(f"Already complete : {len(done_files):>6,} files")
    log(f"To download      : {len(pending_files):>6,} files "
        f"({total_pending_gb:.1f} GB compressed)")
    log(f"Workers          : {args.workers}")

    if not pending_files:
        log("Nothing to download — all files already complete!")
        return

    # ── Download with thread pool ─────────────────────────────────────────────
    stats = {"downloaded_bytes": 0, "errors": 0, "skipped": 0, "done": 0}
    t_start = time.time()

    # Each thread gets its own S3 client to avoid connection contention
    s3_pool = [make_s3() for _ in range(args.workers)]

    with tqdm(total=len(all_files), initial=len(done_files),
              desc="Overall", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as overall_bar:

        def task(item):
            idx   = pending_files.index(item) % args.workers
            s3_c  = s3_pool[idx]
            lp    = local_path(item["key"], args.output_dir)
            return download_one(s3_c, item["key"], item["size"], lp, overall_bar)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(task, f): f for f in pending_files}

            for future in as_completed(futures):
                key, nbytes, skipped, error = future.result()
                fname = os.path.basename(key)

                if error:
                    log(f"  ERROR {fname}: {error}")
                    stats["errors"] += 1
                elif skipped:
                    stats["skipped"] += 1
                else:
                    stats["downloaded_bytes"] += nbytes
                    stats["done"] += 1

    elapsed = time.time() - t_start
    dl_gb   = stats["downloaded_bytes"] / 1e9
    speed   = dl_gb / max(elapsed / 3600, 1e-6)

    log(f"\n{'='*60}")
    log(f"DOWNLOAD COMPLETE")
    log(f"  Files downloaded : {stats['done']:>8,}")
    log(f"  Files skipped    : {stats['skipped']:>8,}")
    log(f"  Errors           : {stats['errors']:>8,}")
    log(f"  Data transferred : {dl_gb:>8.2f} GB")
    log(f"  Time elapsed     : {elapsed/3600:>8.2f} h")
    log(f"  Avg speed        : {speed:>8.2f} GB/h")
    log(f"  Output dir       : {args.output_dir}")

    # Final disk usage
    result = os.popen(f"du -sh {args.output_dir}").read().strip()
    log(f"  Disk usage       : {result.split()[0] if result else 'unknown'}")
    log(f"{'='*60}")

    if stats["errors"] > 0:
        log(f"Re-run to retry {stats['errors']} failed files — already-complete files are skipped.")


if __name__ == "__main__":
    main()
