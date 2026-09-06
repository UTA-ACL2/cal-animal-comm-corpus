#!/usr/bin/env python3
"""
pipeline.py — OpenAlex -> relevant-paper dataset pipeline (reproducibility copy)

This is the data-acquisition/filtering/annotation pipeline used to build the
corpus reported in the survey paper. It is a trimmed copy of the original
internal pipeline: a 4th stage that builds a search index for a downstream
application is not included here since it isn't part of dataset construction.

Stages (run in order):
  1. download  — pull OpenAlex snapshot partitions from S3 (public bucket)
  2. bert      — dedup + SciBERT relevance classification on downloaded partitions
  3. nemotron  — Nemotron metadata extraction + structured relevance verification

Each stage is resumable: already-downloaded files, already-deduped/classified
records, and already-annotated papers are all skipped automatically.

Usage:
  python pipeline.py                      # run all 3 stages once
  python pipeline.py --stage bert         # run one stage only (e.g. for recovery)

Requires a pipeline_config.yaml — see pipeline_config.example.yaml for the
paths/settings the original run used (fill in your own local paths and model
locations before running).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "pipeline_state.json"
LOG_FILE = ROOT / "pipeline.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed (exit {result.returncode}): {' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_download(cfg: dict, test: bool = False) -> None:
    log("=== Stage 1: download ===")
    paths = cfg["paths"]
    dl_cfg = cfg.get("download", {})

    snapshot_dir = ROOT / paths["snapshot_dir"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "pipeline_scripts" / "download_snapshot.py"),
        "--output_dir", str(snapshot_dir),
        "--workers",    str(dl_cfg.get("workers", 8)),
    ]
    if dl_cfg.get("after_date"):
        cmd += ["--after_date", str(dl_cfg["after_date"])]
    if test:
        cmd.append("--test")
    run(cmd, cwd=ROOT / "pipeline_scripts")


def stage_dedup_index(cfg: dict) -> None:
    log("=== Stage 2a: build dedup index (GPU-free pre-pass) ===")
    paths    = cfg["paths"]
    bert_cfg = cfg.get("bert", {})

    snapshot_dir = ROOT / paths["snapshot_dir"]
    output_dir   = ROOT / paths["bert_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    year_end = str(bert_cfg.get("year_end", datetime.now().year))

    run(
        [
            sys.executable, "-u",
            str(ROOT / "pipeline_scripts" / "build_dedup_index.py"),
            "--snapshot_dir", str(snapshot_dir),
            "--output_dir",   str(output_dir),
            "--year_start",   str(bert_cfg.get("year_start", 2000)),
            "--year_end",     year_end,
            "--workers",      str(bert_cfg.get("dedup_index_workers", 16)),
        ],
        cwd=ROOT / "pipeline_scripts",
    )


def stage_bert(cfg: dict) -> None:
    # Always rebuild the dedup index first — this is what lets a rerun after a
    # fresh `download` (new OpenAlex partitions) skip everything already
    # BERT-scored, without re-scanning/re-scoring the whole snapshot. See
    # stage_dedup_index()'s docstring / build_dedup_index.py for why this has
    # to run as a separate GPU-free pass before BERT, not merged into it.
    stage_dedup_index(cfg)

    log("=== Stage 2: BERT filter ===")
    paths   = cfg["paths"]
    bert_cfg = cfg.get("bert", {})

    output_dir   = ROOT / paths["bert_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model  = ROOT / paths["bert_model"]
    config = ROOT / paths["bert_config"]
    index_dir = output_dir / "dedup_index"

    run(
        [
            sys.executable, "-u",
            str(ROOT / "pipeline_scripts" / "filter_bert.py"),
            "--from_index",   str(index_dir),
            "--output_dir",   str(output_dir),
            "--model",        str(model),
            "--config",       str(config),
            "--gpu_configs",  bert_cfg.get("gpu_configs", "0:batch=2048,chunk=16384,queue=4,readers=2"),
            "--threshold",    str(bert_cfg.get("threshold", 0.45)),
            "--seed",         str(bert_cfg.get("seed", 42)),
            "--resume",
        ],
        cwd=ROOT / "pipeline_scripts",
    )


def stage_nemotron(cfg: dict) -> None:
    log("=== Stage 3: Nemotron annotation ===")
    paths        = cfg["paths"]
    nemotron_cfg = cfg.get("nemotron", {})

    # pass1_bert.jsonl (BERT's accumulated Stage-1 survivors, across every run)
    # lives in bert_output_dir. Nemotron writes its own classify.sqlite /
    # final_relevant.jsonl to a SEPARATE directory (nemotron_output_dir) so it
    # judges every paper in pass1_bert.jsonl itself, instead of silently
    # skipping papers an earlier Stage-2 run in bert_output_dir already marked
    # "done".
    input_dir      = ROOT / paths["bert_output_dir"]
    output_dir     = ROOT / paths.get("nemotron_output_dir", "data/processed_nemotron")
    output_dir.mkdir(parents=True, exist_ok=True)
    nemotron_model = paths.get("nemotron_model")
    if not nemotron_model:
        raise ValueError("pipeline_config.yaml: paths.nemotron_model must be set to a local Nemotron checkpoint path")

    run(
        [
            sys.executable,
            str(ROOT / "pipeline_scripts" / "annotate_nemotron.py"),
            "--input_dir",    str(input_dir),
            "--output_dir",   str(output_dir),
            "--model",        nemotron_model,
            "--gpu_mem_util", str(nemotron_cfg.get("gpu_mem_util", 0.80)),
            "--reprocess",
        ],
        cwd=ROOT / "pipeline_scripts",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STAGES: dict[str, object] = {
    "download": stage_download,
    # No separate "dedup_index" entry: stage_bert() always runs it as its
    # first step (see stage_bert()'s docstring comment), so registering it
    # here too would run it twice back-to-back in a full `python pipeline.py`
    # invocation. Run pipeline_scripts/build_dedup_index.py directly if you
    # ever need to rebuild the index in isolation.
    "bert":     stage_bert,
    "nemotron": stage_nemotron,
}


def run_pipeline(cfg: dict, stage: str | None = None, test: bool = False) -> None:
    t0 = time.time()
    names = [stage] if stage else list(STAGES.keys())

    for name in names:
        log(f"--- starting stage: {name} ---")
        try:
            if name == "download":
                stage_download(cfg, test=test)
            else:
                STAGES[name](cfg)  # type: ignore[operator]
        except Exception as e:
            log(f"ERROR in stage '{name}': {e}")
            raise

    elapsed = int(time.time() - t0)
    log(f"Pipeline finished in {elapsed // 60}m{elapsed % 60:02d}s")

    state = load_state()
    state["last_run"] = datetime.now().isoformat()
    state["last_stage"] = stage or "all"
    save_state(state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAlex -> relevant-paper dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", choices=list(STAGES.keys()), default=None,
        help="Run a single stage (default: all stages in order)",
    )
    parser.add_argument(
        "--config", default="pipeline_config.yaml",
        help="Pipeline config file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: download only 3 files (smoke-test the pipeline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, stage=args.stage, test=args.test)


if __name__ == "__main__":
    main()
