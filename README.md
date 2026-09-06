# Dataset Construction Pipeline

Reproducibility copy of the pipeline used to build the animal-communication
paper corpus reported in the survey. This covers dataset construction only —
downloading OpenAlex, filtering, and annotating — not any downstream
search/application code that consumes the resulting dataset.

## Pipeline stages

1. **Download** (`pipeline_scripts/download_snapshot.py`) — pulls OpenAlex
   `works` snapshot partitions (JSONL, gzip-compressed) from the public,
   unsigned `s3://openalex/data/jsonl/works/` bucket. Multi-threaded,
   resumable (skips partitions already downloaded and size-verified).

2. **Dedup index** (`pipeline_scripts/build_dedup_index.py`) — a CPU-only
   pre-pass over every downloaded partition. Applies structural filtering
   (publication year, work type, publication status, language, retraction
   status) and resolves the single latest version of each paper by DOI /
   OpenAlex ID (across snapshot partitions, the same paper can appear more
   than once with different `updated_date`s — the most recent one wins).
   Survivors are written out as chunked Parquet files, so the GPU stage below
   is never wasted scoring a stale/superseded record.

3. **BERT relevance filter** (`pipeline_scripts/filter_bert.py`) — a
   fine-tuned SciBERT (`allenai/scibert_scivocab_uncased`, max sequence
   length 512) binary relevance classifier scores each surviving record's
   title/abstract/venue/keywords/topic text (`pipeline_scripts/text_features.py`
   builds this input text; both training and inference use the exact same
   logic). Records scoring at or above the threshold (0.45 in the reported
   run) proceed to Stage 4. **The trained SciBERT checkpoint and the code
   used to train it are not included in this repo** due to file size limits.
   Contact us at uta.acl2@gmail.com if you'd like access to the model.

4. **Nemotron annotation** (`pipeline_scripts/annotate_nemotron.py`) — every
   BERT-stage survivor is passed to `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
   (served locally via vLLM) for structured metadata extraction (species,
   linguistic/design features, modality, research tradition, etc.) plus a
   second, stricter relevance verification pass. The full system/extraction
   prompt lives in this file (`PASS2_SYSTEM` and the surrounding prompt-
   construction code) — see the file for exact wording. Notable serving
   details:
   - thinking/reasoning is explicitly disabled (`enable_thinking=False`) —
     otherwise the model burns its output-token budget on chain-of-thought
     before emitting the JSON answer.
   - prefix caching is enabled (the ~5.2k-token system prompt is identical
     across every request).
   - the KV cache runs in fp8 for more concurrent-request headroom.
   - the tokenizer is loaded with `fix_mistral_regex=True` (Nemotron's
     tokenizer implementation requires this flag to tokenize correctly).

`pipeline.py` orchestrates all 4 stages above (the dedup index is always run
as the first step of the `bert` stage — see its docstring) and is resumable
at every stage. Run `python pipeline.py --stage <download|bert|nemotron>` for
a single stage, or with no `--stage` flag to run all stages in order.

## Setup

```bash
pip install -r requirements.txt
cp pipeline_config.example.yaml pipeline_config.yaml
# edit pipeline_config.yaml: set snapshot_dir, bert_model, nemotron_model to
# your local paths
```

`configs/bert_classifier.yaml` controls which fields feed the SciBERT input
text (title/abstract/venue/keywords/topic) and the classification threshold;
it's read by `filter_bert.py` at inference time.

## Released dataset

`dataset/final_relevant.parquet` is the final output of this pipeline: the
55,854 papers Nemotron confirmed relevant, projected to a paper-relevant
schema. The figure-generation scripts in `figures/` read this file
directly, so it's the same data that produced the paper's plots. Standard
snappy-compressed Parquet, readable with any Parquet library (`pandas`,
`pyarrow`, `duckdb`, etc.) with no special codec support required.

**Date coverage (locked):** this release reflects an OpenAlex snapshot pull
covering publications up to **2026-06-25**, processed by the pipeline run of
**2026-08-26 to 2026-08-31**. OpenAlex is a living database — re-running this
pipeline today would pull a newer snapshot and produce different candidate
counts and possibly different papers. The numbers in this README and the
paper's figures are frozen to `dataset/final_relevant.parquet` as released
here, not to "OpenAlex as of whenever you read this."

## Scale of the reported run

(See the pipeline's run notes for full detail; summarized here for the paper.)

- Source: OpenAlex `works` snapshot, 2,443 partition files (~620GB
  compressed), covering all years 1950-2026.
- After structural filtering (year/type/publication-status/language/
  retraction) and cross-partition dedup: **~144.4M candidate records**
  (144,431,227) handed to the SciBERT filter.
- SciBERT filter output: **257,036** unique papers passed the 0.45 relevance
  threshold.
- Nemotron annotation output: **55,854** papers confirmed relevant by
  Nemotron's structured verification (after two small top-up passes that
  re-ran silent-failure cases with larger `max_tokens`; ~211 papers, ~0.08%
  of input, remained unparseable after that and are excluded).

### Timing

- Dedup index + SciBERT filter (Phase A + B combined): ~4 days wall-clock on
  a single NVIDIA RTX PRO 6000 Blackwell (~98GB VRAM), on a shared/contended
  multi-user node — not a single clean continuous run, so treat this as an
  approximate figure rather than a clean benchmark.
- Nemotron annotation: ~8h54m for the full 257,036-paper pass (single GPU,
  batch size 2000, `max_tokens=1024`), sustaining ~8 papers/sec. Two small
  top-up passes on the shrinking silent-failure pool added a few more hours
  on top of that.

Hardware: 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU (~98GB
VRAM); a 64-CPU-core shared node for I/O/decompression and dedup-index
construction.

## Not included here

- Any downstream search/application code that consumes the resulting
  dataset — this repo is the dataset construction pipeline only.
- The trained SciBERT checkpoint and the code/labeled data used to train
  it — excluded due to file size limits. Contact uta.acl2@gmail.com for access.
- The raw OpenAlex snapshot and intermediate pipeline outputs (large,
  regenerable from the public OpenAlex bucket via this pipeline).
