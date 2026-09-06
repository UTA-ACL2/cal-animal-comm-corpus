#!/usr/bin/env python3
"""
annotate_nemotron.py — LLM annotation and relevance-verification pipeline.

Runs NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (served via vLLM) over every
BERT-stage survivor for strict relevance verification plus full structured
metadata extraction. Prompting goes through the model's own tokenizer chat
template rather than a hand-rolled prompt string.

Two resume/gap-filling flags:
  - --reprocess: targets only silent failures (score=0, empty/garbage
    summary) and never-processed papers. Use for targeted gap-filling.
  - --reprocess-all-rejected: re-runs ALL papers not in final_relevant.jsonl.
    Every DB record with is_relevant=0 is un-marked and re-queued, regardless
    of whether it was a silent failure or a real rejection. Use this when a
    prior run had GPU/quality issues and you want a full second opinion on
    every rejected paper. Papers with is_relevant=1 are always skipped.
  - Both flags can coexist; --reprocess-all-rejected is the strict superset.
"""

import argparse, json, os, re, sqlite3, time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DEFAULT_OUTPUT_DIR  = "./corpus_output"
NEMOTRON_MODEL      = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"  # fallback if --model isn't passed; set to a local checkpoint path
NEMOTRON_BATCH_SIZE = 100  # even smaller batch for the stubborn-tail --reprocess pass:
# reduces per-batch fallout if a single oversized prompt still trips the whole-batch
# VLLMValidationError path (see _ABSTRACT_TOKEN_CAP comment below), and speeds up seeing
# per-batch progress on the small reprocess set.
PRINT_EVERY         = 2000

# Summaries that indicate a silent failure (model never gave a real answer)
# --reprocess will re-run any DB record whose summary matches these
SILENT_FAILURE_MARKERS = {"silent_failure", "parse_error", "rejected", "", "template_leak"}

# Exact "not relevant" summary text ever used as a literal example in PASS2_SYSTEM's
# OUTPUT FORMAT block. If the model copies one of these verbatim instead of writing a
# paper-specific reason, that's not a real judgement -- flag it as template_leak so
# --reprocess re-runs it instead of trusting it as a genuine rejection.
_TEMPLATE_LEAK_SUMMARIES = {
    "not relevant: industrial gearbox vibration analysis, no animal data",
    "not relevant: pure industrial vibration analysis, no animal data",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt(s):
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


# ============================================================
# OPENALEX TEXT PARSING
# ============================================================

def reconstruct_abstract(inv) -> str:
    if not inv or not isinstance(inv, dict):
        return ""
    try:
        pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
        pairs.sort()
        return " ".join(w for _, w in pairs)
    except Exception:
        return ""


def _safe_str(x) -> str:
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


def extract_paper_text(paper: dict) -> dict:
    title    = (paper.get("title") or paper.get("display_name") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))

    raw_topics   = paper.get("topics")   or []
    raw_concepts = paper.get("concepts") or []
    raw_keywords = paper.get("keywords") or []

    topics   = [s for t in raw_topics[:8]   if (s := _safe_str(t).strip())]
    concepts = [s for c in raw_concepts[:8] if (s := _safe_str(c).strip())]
    keywords = [s for k in raw_keywords[:8] if (s := _safe_str(k).strip())]

    pt = paper.get("primary_topic")
    topic_field = ""
    if isinstance(pt, dict):
        raw = pt.get("field") or pt.get("subfield") or ""
        topic_field = _safe_str(raw).strip()
    elif isinstance(pt, str):
        topic_field = pt.strip()

    return {
        "title":       title,
        "abstract":    abstract[:2000],
        "topics":      topics,
        "concepts":    concepts,
        "keywords":    keywords,
        "topic_field": topic_field,
    }


# ============================================================
# PROMPTS
# ============================================================

PASS2_SYSTEM = """You are a Research Metadata Extractor and Quality Verifier for a broad
interdisciplinary corpus covering Animal Communication and Bioacoustics.

This corpus spans THREE equally valid research traditions:
  (1) BIOLOGICAL — ethological/ecological studies of how and why animals communicate
      (bioacoustics, behavioral ecology, call function, vocal evolution, acoustic ecology)
  (2) COMPUTATIONAL — ML, deep learning, or signal processing applied to animal signals
      (species ID, PAM pipelines, foundation models, audio classifiers)
  (3) ANIMAL LINGUISTICS — structural analysis of animal communication systems
      (syntax, compositionality, vocal learning, referential calls, repertoire structure)

A paper does NOT need to use computational methods to belong in this corpus.
Pure biological field or lab studies of animal vocalizations are equally relevant.

================================================================================
TASK 1: STRICT VERIFICATION
================================================================================
Set is_relevant=false ONLY if the paper is CLEARLY about:
- Human speech, language disorders, or NLP with no animal data and no explicit
  comparison to animal communication systems
- Clinical/medical studies using animals ONLY as models for human disease
  (e.g., mouse USVs used exclusively to screen autism drugs — the animal is
  a research tool, not the subject of communication study)
- General ML, robotics, or computer vision with no biological animal data of
  any kind (no animal audio, video, behavior tracking, or signals)
- Industrial signal processing (motors, gears, pipelines, machinery vibration)
- Optimization algorithms named after animals (whale optimizer, bat algorithm,
  grey wolf optimizer, etc.)

Set is_relevant=true if the paper:
- Studies animal communication, vocalizations, or signaling in ANY modality,
  using ANY methodology — biological field study, lab experiment, or computational
- Investigates bioacoustics: sound production, structure, perception, or function
  in any animal species (even with zero ML or computation)
- Uses ML or computational methods specifically on animal audio, video of
  communicative displays, or multimodal animal signal data
- Analyzes call repertoires, vocal sequences, or signal structure for any purpose
- Studies vocal learning, cultural transmission, or individual recognition
- Monitors wildlife acoustically (PAM, soundscape ecology, biodiversity surveys)
- Performs animal linguistics (syntax, compositionality, referential calls, etc.)

IMPORTANT — new-species/taxonomy papers: a paper describing a new species or
subspecies is_relevant=true if it documents the animal's calling song,
advertisement call, or other acoustic/vocal signal (even briefly, even as one
part of a broader taxonomic description). The taxonomy/species-description
framing does NOT disqualify it — the calling song or advertisement call IS
bioacoustics data. Do not reject a paper as "just taxonomy" if the
title/abstract mentions a call, song, or acoustic signal was recorded or
described.

================================================================================
TASK 2: METADATA EXTRACTION (only when is_relevant=true)
================================================================================

INFERENCE RULES — these differ by field, read carefully:

► SPECIES:
  - Extract ONLY animals explicitly named or clearly described.
  - Do NOT invent or infer unstated species.
  - Use the name exactly as written in the text.
  - IMPORTANT: Bats (including any species of Chiroptera, e.g. Myotis, Eptesicus,
    Rhinolophus, Pteropus, Tadarida, Noctilio) must have category = "Bat", not "Other".

► STAGES, FEATURES, MODALITIES, CONTEXTS, DOMAINS, SOCIAL SCALE,
  RESEARCH TRADITION, PAPER TYPE:
  - USE YOUR JUDGMENT and INFER from the paper's subject, methods, and framing.
  - These labels rarely appear verbatim — you must map what the paper does onto
    the fixed list.
  - Examples:
      "deep learning classifier for bat echolocation calls"
        → stage: Analysis & Classification
        → modality: Acoustic
        → research_tradition: computational
      "we recorded free-ranging dolphins at sea"
        → context: Field / Wild
        → social_scale: group (social animals recorded in natural groups)
      "juvenile songbirds copy tutor songs"
        → feature: Vocal Learning
        → social_scale: dyadic (tutor-pupil pair)
  - The evidence field must be a real phrase from the text that SUPPORTS your
    inference — choose the most relevant phrase available.

► HAS_DATASET, HAS_CODE, HAS_BENCHMARK:
  - Set true ONLY if explicitly stated in title or abstract
    (e.g. "we release a dataset", "code available at", "benchmark").
  - Do NOT infer these from context.

► EXTRA FIELDS:
  - If the paper has properties that clearly do not fit any item in the fixed
    lists (features, stages, domains, contexts), you MAY add them as free-text
    entries with "out_of_schema": true and an evidence phrase.
  - Use this sparingly — only for genuinely distinct concepts not covered.
  - Example: a paper on vocal rhythm and beat perception could add
    {"feature": "Beat Induction", "evidence": "...", "out_of_schema": true}

────────────────────────────────────────────────────────────────────────────────

All fields must be inferred ONLY from the TITLE and ABSTRACT provided.
Do NOT use any outside knowledge or web search.

1. SPECIES
   List every animal species or taxonomic group mentioned.
   - name: exactly as written in the text (e.g. "humpback whale", "Mus musculus", "songbird")
   - scientific_name: the currently ACCEPTED binomial nomenclature (Genus species) for
     this taxon, per standard taxonomic authorities (GBIF Backbone Taxonomy / ITIS /
     NCBI Taxonomy) — NOT a taxonomic synonym or outdated name. This is for
     normalizing species across papers, so always resolve common names and synonyms
     to the currently accepted name (e.g. "blue tit" -> "Cyanistes caeruleus"; an
     older/synonym scientific name used in the text -> today's accepted name).
     If the text only names a genus, family, or broader group (e.g. "songbird",
     "chiroptera"), give the genus/family name instead of a binomial.
     If you are not confident of the accepted name, leave this field as an empty string
     rather than guess.
   - category: ONE of — Terrestrial Mammal | Marine Mammal | Bird | Primate | Bat
               | Amphibian | Insect | Fish | Reptile | Other
     NOTE: All bats and chiropteran species → "Bat" (never "Other")
   - evidence: exact phrase from TITLE or ABSTRACT that names or describes the animal

2. COMPUTATIONAL STAGES — select ALL that apply, from this fixed list ONLY.
   If a stage clearly applies but doesn't fit, add it with "out_of_schema": true.
   - Data Collection          : field recording, lab capture, crowdsourcing, existing datasets
   - Preprocessing            : denoising, source separation, sound event detection, segmentation
   - Sequence Representation  : unit discovery, tokenization, syllable segmentation, spectrograms
   - Analysis & Classification: ML classifiers, species ID, call type classification,
                                individual ID, emotion recognition
   - Meaning Identification   : linking calls to semantic meaning, behavior, or context
   - Generation               : synthesizing animal sounds, generative models, playback stimuli
   - Foundation Model         : large pretrained models, self-supervised learning,
                                transfer learning for audio
   For each entry provide:
   - stage    : exact name from the list above (copy exactly, do not paraphrase)
   - evidence : phrase from TITLE or ABSTRACT that supports this stage

3. LINGUISTIC FEATURES — select ALL that apply, from this fixed list ONLY.
   For each, a STRICT EVIDENCE THRESHOLD is given — if the abstract does not
   meet that threshold, do NOT include the feature.
   If a linguistic property clearly applies but is not in this list, add it
   with "out_of_schema": true and an evidence phrase.

   - Vocal Auditory Channel
       ✓ Include if: paper studies any animal sound, call, song, vocalization,
         echolocation, or ultrasound
       ✗ Exclude if: paper is purely visual/chemical/seismic signaling with
         no acoustic component
       Evidence must contain: a word related to sound, call, vocalization,
         audio, acoustic, or song

   - Turn-taking
       ✓ Include if: paper explicitly studies alternating, antiphonal, or
         duet-style vocal exchanges; response latency between callers
       ✗ Exclude if: paper merely records multiple animals — co-occurrence
         of calls is NOT turn-taking
       Evidence must contain: "alternating", "antiphonal", "duet", "response
         latency", "counter-singing", or "vocal exchange"

   - Reference
       ✓ Include if: paper studies calls that designate specific external
         objects, predators, food, or events (referential signals, alarm calls,
         food calls, functionally referential, contact calls with identity info)
       ✗ Exclude if: paper simply classifies call types without studying
         what the calls refer to
       Evidence must contain: "referential", "alarm", "food call", "predator",
         "contact call", "identify", or explicit description of call-object mapping

   - Displacement
       ✓ Include if: paper explicitly studies communication about events not
         currently present in time or space (e.g., bee waggle dance direction,
         past predator encounter)
       ✗ Exclude if: the paper only studies calls given in the presence of the
         stimulus — displacement requires ABSENCE of the referent
       Evidence must contain: explicit mention of absent referent, past event,
         or future event being communicated about
       BASE RATE: assign to fewer than 5% of papers — this is genuinely rare

   - Syntax
       ✓ Include if: paper studies the sequential arrangement, ordering rules,
         combinatorial structure, or grammar of vocal units; phrase structure;
         note sequences; motif ordering
       ✗ Exclude if: paper only studies single call types or classifies
         individual notes without studying their combination or ordering
       Evidence must contain: "sequence", "syntax", "combinatorial", "order",
         "phrase", "motif structure", "grammar", or "arrangement"

   - Recursion
       ✓ Include if: paper explicitly studies self-embedded, hierarchically
         nested, or iteratively repeated syntactic structures in vocalizations
       ✗ Exclude if: paper studies sequences or repetition without explicit
         hierarchical embedding claim
       Evidence must contain: "recursive", "nested", "hierarchical structure",
         or "self-embedded"
       BASE RATE: assign to fewer than 5% of papers — this is genuinely rare

   - Semanticity
       ✓ Include if: paper studies calls that carry specific meaning, semantic
         content, or information beyond simple arousal; meaning of call types
       ✗ Exclude if: paper studies call acoustics or classification without
         addressing meaning or information content
       Evidence must contain: "meaning", "semantic", "information content",
         "signal content", "encode", or explicit discussion of what a call means

   - Cultural Transmission
       ✓ Include if: paper studies how vocal traditions, dialects, or
         repertoires spread across generations or between individuals through
         learning (not genetic inheritance)
       ✗ Exclude if: paper studies vocal learning in a single individual
         without addressing population-level or cross-generational spread
       Evidence must contain: "cultural", "tradition", "transmission",
         "generational", "population-level learning", or "spread across"

   - Vocal Learning
       ✓ Include if: paper studies acquisition of vocalizations through
         imitation, tutoring, or experience; song learning; vocal development
         in juveniles; mimicry
       ✗ Exclude if: paper studies innate calls that don't require learning,
         or only uses call recordings without studying the acquisition process
       Evidence must contain: "learning", "imitation", "tutor", "acquisition",
         "development", "juvenile", "mimic", or "copy"

   - Discreteness
       ✓ Include if: paper studies categorical, distinct call types or
         note categories; discrete unit inventories; repertoire structure
         with distinct elements
       ✗ Exclude if: paper only studies continuous acoustic variation
         (e.g., graded signals, continuous frequency modulation) without
         identifying discrete categories
       Evidence must contain: "call type", "repertoire", "discrete",
         "categorical", "unit", "element", or "note type"

   - Individual Variation
       ✓ Include if: paper studies acoustic differences between individuals;
         individual recognition by voice; signature calls or whistles;
         individual identity encoded in calls; comparing repertoire size or
         acoustic consistency across individuals; caller identification
       ✗ Exclude if: paper pools all recordings without any individual-level
         analysis whatsoever
       Evidence must contain: "individual", "recognition", "identity",
         "signature", "between individuals", "caller", "repertoire size",
         "vocal consistency", or "personally distinctive"

   - Dialect
       ✓ Include if: paper studies geographic or group-level variation in
         vocalizations; regional song variants; population-level vocal
         differences across sites
       ✗ Exclude if: paper studies individual variation or temporal change
         within a single population at a single site
       Evidence must contain: "dialect", "geographic variation", "regional",
         "population differences", "site", or "between populations"

   - Emotion/Affect
       ✓ Include if: paper studies how emotional or affective states
         (fear, stress, arousal, valence, pain) are expressed or encoded
         in vocalizations; acoustic correlates of welfare states
       ✗ Exclude if: paper uses emotional framing loosely without studying
         acoustic correlates of affective states; do not infer from
         "distress calls" alone unless acoustics-affect link is explicit
       Evidence must contain: "emotion", "affect", "stress", "arousal",
         "valence", "welfare", "pain", or explicit acoustic-affect mapping

   HARD RULE: If you cannot find a phrase in the TITLE or ABSTRACT that
   satisfies the evidence threshold above, do NOT include that feature.
   It is better to under-assign schema features than to hallucinate them.
   Use "out_of_schema": true for genuinely novel properties you observe.

4. SIGNAL MODALITY — physical channel the signal travels through. Select ALL that apply:
   - Acoustic    : sound waves (vocalizations, calls, songs, ultrasound, infrasound, clicks, echolocation)
   - Visual      : light-based signals (body posture, color, gesture, facial expression, bioluminescence, display)
   - Chemical    : molecular signals (pheromones, scent marking, olfactory cues)
   - Tactile     : direct physical contact signals (grooming, touch)
   - Seismic     : substrate-borne vibration (drumming, tremulation, footfalls)
   - Electrical  : electric organ discharge (weakly/strongly electric fish)
   - Multimodal  : paper explicitly studies two or more modalities combined as a signal
   - Other       : any modality not listed (note in evidence)
   For each entry provide:
   - modality : exact name from the list above
   - evidence : phrase from TITLE or ABSTRACT that supports this modality
   DEFAULT: if the paper clearly studies vocalizations, calls, songs, or echolocation
   and no modality is mentioned, output:
     [{"modality": "Acoustic", "evidence": "<title or key phrase>"}].

5. RESEARCH CONTEXT — where and how the data was collected. Select ALL that apply:
   - Field / Wild             : data from free-living wild animals in natural habitat
   - Lab / Controlled         : laboratory, anechoic chamber, or controlled enclosure
   - Captive / Zoo            : zoo, aquarium, sanctuary, or farm animals
   - Passive Acoustic Mon.    : autonomous unattended recorders (ARUs, hydrophones, buoys, PAM)
   - Playback Experiment      : animals' responses to broadcast stimuli were tested
   - Citizen Science          : crowd-sourced recordings (Xeno-canto, iNaturalist, Macaulay Library)
   - Simulation / Synthetic   : synthetic, simulated, or augmented data — no live animals;
                                ALSO use this when a paper's primary output is generated/synthetic signals
   - Existing Dataset         : paper reuses a previously published dataset without new collection
   - Other                    : context not listed above
   For each entry provide:
   - context  : exact name from the list above
   - evidence : phrase from TITLE or ABSTRACT that supports this context
   INFER:
     - "long-term acoustic monitoring" → Passive Acoustic Mon.
     - "sound-proof chambers" → Lab / Controlled
     - "Xeno-canto recordings" → Citizen Science + Existing Dataset
     - "we generate / synthesize vocalizations" → Simulation / Synthetic

6. APPLICATION DOMAIN — real-world goal or use case. Select ALL that apply.
   Read these definitions carefully — many papers are mis-tagged on domains:

   - Species Identification    : automated detection or classification of WHICH SPECIES is present
   - Individual ID             : identifying specific INDIVIDUAL ANIMALS by voice or signal
   - Conservation & Ecology    : biodiversity monitoring, population estimation, habitat assessment,
                                 climate impact on wildlife — NOT for behavioural ecology studies
                                 of communication function (e.g., sexual selection papers do NOT
                                 belong here unless they address population monitoring)
   - Animal Welfare            : monitoring stress, pain, or emotional state via signals in
                                 CAPTIVE or managed animals — NOT for wild behavioural studies
                                 unless welfare is explicitly the stated goal
   - Decoding / Translation    : inferring semantic meaning, referential content, or INTENT
                                 from signals; what a call MEANS or COMMUNICATES
   - Vocal Learning & Culture  : how vocalizations are ACQUIRED or culturally transmitted
   - Human-Animal Interaction  : animal responses to humans, or intentional interspecies communication
   - Cross-Species Interaction : heterospecific eavesdropping, alarm-call recognition between species
   - Neuroscience              : neural mechanisms of vocal production, perception, or learning
   - Bioacoustic Methods       : new recording hardware, annotation tools, signal pipelines,
                                 or new COMPUTATIONAL METHODS for analysing animal sounds
                                 (GANs, classifiers, new algorithms applied to animal audio)
   - Soundscape Ecology        : acoustic analysis of entire habitat or multi-species community
   - Noise & Anthropogenic     : effects of human-generated noise on animal communication or welfare
   - Other                     : domain not listed above; also use for behavioural ecology studies
                                 of communication function (e.g., intrasexual competition, mate
                                 attraction) that don't fit the above categories

   For each entry provide:
   - domain  : exact name from the list above
   - evidence: phrase from TITLE or ABSTRACT that supports this domain

7. SOCIAL SCALE — level of social organization studied. Choose ONE best match:
   - individual       : single animal's signal production or perception studied in isolation
   - dyadic           : pairwise interaction (courtship duet, mother-infant, tutor-pupil, rival pair)
   - group            : communication within a group, flock, pod, or colony
   - population       : population-level variation, dialects, or geographic signal distribution
   - community        : multi-species acoustic community or soundscape with multiple species
   - cross-species    : interspecific signal sharing, eavesdropping, mixed-species groups
   - not_applicable   : paper does not study a specific social scale (pure methods/dataset paper)
   INFER from the study design, not from explicit mention of these words.

8. RESEARCH TRADITION — primary disciplinary framing. Choose ONE:
   - biological    : primarily ethological, ecological, or behavioral biology approach
   - computational : primarily ML, signal processing, NLP, or AI approach
   - hybrid        : substantial integration of both biological and computational methods
   INFER:
     - presence of CNNs, GANs, classifiers, computational models → at least computational
     - behavioral experiments, evolutionary/functional hypotheses → biological
     - both together → hybrid
     - neuroscience with computational modelling → hybrid

9. PAPER TYPE — Choose ONE:
   - empirical    : original research with collected biological data and experiments
   - dataset      : introduces or describes a new dataset or benchmark
   - method       : proposes a new computational METHOD, model, or algorithm
                    (use this when the primary contribution is a new technique,
                     even if validated on animal data)
   - review       : literature survey or systematic review
   - theoretical  : theoretical framework, model, or hypothesis paper
   - tool         : software library, annotation tool, or platform
   - irrelevant   : use ONLY when is_relevant=false

10. RESOURCES — set true ONLY if explicitly stated in title or abstract:
    - has_dataset   : paper introduces or publicly releases a labeled dataset
    - has_code      : paper releases code, model weights, or a software library
    - has_benchmark : paper proposes or evaluates on a formal benchmark

11. RELEVANCE SCORE — integer 0-10:
    10 = paper is entirely about animal communication/signaling
     7 = animal communication is the primary focus
     5 = animal communication is one of several equal topics
     3 = animal signals mentioned but not the main focus
     1 = very peripheral mention of animal signals
     0 = not relevant (use only when is_relevant=false)

12. SUMMARY — one sentence, max 30 words, describing what THIS paper
    specifically contributes to animal communication or bioacoustics — it
    must reference this paper's actual species/topic/method, not a generic
    phrase. If is_relevant=false, write "Not relevant: " followed by a
    one-phrase reason specific to THIS paper's actual subject matter (what
    it is really about, and why that's outside animal communication/
    bioacoustics). The OUTPUT FORMAT examples below are illustrative only —
    copying their summary text verbatim, or writing a generic reason like
    "no animal data" that ignores the paper's real topic, is an error.

================================================================================
STRICT OUTPUT RULES — VIOLATING THESE CAUSES OUTPUT TO BE DISCARDED
================================================================================
- Output ONLY a single valid JSON object — no markdown, no code fences, no extra text
- is_relevant MUST be boolean: true or false (not the string "true" or "false")
- relevance_score MUST be an integer 0-10 (not a float, not a string)
- species, stages, features, modalities, contexts, domains MUST be JSON arrays
  ([] if none, never null)
- scientific_name MUST be a string (empty string "" if unknown/not confident) —
  never null, never omitted
- social_scale MUST be exactly one string from the list above
- research_tradition MUST be exactly one of: "biological", "computational", "hybrid"
- paper_type MUST be exactly one string from the list above
- has_dataset, has_code, has_benchmark MUST be booleans: true or false
- ALL fixed-list names MUST be copied EXACTLY — do NOT paraphrase or invent labels
- For items NOT in the fixed list: include them with "out_of_schema": true
- evidence MUST be a real phrase from the TITLE or ABSTRACT — not invented
- If the abstract is very short or missing, use the title for all evidence

OUTPUT FORMAT (relevant):
{
  "is_relevant": true,
  "species": [{"name": "zebra finch", "scientific_name": "Taeniopygia guttata", "category": "Bird", "evidence": "juvenile zebra finches"}],
  "stages": [{"stage": "Analysis & Classification", "evidence": "deep learning classifier"}],
  "features": [{"feature": "Vocal Learning", "evidence": "copy tutor songs"}],
  "modalities": [{"modality": "Acoustic", "evidence": "song recordings"}],
  "contexts": [{"context": "Lab / Controlled", "evidence": "sound-proof chambers"}],
  "domains": [{"domain": "Vocal Learning & Culture", "evidence": "song acquisition in juveniles"}],
  "social_scale": "dyadic",
  "research_tradition": "hybrid",
  "paper_type": "empirical",
  "has_dataset": false,
  "has_code": false,
  "has_benchmark": false,
  "relevance_score": 9,
  "summary": "Examines how juvenile zebra finches copy tutor songs using deep learning analysis of lab recordings."
}

OUTPUT FORMAT (not relevant):
{
  "is_relevant": false,
  "species": [], "stages": [], "features": [], "modalities": [], "contexts": [], "domains": [],
  "social_scale": "not_applicable",
  "research_tradition": "biological",
  "paper_type": "irrelevant",
  "has_dataset": false, "has_code": false, "has_benchmark": false,
  "relevance_score": 0,
  "summary": "Not relevant: <write the REAL reason for THIS paper here, e.g. its actual topic and why it has no animal-communication content — do not copy this bracketed text>"
}"""

# ============================================================
# PROMPT BUILDER
# ============================================================
# Rather than hand-building chat-template tags (which vary by model family),
# we defer to Nemotron's own tokenizer chat template.
# _TOKENIZER is set once in run_stage_2() after the model path is known.
_TOKENIZER = None


# PASS2_SYSTEM alone is ~5,237 tokens against an 8,192-token context (with
# max_tokens reserved for output), leaving only limited budget for
# TITLE/ABSTRACT/TOPICS/CONCEPTS/KEYWORDS combined. A single oversized
# abstract (garbled inverted-index reconstruction, huge front matter, etc.)
# pushes the whole prompt over the model's context limit — and since prompts
# are sent to llm.generate() as a batch, that one record kills every paper
# in the batch with a VLLMValidationError instead of failing just itself
# (confirmed: this crashed the entire pipeline process at 28k/134,790 papers
# processed). Cap the abstract token-wise to keep every prompt in budget.
_ABSTRACT_CHAR_PREFILTER = 24000  # ~6,000 tokens at ~4 chars/token; cheap guard before tokenizing
_ABSTRACT_TOKEN_CAP = 6000
# max_model_len(16384) - max_tokens(6144, further raised from 3072 for the stubborn-tail
# --reprocess pass: these papers still silent-failed at max_tokens=3072, so give even more
# generation room), minus a safety margin for chat-template scaffolding tokens (role tags,
# BOS, generation prompt, etc).
_PROMPT_TOKEN_BUDGET = 16384 - 6144 - 100


def _build_prompt(title, abstract, topics, concepts, keywords):
    body = (f"TITLE: {title}\n\n"
            f"ABSTRACT: {abstract}\n\n"
            f"TOPICS: {topics}\n"
            f"CONCEPTS: {concepts}\n"
            f"KEYWORDS: {keywords}")
    messages = [
        {"role": "system", "content": PASS2_SYSTEM},
        {"role": "user", "content": body},
    ]
    # enable_thinking=False: this is a reasoning model that otherwise emits a
    # long chain-of-thought before the JSON answer, blowing past max_tokens
    # on most papers (verified empirically — thinking on produced 19/20
    # silent failures on a smoke sample; thinking off produced clean JSON).
    return _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def make_pass2_prompt(p: dict) -> str:
    parsed   = extract_paper_text(p)
    title    = parsed["title"]
    abstract = parsed["abstract"]
    topics   = ", ".join(parsed["topics"])
    concepts = ", ".join(parsed["concepts"])
    keywords = ", ".join(parsed["keywords"])

    if len(abstract) > _ABSTRACT_CHAR_PREFILTER:
        ids = _TOKENIZER(abstract)["input_ids"]
        if len(ids) > _ABSTRACT_TOKEN_CAP:
            abstract = _TOKENIZER.decode(ids[:_ABSTRACT_TOKEN_CAP])

    prompt = _build_prompt(title, abstract, topics, concepts, keywords)

    # Final hard check: TOPICS/CONCEPTS/KEYWORDS can independently be huge for
    # papers with many OpenAlex tags, so the abstract cap alone isn't
    # sufficient (confirmed empirically: worst case in a 5k sample hit 7,763
    # tokens against a 7,168 budget even with the abstract capped at 1,000).
    # Shrink the abstract further by exactly the observed overflow rather than
    # guessing a lower cap — keeps the common case (short abstract, no
    # truncation at all) as cheap as before.
    n_tokens = len(_TOKENIZER(prompt)["input_ids"])
    if n_tokens > _PROMPT_TOKEN_BUDGET:
        overflow = n_tokens - _PROMPT_TOKEN_BUDGET
        abstract_ids = _TOKENIZER(abstract)["input_ids"]
        keep = max(0, len(abstract_ids) - overflow)
        abstract = _TOKENIZER.decode(abstract_ids[:keep])
        prompt = _build_prompt(title, abstract, topics, concepts, keywords)

    return prompt


# ============================================================
# SQLITE STATE MANAGER
# ============================================================

class ClassifyDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-32000")
        self._create_tables()
        self._s2 = set()
        self._load_sets()
        self._pending_s2 = []

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS processed_s2 (
                oa_id            TEXT PRIMARY KEY,
                is_relevant      INTEGER DEFAULT 0,
                relevance_score  INTEGER DEFAULT 0,
                paper_type       TEXT,
                summary          TEXT,
                processed_at     TEXT
            );
        """)
        self.conn.commit()
        # source_updated_date: the OpenAlex updated_date of the pass1_bert.jsonl
        # row that produced this verdict. Lets a later run detect that OpenAlex
        # has since republished a richer version of the same paper (e.g. an
        # abstract backfilled onto a stub) and re-judge it instead of trusting
        # a verdict made on stale/thin data forever. Added after the column
        # didn't exist in earlier DBs, hence the migration guard.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(processed_s2)")}
        if "source_updated_date" not in cols:
            self.conn.execute(
                "ALTER TABLE processed_s2 ADD COLUMN source_updated_date TEXT DEFAULT ''"
            )
            self.conn.commit()

    def _load_sets(self):
        self._s2 = {
            r[0]: (r[1] or "")
            for r in self.conn.execute(
                "SELECT oa_id, source_updated_date FROM processed_s2"
            )
        }
        log(f"  DB loaded: {len(self._s2):,} Stage2 records")

    def s2_done(self, oa_id):
        return oa_id in self._s2

    def s2_stale(self, oa_id, source_updated_date):
        """True if `source_updated_date` (from the current pass1_bert.jsonl row)
        is newer than what this oa_id's stored verdict was judged from — i.e.
        OpenAlex has republished this paper since we last judged it."""
        if not source_updated_date or oa_id not in self._s2:
            return False
        return source_updated_date > self._s2[oa_id]

    def mark_s2(self, oa_id, is_relevant, score, paper_type, summary, source_updated_date=""):
        if oa_id in self._s2:
            return
        self._s2[oa_id] = source_updated_date
        self._pending_s2.append((
            oa_id, 1 if is_relevant else 0, score, paper_type,
            summary, datetime.now().isoformat(), source_updated_date
        ))

    def overwrite_s2(self, oa_id, is_relevant, score, paper_type, summary, source_updated_date=""):
        """Used by --reprocess and by staleness-triggered reprocessing to
        update an existing record in place."""
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_s2 "
            "(oa_id,is_relevant,relevance_score,paper_type,summary,processed_at,source_updated_date) "
            "VALUES (?,?,?,?,?,?,?)",
            (oa_id, 1 if is_relevant else 0, score, paper_type,
             summary, datetime.now().isoformat(), source_updated_date)
        )
        self.conn.commit()
        self._s2[oa_id] = source_updated_date

    def get_silent_failures(self) -> set:
        """
        Return oa_ids of records that are silent failures:
        is_relevant=0, relevance_score=0, and summary indicates no real Nemotron decision.
        These are papers where Nemotron produced empty/garbage output (GPU OOM, crash, etc.)
        and were wrongly rejected without any real classification.
        """
        rows = self.conn.execute(
            "SELECT oa_id, summary FROM processed_s2 "
            "WHERE is_relevant=0 AND relevance_score=0"
        ).fetchall()
        failures = set()
        for oa_id, summary in rows:
            s = (summary or "").strip().lower()
            if s in SILENT_FAILURE_MARKERS:
                failures.add(oa_id)
        log(f"  Silent failures in DB: {len(failures):,}")
        return failures

    def get_all_rejected(self) -> set:
        """
        Return oa_ids of ALL records with is_relevant=0.
        Used by --reprocess-all-rejected to re-run every paper that did not
        make it into final_relevant.jsonl, regardless of rejection reason.
        Papers with is_relevant=1 are never touched.
        """
        rows = self.conn.execute(
            "SELECT oa_id FROM processed_s2 WHERE is_relevant=0"
        ).fetchall()
        result = {r[0] for r in rows}
        log(f"  All rejected in DB: {len(result):,}")
        return result

    def flush_s2(self):
        if not self._pending_s2:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO processed_s2 "
            "(oa_id,is_relevant,relevance_score,paper_type,summary,processed_at,source_updated_date) "
            "VALUES (?,?,?,?,?,?,?)",
            self._pending_s2)
        self.conn.commit()
        self._pending_s2.clear()

    def s2_count(self):
        return len(self._s2)

    def close(self):
        self.flush_s2()
        self.conn.close()


# ============================================================
# OUTPUT WRITER
# ============================================================

class OutputWriter:
    def __init__(self, path):
        self.path = path
        self._buf = []

    def add(self, paper):
        self._buf.append(paper)

    def flush(self):
        if not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for p in self._buf:
                f.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + "\n")
        self._buf.clear()

    def count_existing(self):
        if not os.path.exists(self.path):
            return 0
        return sum(1 for _ in open(self.path, encoding="utf-8"))


# ============================================================
# LEGACY MIGRATION
# ============================================================

def migrate_legacy(db, output_dir):
    log("Checking legacy files to migrate into DB...")
    s2_files = [
        os.path.join(output_dir, "final_classified.jsonl"),
        os.path.join(output_dir, "relevant_papers_only.jsonl"),
        os.path.join(output_dir, "final_relevant.jsonl"),
    ]
    s2_migrated = 0
    for fpath in s2_files:
        if not os.path.exists(fpath):
            continue
        log(f"  Migrating Stage2 legacy: {fpath}")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                oa_id = _get_oa_id(p)
                if not oa_id or db.s2_done(oa_id):
                    continue
                is_rel = bool(p.get("is_relevant",
                              p.get("classification", {}).get("relevant", True)))
                score  = int(p.get("relevance_score", 5 if is_rel else 0))
                ptype  = str(p.get("paper_type", "unknown"))
                summ   = str(p.get("ai_summary", ""))
                db.mark_s2(oa_id, is_rel, score, ptype, summ)
                s2_migrated += 1
    db.flush_s2()
    log(f"  Stage2 migration: {s2_migrated:,} new records  |  DB total: {db.s2_count():,}")


# ============================================================
# ID HELPER
# ============================================================

def _get_oa_id(paper: dict):
    raw = paper.get("id") or paper.get("oa_id") or ""
    m = re.search(r'(W\d+)', str(raw))
    if m:
        return m.group(1)
    doi = paper.get("doi") or ""
    if doi:
        return "doi:" + doi.lower().strip() \
            .replace("https://doi.org/", "").replace("http://doi.org/", "")
    return None


# ============================================================
# JSON HELPERS
# ============================================================

def extract_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


# ============================================================
# VALID VALUE SETS
# ============================================================

VALID_SOCIAL_SCALES = {
    "individual", "dyadic", "group", "population",
    "community", "cross-species", "not_applicable",
}
VALID_TRADITIONS   = {"biological", "computational", "hybrid"}
VALID_PAPER_TYPES  = {
    "empirical", "dataset", "method", "review",
    "theoretical", "tool", "irrelevant",
}
VALID_MODALITIES = {
    "Acoustic", "Visual", "Chemical", "Tactile",
    "Seismic", "Electrical", "Multimodal", "Other",
}
VALID_CONTEXTS = {
    "Field / Wild", "Lab / Controlled", "Captive / Zoo",
    "Passive Acoustic Mon.", "Playback Experiment",
    "Citizen Science", "Simulation / Synthetic",
    "Existing Dataset", "Other",
}
VALID_DOMAINS = {
    "Species Identification", "Individual ID",
    "Conservation & Ecology", "Animal Welfare",
    "Decoding / Translation", "Vocal Learning & Culture",
    "Human-Animal Interaction", "Cross-Species Interaction",
    "Neuroscience", "Bioacoustic Methods",
    "Soundscape Ecology", "Noise & Anthropogenic", "Other",
}
VALID_FEATURES = {
    "Vocal Auditory Channel", "Turn-taking", "Reference", "Displacement",
    "Syntax", "Recursion", "Semanticity", "Cultural Transmission",
    "Vocal Learning", "Discreteness", "Individual Variation",
    "Dialect", "Emotion/Affect",
}
VALID_STAGES = {
    "Data Collection", "Preprocessing", "Sequence Representation",
    "Analysis & Classification", "Meaning Identification",
    "Generation", "Foundation Model",
}

_BAT_KEYWORDS = {
    "bat", "bats", "chiroptera", "chiropteran",
    "myotis", "eptesicus", "rhinolophus", "pteropus",
    "tadarida", "noctilio", "pipistrellus", "molossus",
    "vespertilionidae", "phyllostomidae", "pteropodidae",
}


def _fix_bat_category(species_list: list) -> list:
    for sp in species_list:
        if not isinstance(sp, dict):
            continue
        name_lower = sp.get("name", "").lower()
        if any(kw in name_lower for kw in _BAT_KEYWORDS):
            sp["category"] = "Bat"
    return species_list


def _validate_list_field(items, valid_set, name_key):
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        val = item.get(name_key, "")
        if not isinstance(val, str) or not val.strip():
            continue
        if val not in valid_set:
            item["out_of_schema"] = True
        out.append(item)
    return out


def normalize_pass2(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return _empty_meta("parse_error")

    # Empty dict = Nemotron produced no parseable output (GPU OOM / garbage output)
    # Tag as silent_failure so --reprocess can target these specifically
    if not meta or (len(meta) == 1 and "_parsed_title" in meta):
        return _empty_meta("silent_failure")

    # ── is_relevant ──────────────────────────────────────────────────────────
    if "is_relevant" not in meta:
        score = meta.get("relevance_score", 0)
        meta["is_relevant"] = bool(isinstance(score, (int, float)) and score > 0)
    meta["is_relevant"] = bool(meta["is_relevant"])

    if not meta["is_relevant"]:
        summary = meta.get("summary", "rejected")
        norm = (summary or "").strip().lower()
        if norm in _TEMPLATE_LEAK_SUMMARIES or ("<" in norm and ">" in norm):
            return _empty_meta("template_leak")
        return _empty_meta(summary)

    # ── Array fields ──────────────────────────────────────────────────────────
    for key in ("species", "stages", "features", "modalities", "contexts", "domains"):
        if not isinstance(meta.get(key), list):
            meta[key] = []

    meta["species"]    = _fix_bat_category(meta["species"])
    meta["modalities"] = _validate_list_field(meta["modalities"], VALID_MODALITIES, "modality")
    meta["contexts"]   = _validate_list_field(meta["contexts"],   VALID_CONTEXTS,   "context")
    meta["domains"]    = _validate_list_field(meta["domains"],    VALID_DOMAINS,    "domain")
    meta["features"]   = _validate_list_field(meta["features"],   VALID_FEATURES,   "feature")
    meta["stages"]     = _validate_list_field(meta["stages"],     VALID_STAGES,     "stage")

    if not meta["modalities"]:
        parsed = meta.get("_parsed_title", "")
        meta["modalities"] = [{"modality": "Acoustic",
                               "evidence": parsed or "inferred from context"}]

    try:
        meta["relevance_score"] = max(1, min(10, int(meta.get("relevance_score", 5))))
    except (ValueError, TypeError):
        meta["relevance_score"] = 5

    ss = meta.get("social_scale", "not_applicable")
    meta["social_scale"] = ss if ss in VALID_SOCIAL_SCALES else "not_applicable"

    rt = meta.get("research_tradition", "biological")
    meta["research_tradition"] = rt if rt in VALID_TRADITIONS else "biological"

    pt = meta.get("paper_type", "empirical")
    meta["paper_type"] = pt if pt in VALID_PAPER_TYPES else "empirical"

    meta["has_dataset"]   = bool(meta.get("has_dataset",   False))
    meta["has_code"]      = bool(meta.get("has_code",      False))
    meta["has_benchmark"] = bool(meta.get("has_benchmark", False))

    if not isinstance(meta.get("summary"), str):
        meta["summary"] = ""

    return meta


def _empty_meta(reason: str = "") -> dict:
    return {
        "is_relevant":        False,
        "species":            [],
        "stages":             [],
        "features":           [],
        "modalities":         [],
        "contexts":           [],
        "domains":            [],
        "social_scale":       "not_applicable",
        "research_tradition": "biological",
        "paper_type":         "irrelevant",
        "has_dataset":        False,
        "has_code":           False,
        "has_benchmark":      False,
        "relevance_score":    0,
        "summary":            reason,
    }


# ============================================================
# STAGE 2 — NEMOTRON EXTRACTION
# ============================================================

def run_stage_2(output_dir: str, gpu_util: float, reprocess: bool, reprocess_all_rejected: bool,
                 input_dir: str = None):
    # input_dir defaults to output_dir (legacy behavior: read+write the same
    # directory). Pass a different input_dir to read pass1_bert.jsonl from a
    # directory whose classify.sqlite/final_relevant.jsonl belong to a
    # different Stage-2 model run (e.g. reading BERT's accumulated output
    # while writing Nemotron's own fresh DB, so old model's done-markers
    # don't cause this run to skip papers it hasn't actually judged itself).
    input_dir   = input_dir or output_dir
    input_file  = os.path.join(input_dir, "pass1_bert.jsonl")
    output_file = os.path.join(output_dir, "final_relevant.jsonl")
    db_path     = os.path.join(output_dir, "classify.sqlite")

    log("=" * 65)
    log("STAGE 2: Nemotron strict verification + metadata extraction")
    log(f"Input  : {input_file}")
    log(f"Output : {output_file}")
    if reprocess_all_rejected:
        mode_str = "--reprocess-all-rejected (re-run ALL rejected + never-processed)"
    elif reprocess:
        mode_str = "--reprocess (silent failures + never-processed only)"
    else:
        mode_str = "normal (skip all already-processed papers)"
    log(f"Mode   : {mode_str}")
    log("=" * 65)

    if not os.path.exists(input_file):
        log(f"ERROR: {input_file} not found.")
        return

    db = ClassifyDB(db_path)
    migrate_legacy(db, output_dir)

    # ── Build the set of IDs to FORCE reprocess ───────────────────────────────
    # reprocess_ids: papers we must remove from the DB "done" set so the main
    # loop re-queues them instead of skipping them.
    reprocess_ids: set = set()
    if reprocess_all_rejected:
        # Re-run EVERY paper that is not already relevant — this is the full
        # "give everything a second chance" mode. Pulls all is_relevant=0 IDs.
        reprocess_ids = db.get_all_rejected()
        log(f"  --reprocess-all-rejected: {len(reprocess_ids):,} rejected papers un-marked for re-run")
    elif reprocess:
        # Targeted mode: only silent failures (Nemotron produced no real output)
        reprocess_ids = db.get_silent_failures()
        log(f"  --reprocess: {len(reprocess_ids):,} silent failures un-marked for re-run")

    # Remove reprocess_ids from in-memory done-set so main loop re-queues them.
    # Never-processed papers (not in DB at all) need no special handling —
    # they are absent from _s2 already and will be queued automatically.
    for oid in reprocess_ids:
        db._s2.pop(oid, None)

    writer   = OutputWriter(output_file)
    existing = writer.count_existing()
    log(f"Existing final_relevant.jsonl lines: {existing:,}")

    log("Counting input lines...")
    total = sum(1 for _ in open(input_file, encoding="utf-8"))
    log(f"Input: {total:,}  |  DB done: {db.s2_count():,}  |  "
        f"New/reprocess: {total - db.s2_count():,}")

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    global _TOKENIZER
    # fix_mistral_regex=True: this tokenizer is Mistral-derived and warns that
    # its default regex mis-tokenizes without this flag (see HF discussion
    # mistralai/Mistral-Small-3.1-24B-Instruct-2503#84).
    _TOKENIZER = AutoTokenizer.from_pretrained(
        NEMOTRON_MODEL, trust_remote_code=True, fix_mistral_regex=True,
    )

    num_gpus = torch.cuda.device_count()
    log(f"Loading Nemotron on {num_gpus} GPU(s): {NEMOTRON_MODEL}")
    llm = LLM(
        model=NEMOTRON_MODEL,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        # Model supports up to 131,072 (config.json max_sequence_length); 8192
        # was our own conservative choice and was too tight — PASS2_SYSTEM
        # alone is ~5.2k tokens, leaving almost no room for TOPICS/CONCEPTS/
        # KEYWORDS on papers with many OpenAlex tags, causing hard truncation
        # (and, before the truncation guard existed, a batch-killing crash).
        # 16384 gives ~9.3k tokens of headroom for paper content while still
        # keeping KV cache size (fp8 + prefix caching) modest.
        max_model_len=16384,
        dtype="bfloat16",
        # PASS2_SYSTEM is a ~5.2k-token fixed prefix shared by every request
        # (only the per-paper title/abstract/topics tail differs), so caching
        # its KV once and reusing it avoids redundant prefill on nearly every
        # request — the single biggest inference-speed lever here.
        enable_prefix_caching=True,
        # fp8 KV cache: halves per-token KV memory, letting more requests run
        # concurrently at the same gpu_memory_utilization. Validated on this
        # exact model/GPU by a sibling project's vLLM tuning
        # (cat_video_caption/extract_video_context/produce_tag/serve_nemotron_text.sh).
        kv_cache_dtype="fp8",
    )
    # Nemotron's own EOS (<|im_end|>, per special_tokens_map.json) is applied
    # automatically by vLLM's SamplingParams; no manual stop list needed.
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.95, max_tokens=6144,
    )
    log("Model loaded")

    gpu_queue    = []
    processed    = 0
    relevant     = 0
    model_rej     = 0
    silent_fails = 0
    skipped      = 0
    batch_count  = 0
    t0           = time.time()

    def _run_gpu_queue():
        nonlocal relevant, model_rej, silent_fails, batch_count
        if not gpu_queue:
            return

        prompts = [make_pass2_prompt(p) for p in gpu_queue]
        outputs = llm.generate(prompts, sampling_params)

        for i, out in enumerate(outputs):
            p   = gpu_queue[i]
            oid = _get_oa_id(p)

            raw_meta = extract_json(out.outputs[0].text)
            if raw_meta is None:
                raw_meta = {}

            parsed = extract_paper_text(p)
            raw_meta["_parsed_title"] = parsed["title"]

            meta = normalize_pass2(raw_meta)
            meta.pop("_parsed_title", None)

            # Detect silent failure (Nemotron gave no real output)
            is_silent = (meta["summary"] == "silent_failure")
            if is_silent:
                silent_fails += 1

            # Write all fields to the paper record
            p["is_relevant"]        = meta["is_relevant"]
            p["species"]            = meta["species"]
            p["stages"]             = meta["stages"]
            p["features"]           = meta["features"]
            p["modalities"]         = meta["modalities"]
            p["contexts"]           = meta["contexts"]
            p["domains"]            = meta["domains"]
            p["social_scale"]       = meta["social_scale"]
            p["research_tradition"] = meta["research_tradition"]
            p["paper_type"]         = meta["paper_type"]
            p["has_dataset"]        = meta["has_dataset"]
            p["has_code"]           = meta["has_code"]
            p["has_benchmark"]      = meta["has_benchmark"]
            p["relevance_score"]    = meta["relevance_score"]
            p["ai_summary"]         = meta["summary"]
            p["ai_metadata"]        = meta

            p["classification"] = {
                "relevant":   meta["is_relevant"],
                "confidence": 1.0 if meta["is_relevant"] else 0.0,
                "reason":     meta["summary"],
                "method":     "bert+nemotron",
            }

            p_udate = (p.get("updated_date") or "").strip()
            if oid:
                if oid in reprocess_ids:
                    # Overwrite existing DB record (silent failure, forced
                    # reprocess, or a stale verdict superseded by newer data).
                    db.overwrite_s2(oid, meta["is_relevant"], meta["relevance_score"],
                                    meta["paper_type"], meta["summary"], p_udate)
                else:
                    db.mark_s2(oid, meta["is_relevant"], meta["relevance_score"],
                               meta["paper_type"], meta["summary"], p_udate)

            if meta["is_relevant"]:
                writer.add(p)
                relevant += 1
            else:
                model_rej += 1

        batch_count += 1
        gpu_queue.clear()
        db.flush_s2()
        writer.flush()   # flush every batch — no gaps on crash

    # ── Main loop ─────────────────────────────────────────────────────────────
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except Exception:
                continue

            oid = _get_oa_id(p)
            paper_udate = (p.get("updated_date") or "").strip()
            if oid and db.s2_done(oid):
                # In reprocess mode, silent_failure IDs were removed from _s2
                # above, so they won't be skipped here — they'll be re-queued.
                # Staleness check: if pass1_bert.jsonl's row for this paper is
                # newer than what our stored verdict was judged from (OpenAlex
                # republished it — e.g. an abstract backfilled onto a stub),
                # force a re-judge instead of trusting the old verdict forever.
                if db.s2_stale(oid, paper_udate):
                    reprocess_ids.add(oid)
                    db._s2.pop(oid, None)
                else:
                    skipped   += 1
                    processed += 1
                    if processed % PRINT_EVERY == 0:
                        _print_s2(processed, total, relevant, model_rej,
                                   silent_fails, skipped, t0)
                    continue

            gpu_queue.append(p)
            processed += 1

            if len(gpu_queue) >= NEMOTRON_BATCH_SIZE:
                _run_gpu_queue()

            if processed % PRINT_EVERY == 0:
                _print_s2(processed, total, relevant, model_rej,
                           silent_fails, skipped, t0)

    # ── Final flush ───────────────────────────────────────────────────────────
    _run_gpu_queue()
    db.flush_s2()

    # final_relevant.jsonl is append-only, so a paper reprocessed because its
    # source data went stale (see s2_stale) can now have more than one line
    # in it, and a paper that flips from relevant to rejected on reprocessing
    # leaves its old (now-wrong) line behind. Rebuild it against the DB: keep
    # only the LAST line per oa_id (freshest write) and only where the DB's
    # current verdict is still is_relevant=1.
    still_relevant = {
        r[0] for r in db.conn.execute(
            "SELECT oa_id FROM processed_s2 WHERE is_relevant=1"
        )
    }
    if os.path.exists(output_file):
        best_line: dict = {}
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = _get_oa_id(rec)
                if rid:
                    best_line[rid] = line  # last occurrence wins
        kept = [best_line[rid] for rid in best_line if rid in still_relevant]
        if len(kept) != sum(1 for _ in open(output_file, encoding="utf-8")):
            tmp = output_file + ".dedup_tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
            os.replace(tmp, output_file)
            log(f"  final_relevant.jsonl deduped/reconciled against DB: {len(kept):,} lines")

    db.close()

    log(f"\nStage 2 complete:")
    log(f"  Total input        : {processed:,}")
    log(f"  Skipped (in DB)    : {skipped:,}")
    log(f"  Verified relevant  : {relevant:,}")
    log(f"  Nemotron rejected  : {model_rej:,}")
    log(f"  Silent failures    : {silent_fails:,}  (will be re-run with --reprocess)")
    log(f"  Output             : {output_file}")
    log(f"\nDB stats:")
    log(f'  sqlite3 {db_path} "SELECT COUNT(*),AVG(relevance_score) FROM processed_s2 WHERE is_relevant=1;"')
    if silent_fails > 0:
        log(f"\n  WARNING: {silent_fails:,} papers produced empty/garbage Nemotron output.")
        log(f"  This usually means GPU memory pressure (model too large for available VRAM).")
        log(f"  Re-run with --reprocess to retry these papers.")
        log(f'  sqlite3 {db_path} "SELECT COUNT(*) FROM processed_s2 WHERE is_relevant=0 AND relevance_score=0 AND summary=\'silent_failure\';"')


def _print_s2(processed, total, relevant, model_rej, silent_fails, skipped, t0):
    elapsed = time.time() - t0
    rate    = processed / max(elapsed, 1)
    eta     = (total - processed) / max(rate, 0.001)
    pct     = processed / max(total, 1) * 100
    log(f"  {processed:>8,}/{total:,} ({pct:.1f}%)  "
        f"relevant={relevant:,}  model_rej={model_rej:,}  "
        f"silent_fail={silent_fails:,}  skipped={skipped:,}  "
        f"{rate:.0f}/s  ETA {_fmt(eta)}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="annotate_nemotron.py — Nemotron annotation stage (input: pass1_bert.jsonl)"
    )
    parser.add_argument("--output_dir",   type=str,   default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write final_relevant.jsonl and classify.sqlite to")
    parser.add_argument("--input_dir",    type=str,   default=None,
                        help=(
                            "Directory containing pass1_bert.jsonl to read from "
                            "(defaults to --output_dir). Pass a different directory here "
                            "when output_dir's classify.sqlite belongs to a different, "
                            "unrelated annotation run and should not cause this run "
                            "to skip papers as already-done."
                        ))
    parser.add_argument("--gpu_mem_util", type=float, default=0.80,
                        help="vLLM GPU memory utilization (default 0.80)")
    parser.add_argument("--model",        type=str,   default=None,
                        help="Override Nemotron model path")
    parser.add_argument("--reprocess",    action="store_true",
                        help=(
                            "Re-run papers that were never processed or silently failed. "
                            "Specifically targets: "
                            "(1) Papers in pass1_bert.jsonl but absent from the DB entirely "
                            "    (crash gap / GPU OOM during a previous run). "
                            "(2) Papers in DB with is_relevant=0, relevance_score=0, and "
                            "    summary in {silent_failure, parse_error, rejected, ''} — "
                            "    these are papers where Nemotron produced no real output. "
                            "Already-correctly-processed papers (both relevant and legitimately "
                            "rejected with a real Nemotron summary) are always skipped."
                        ))
    parser.add_argument("--reprocess-all-rejected", dest="reprocess_all_rejected",
                        action="store_true",
                        help=(
                            "Re-run EVERY paper that did not make it into final_relevant.jsonl. "
                            "Removes all is_relevant=0 DB records from the skip-set and re-queues "
                            "them for Nemotron, regardless of whether they were a silent failure or a "
                            "real rejection. Use this when the original run had GPU/quality issues "
                            "and you want a full second-pass on all rejected papers. "
                            "Papers already marked is_relevant=1 are always skipped. "
                            "NOTE: this re-runs ~34k-38k papers per shard — expect a full run time."
                        ))
    args = parser.parse_args()

    if args.model:
        global NEMOTRON_MODEL
        NEMOTRON_MODEL = args.model

    os.makedirs(args.output_dir, exist_ok=True)
    run_stage_2(args.output_dir, args.gpu_mem_util, args.reprocess, args.reprocess_all_rejected,
                input_dir=args.input_dir)


if __name__ == "__main__":
    main()
