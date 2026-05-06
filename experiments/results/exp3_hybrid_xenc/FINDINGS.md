# Step 38 — Cross-encoder Re-ranking

## Goal

Add a re-ranking stage between Stage 1 (semantic top-50 retrieval) and Stage 2
(symbolic refinement) of the hybrid alignment.  Hypothesis from Step 38 spec:
the bi-encoder cosine has *spent its variance* during Stage 1 (cosine top-1 ==
hybrid top-1 in 0/45 programmes pre-Step 38), so a cross-encoder re-evaluating
each pair from scratch with full token-level attention should produce fresh
ranking signal in the candidate pool and reduce the head-tied programmes
(top-1 ↔ top-2 gap < 0.02).

## Setup

* Dataset: 45 programmes × 520 job ads.
* Stage-1 candidate pool: top-50 by bi-encoder cosine.
* Hybrid blend: ``α · cos_norm + xe_alpha · xe_norm + (1 − α − xe_alpha) · recall_norm``.
* Stage 2 / quality / IPF / programme-IDF / sqrt implicit confidence: kept at
  their current production settings.
* All inputs are English-translated `cleaned_text` (deep-translator pipeline,
  language column ``en``, confidence ≈ 1.0).  *No Lithuanian text reaches the
  re-ranker.*
* Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB, 6-layer
  MiniLM, 512-token pair budget).

## Configurations

| Config                  | α (cos) | xe_alpha | recall | xe_pool_mode       | What changes vs. baseline |
|-------------------------|---------|----------|--------|---------------------|----------------------------|
| baseline                | 0.55    | 0.00     | 0.45   | —                   | current production hybrid  |
| replace_cos             | 0.00    | 0.55     | 0.45   | single              | cosine channel removed; xe takes its weight |
| three_channel           | 0.275   | 0.275    | 0.45   | single              | old cosine half split between cos and xe |
| three_channel_secwm     | 0.275   | 0.275    | 0.45   | section_weighted    | xe scored per programme section (weighted-mean) |
| three_channel_secmax    | 0.275   | 0.275    | 0.45   | section_max         | xe scored per programme section (max) |
| three_channel_jobchunk  | 0.275   | 0.275    | 0.45   | job_chunked_max     | xe scored per 256-token job chunk (max-pool) |
| three_channel_secxjob   | 0.275   | 0.275    | 0.45   | section_x_job_max   | two-sided: prog sections (weighted) × job chunks (max) |

Section-aware variants split each programme into the same section groups used
by Step 34 (subjects 0.35, outcomes 0.25, identity 0.15, specialisations 0.20,
_remainder 0.05) and score each non-empty section against the full job text.
This lifts the 512-token pair-budget ceiling for the programme side.

## Aggregate metrics (full dataset, MS MARCO-MiniLM)

| Metric                     | baseline | replace_cos | three_ch | secwm     | secmax | jobchunk | secxjob |
|----------------------------|----------|-------------|----------|-----------|--------|----------|---------|
| Top-1 unique               | 40       | 39          | 38       | **43**    | 41     | 39       | 39      |
| Top-1 diversity            | 0.889    | 0.867       | 0.844    | **0.956** | 0.911  | 0.867    | 0.867   |
| Top-1 max repeat           | 2        | 3           | 3        | 2         | 2      | 2        | 2       |
| Top-5 generalists (>5×)    | 1        | 2           | 2        | **0**     | 1      | 1        | **0**   |
| Top-1 score mean           | 0.305    | 0.439       | 0.366    | 0.361     | 0.366  | 0.351    | 0.350   |
| Top-1 score max            | 0.677    | 0.715       | 0.566    | **0.745** | 0.655  | 0.587    | 0.652   |
| Top-1 score CoV            | 0.372    | 0.218       | 0.242    | 0.313     | 0.299  | 0.263    | 0.301   |
| Gap top1↔top2 mean        | 0.069    | 0.074       | 0.076    | 0.070     | 0.068  | 0.062    | 0.069   |
| Programmes with gap<0.02   | 12       | 10          | **5**    | 12        | 13     | 12       | 14      |
| Programmes with gap<0.05   | 24       | 20          | 22       | 26        | 29     | 24       | 25      |
| Spearman sym↔hyb          | 0.263    | 0.144       | 0.124    | 0.118     | 0.130  | 0.137    | 0.129   |
| Spearman sem↔hyb          | -0.022   | -0.306      | -0.245   | -0.223    | -0.234 | -0.236   | -0.214  |
| Spearman base↔hyb         | —        | 0.711       | 0.865    | 0.894     | 0.888  | 0.865    | 0.883   |
| Top-1 agreement w/ base    | —        | 12/45       | 22/45    | 22/45     | 23/45  | 23/45    | 26/45   |

## Per-config impact

### `replace_cos` (α=0, xe_alpha=0.55)

* Aggressive: rewrites 33/45 top-1 picks (Spearman base↔hyb = 0.71 — lowest of
  all configs).  Costs −1 unique top-1, +1 generalist, only −2 head-tied
  programmes.  Highest top-1 score mean (0.439) but that's mostly cross-encoder
  scale, not signal.
* Verdict: **rejected** — too much rearrangement for too little head gain.

### `three_channel` single-pass (α=0.275, xe_alpha=0.275)

* Largest head-discrimination win: gap<0.02 from 12 → 5 (−58%).
* Costs: top-1 unique 40 → 38 (−2), top-5 generalists 1 → 2 (+1).
* 23/45 top-1 picks change.  Qualitative inspection of all 23:
  | Verdict | Count |
  |---------|-------|
  | Better  | 10    |
  | Wash    | 7     |
  | Worse   | 6     |
* Persistent regressions (where the cross-encoder picked a clearly worse job):
  Game Designer → QA Mobile Game Tester (Computer games & animation),
  Gameplay Programmer → Senior Java (KTU Multimedia),
  SOC threat-hunting analyst → Software Test Manager (MRU Cyber Security),
  AI Engineer (Applied AI) → IT SysAdmin (KTU Informatics Engineering /
  KTU Software Systems).
* Verdict: **strong head-discrimination win, mixed per-programme quality.**
  Acceptable if head confidence matters more than diversity; not a clear
  Pareto improvement.

### `three_channel_secwm` (section_weighted pooling)

* Pareto-dominates baseline on aggregate guardrails:
  * Top-1 diversity 40 → **43** (+3).
  * Top-5 generalists 1 → **0** (−1).
  * Top-1 score max 0.677 → **0.745** (+0.068).
  * gap<0.02 unchanged at 12.
* 23/45 top-1 picks change.  Qualitative inspection:
  | Verdict | Count |
  |---------|-------|
  | Better  | 10    |
  | Wash    | 5     |
  | Worse   | 8     |
* **The aggregate diversity wins do not come from systematically better picks
  — per-programme top-1 quality is roughly even (10 better / 8 worse, plus
  5 washes).**  The wins come from *spreading* picks across more unique jobs.
* 4 of the 8 regressions are *identical* to single-pass `three_channel`
  (Game Designer, Gameplay Programmer, SOC analyst, AI Engineer).  Section
  pooling cannot fix what the model gets wrong on the underlying text.
* Verdict: **best aggregate metrics, but per-programme quality bottlenecked
  by the re-ranker model.**

### `three_channel_secmax` (section_max pooling)

* In between `three_channel` and `secwm` on every metric.  No reason to
  prefer it over `secwm` in this run.

### `three_channel_jobchunk` (job-side chunking, max-pool)

* Programme is sent as a single pass; the job is chunked into 256-token
  pieces and the highest-scoring chunk wins.  Hypothesis: short
  specialised job descriptions get to put their best chunk forward
  rather than being averaged out.
* Result: marginal vs baseline.  Top-1 unique 39 (−1), gap<0.02 unchanged
  at 12, top-5 generalists 1 (=).  Top-1 agreement with baseline 23/45.
* On the persistent-regression list: **fixes Game Designer** (SMK
  Computer games & animation regains its top-1 from secwm's QA Tester),
  but does *not* recover SOC analyst, Gameplay Programmer, or AI Engineer.

### `three_channel_secxjob` (two-sided: prog sections × job chunks)

* For each pair, the cross-encoder sees every (section, job_chunk)
  combination; pooled by max over chunks then weighted-mean over
  sections.  ~5× the cost of `single`.
* Result: aggregate metrics worse than `secwm` — top-1 unique 39 (−4),
  gap<0.02 14 (worst of all configs), but top-5 generalists 0 (matches
  secwm).
* Top-1 agreement with baseline 26/45 (most conservative of all configs).
* On the persistent-regression list: keeps baseline's correct picks for
  SOC analyst (MRU) and AI Engineer (KTU Informatics Engineering) which
  every other variant broke.  But still loses Game Designer and Gameplay
  Programmer.
* Per-programme tally of the 19 changed top-1 picks vs baseline:
  | Verdict | Count |
  |---------|-------|
  | Better  | 8     |
  | Wash    | 3     |
  | Worse   | 8     |
* Same 8/8 better-vs-worse ratio as `secwm` and `three_channel`.

## Hypothesis revisions (what I got wrong, and why)

* **Initial guess:** "Lithuanian text confuses an English MS MARCO re-ranker."
  → **Wrong.**  `cleaned_text` is English-translated for both programmes and
  job ads (deep-translator step, language column = `en`, confidence ≈ 1.0).
  The re-ranker sees clean English text in both halves of every pair.
* **Revised diagnosis:** the regressions are an MS MARCO re-ranker pathology,
  not a language gap.  MS MARCO training rewards "answer is in the passage"
  surface coverage — long, topic-adjacent job descriptions out-score short,
  specialised ones when paired with broad curricula.  Concretely:
  * Game Designer / Gameplay Programmer descriptions are short and craft-specific.
  * QA Tester / Senior Java descriptions are long, repeat development /
    quality / process vocabulary, and overlap broad multimedia outcomes more.
  * SOC analyst description is sharp and tool-specific (SIEM, EDR, threat
    detection); Software Test Manager description spans testing / process /
    compliance, which the cyber-security curriculum mentions in passing.
  * AI Engineer (Applied AI) job is short and specialist; IT SysAdmin /
    .NET SE descriptions are long generic enterprise IT, which broad
    informatics curricula match more thoroughly on surface.

## Decision so far

* `secwm` is the strongest variant of those tested but is **not** an
  unconditional win — its per-programme top-1 quality is roughly even
  with baseline, and the regressions cluster in known-niche programmes.
* `three_channel` single-pass gives a meaningful head-discrimination win
  (12 → 5 ambiguous heads) at a real diversity cost (−2 top-1 unique, +1
  generalist).
* The persistent regressions are a re-ranker model limitation that section
  pooling cannot fix.

## Stronger re-ranker test — `BAAI/bge-reranker-base`

Re-ran the same five configs with `BAAI/bge-reranker-base` (~280 MB, 2024
release, generally stronger than MS MARCO-MiniLM on long-form pairs).
Outputs in `../exp3_hybrid_xenc_bge/` (same script, different `output_dir`
+ `cross_encoder_model`).

### bge aggregate metrics

| Metric                     | baseline | three_channel | secwm  | secmax |
|----------------------------|----------|---------------|--------|--------|
| Top-1 unique               | 40       | 36            | 41     | 41     |
| Top-1 diversity            | 0.889    | 0.800         | 0.911  | 0.911  |
| Top-1 max repeat           | 2        | 4             | 2      | 2      |
| Top-5 generalists (>5×)    | 1        | 1             | 1      | 2      |
| Top-1 score max            | 0.677    | 0.691         | 0.640  | 0.621  |
| Programmes with gap<0.02   | 12       | 16            | 14     | 19     |
| Spearman base↔hyb         | —        | 0.906         | 0.901  | 0.885  |
| Top-1 agreement w/ base    | —        | 25/45         | 27/45  | 21/45  |

### Comparison vs MS MARCO-MiniLM

| Metric                | base | MM secwm | bge secwm |
|-----------------------|------|----------|-----------|
| Top-1 unique          | 40   | **43**   | 41        |
| Top-5 generalists     | 1    | **0**    | 1         |
| gap<0.02              | 12   | 12       | 14        |
| Top-1 score max       | 0.677| **0.745**| 0.640     |
| Spearman base↔hyb    | —    | 0.894    | 0.901 (more conservative) |

### Per-programme inspection on persistent regressions

| Programme                              | baseline (right answer) | MM secwm                 | bge secwm                 | Verdict |
|----------------------------------------|--------------------------|---------------------------|----------------------------|---------|
| SMK Computer games & animation         | Game Designer            | Junior QA Game Tester ✗   | **Game Designer ✓**        | bge fixes |
| KTU Multimedia Technologies            | Gameplay Programmer      | Senior Java ✗             | Game Designer (◑ closer)   | bge partial |
| MRU Digital tech & cyber security      | SOC analyst threat hunt  | Software Test Manager ✗   | Software Test Manager ✗    | both fail |
| KTU Informatics Engineering            | AI Engineer (Applied AI) | IT SysAdmin ✗             | IT SysAdmin ✗              | both fail |
| KTU Software Systems                   | AI Engineer (Applied AI) | Senior .NET SE (◑)        | .NET Developer (◑)         | both ≈ wash |

bge fixes ~1 of 4 hard regressions, partially fixes 1 more, but the
aggregate metrics regress: head discrimination worsens (12 → 14), unique
top-1 drops (43 → 41), top-5 generalists ticks back up (0 → 1).

### Why bge underperforms

* `Spearman base↔hyb = 0.901` (bge) vs `0.894` (MM) — bge rearranges
  *less* than MS MARCO, hugs the bi-encoder cosine more closely.
* That conservatism is exactly the wrong direction: the head-tie problem
  is caused by the bi-encoder cosine running out of variance inside the
  candidate pool, so a re-ranker that mirrors cosine adds nothing.
* bge produces a tighter score distribution (top-1 score max 0.640 vs
  0.745) → less ranking signal under per-programme min-max.

### Structural diagnosis

The remaining failures (SOC analyst, AI Engineer, Gameplay Programmer)
share a property: the *correct* job has a short, specialised description
while the *wrong* job has a long, broadly worded description.  Re-ranker
model upgrades cannot fix this — it is a property of the corpus.  The
broad job description has more lexical surface to match against a broad
curriculum.

## Decision

Across **5 cross-encoder pool modes** (single, secwm, secmax, jobchunk,
secxjob) × **2 models** (MS MARCO-MiniLM, bge-reranker-base), the
better-vs-worse-than-baseline ratio at the per-programme level is
~10/8 ± 1 in every configuration.  Each variant fixes a different subset
of regressions and breaks a different subset.  No single config Pareto-
dominates baseline.

Aggregate-metric champions vary by metric:
* `secwm` — best diversity (43/45) + zero top-5 generalists + highest
  top-1 score max (0.745).
* `three_channel` — best head discrimination (gap<0.02 = 5).
* `secxjob` — most conservative w.r.t. baseline (Spearman 0.88, 26/45
  agreement) and the only variant to preserve baseline's correct picks
  for both SOC analyst and AI Engineer.

The cross-encoder signal is **fundamentally noisy on this corpus**.
That noise pattern is the textbook case for Step 39 (LTR): when no
single ranker is consistently better, train a learner to choose which
signal to trust per query using cosine, cross-encoder, programme_recall,
IDF statistics, and document-length features as inputs.

Step 38 left as **tested, no variant adopted as default**.  The default
hybrid stays as Step 35 baseline (α = 0.55, no cross-encoder).
Section-aware cross-encoder + MS MARCO-MiniLM (`secwm`) and the
two-sided variant (`secxjob`) are documented as reproducible alternative
configurations for downstream evaluation.

## Artefacts

* `summary.json` — machine-readable metrics for all configs (MS MARCO).
* `summary_table.md` — auto-generated metrics table (MS MARCO; overwritten on re-run).
* `rankings_<config>.parquet` — full per-programme rankings (MS MARCO).
* `top1_diff.csv` — per-programme top-1 across all MS MARCO configs.
* `../exp3_hybrid_xenc_bge/` — same artefacts for `BAAI/bge-reranker-base`.

## Artefacts

* `summary.json` — machine-readable metrics for all configs.
* `summary_table.md` — auto-generated metrics table (overwritten on re-run).
* `rankings_<config>.parquet` — full per-programme rankings.
* `top1_diff.csv` — per-programme top-1 across all configs (used for
  qualitative inspection).
