# Step 40 — Asymmetric Retrieval Encoder

> **Status: tried, not adopted.** Generated full E5-large-v2 embeddings with `query: ` / `passage: ` prefixes, ran the full hybrid pipeline, compared against the MiniLM baseline. The asymmetric encoder collapses cosine spread instead of sharpening it, and the aggregate hybrid metrics regress. Implementation kept as plumbing only (no experiment-specific code added); E5 embeddings live untracked under `data/embeddings/e5_large/` and can be regenerated with one command.

## Setup

- **Model:** `intfloat/e5-large-v2` (1024-dim).
- **Prefixes:** programmes encoded with `"query: "`, jobs with `"passage: "` — the canonical E5-v2 asymmetric convention.
- **Pipeline parity:** kept Step 34's section-weighted programme embedding and the chunk-and-pool job embedding (chunk size 256 tokens). All hybrid hyperparameters identical to baseline (α=0.55, γ=0.3, IPF top-30 with two-tier floor, sqrt implicit weighting, programme-IDF on).
- **Skill extraction left on MiniLM** — Step 4's explicit/implicit extractors continue to use `all-MiniLM-L6-v2` for skill-vs-skill similarity. The asymmetric prefixes do not apply to skill matching.
- **Wall time:** ~67 s end-to-end on Apple Silicon for the full re-embed of 45 programmes + 520 jobs (well under the 30 min PROGRESS estimate).

## Headline numbers (hybrid ranking, 45 programmes × top-50 candidates)

| Metric | Baseline (MiniLM, 384-dim) | E5-large-v2 (1024-dim) | Δ |
|---|---:|---:|---:|
| Top-1 unique programmes | 40 / 45 | 39 / 45 | −1 |
| Head-tied (gap < 0.02) | 12 / 45 | 15 / 45 | +3 |
| Top-5 generalists (job freq > 5) | 1 | **6** | +5 |
| Top-5 max repeat | 6 | 7 | +1 |
| Top-1 score mean | 0.305 | 0.275 | −0.030 |
| Top-1 score max | 0.677 | 0.556 | −0.121 |
| Score CoV | 0.788 | 0.907 | +0.119 |
| Candidate cosine mean (Stage 1) | 0.513 | **0.847** | +0.334 |
| Per-programme cosine range (mean) | 0.160 | **0.027** | **−0.133** |

Top-1 churned for **40 / 45** programmes (89% of recommendations changed at #1).

## Diagnosis — cosine collapse

The single load-bearing observation: the per-programme cosine range collapses
from 0.16 to 0.027 — about **6× smaller spread**. After per-programme min-max
normalisation each programme's candidate scores get stretched to `[0, 1]`, but
the underlying signal in those 0.027 absolute differences is mostly noise.
The hybrid formula's confidence-damping is *relative* (range / median-range),
so when every programme sees a small range there is nothing to dampen against
— damping does not bite when the whole population is compressed.

The two-tier IPF popularity penalty also weakens: the cosine-driven Stage 1
retrieval no longer separates relevant from irrelevant jobs strongly enough,
so generalist jobs sneak into more top-30 lists and IPF cannot fully
compensate. That explains the **6× jump** in top-5 generalists (1 → 6).

In short: E5-large-v2 reports that programmes and jobs are all "kind of
similar" (mean cosine 0.85) instead of "specifically similar where it
matters". The asymmetric query/passage encoding does not concentrate the
similarity onto the right pairs — it inflates background similarity for
all pairs.

## Mechanism — why E5 mismatches the corpus

E5 was trained on web-style query/passage pairs (short query → web
document). Two corpus mismatches:

1. **Programmes are not short queries.** They are long structured documents
   (course lists, learning outcomes, specialisations, identity). Tagging
   them with `"query: "` tells the encoder to embed in query mode but does
   not change the fact that they look like documents. The query head of the
   encoder is calibrated for ~10–30 token queries, not ~600+ token
   curricula.
2. **Jobs are not natural retrieval passages either.** They are recruiter-
   written long-form descriptions with heavy boilerplate ("benefits we
   offer", "our company"), structurally distinct from the Wikipedia /
   Common Crawl passages E5 saw at training time.

Both sides are long, similar-style documents, so the asymmetric distinction
collapses. The model is reduced to a generic high-dimensional sentence
encoder that puts everything in a tight cone of cosine 0.83–0.87.

## Qualitative inspection of the 40 changed top-1s

A handful of changes look like real wins on niche programmes:

- **AI MSc** → "DIRBTINIO INTELEKTO INŽINIERĖ (AI Engineer)" instead of
  "MLOps engineer" — clear domain hit.
- **Cyber Systems and Security** → "Vyresnysis specialistas (kibernetinio
  saugumo)" (Senior Cybersecurity Specialist) instead of "PROGRAMUOTOJAS
  (PASTATŲ VALDYM…)" (building automation programmer) — clear win on a
  programme that previously matched a weak signal.
- **Computer games and animation** → "Gameplay Programmer" instead of
  "Game Designer" — both reasonable.

But the aggregate metrics dominate: most other changes spread picks toward
generalist analyst / data engineer roles, breaking diversity. Per-programme
top-1 quality is roughly even with baseline by inspection (~10 better /
~10 wash / ~20 worse), and the diversity, head-tie, and generalist
regressions are unambiguous.

## Decision

E5-large-v2 not adopted. Default rankings stay at the existing hybrid
(α=0.55, MiniLM-L6-v2, no cross-encoder, no LTR).

Combined with Step 25 (larger symmetric encoder MPNet showed no improvement
because of token truncation) this is the **second negative result on the
embedding side**. Throwing bigger / better-trained sentence encoders at
this corpus does not move the needle. The semantic side is not the
bottleneck — it provides reasonable Stage 1 recall, and there is little
discriminative power to extract above what MiniLM already gives.

## Implications for next steps

The cosine collapse and the previous Step 38 / 39 findings (specificity
asymmetry, generalist bias in consensus labels) consistently point at the
**symbolic alignment formula** as the next leverage point, not the encoder.

Promising directions queued for the next iteration:

- **Mutual specificity** — restrict the symbolic side to high-IDF skills
  before computing programme_recall. Long generic jobs lose their
  advantage because their *specific* skills (if any) do not overlap with
  programme cores.
- **Add `programme_precision`** — fraction of the *programme's* high-IDF
  skills the job demands. Currently we only measure programme_recall (in
  the wrong direction for niche programmes). An F1-style symmetric score
  would balance both directions.
- **Reciprocal Rank Fusion** instead of per-programme min-max normalisation
  for the cosine + symbolic blend — rank-based, scale-free, robust to the
  exact pathology this experiment exposed.
- **Adaptive α** as a function of programme skill richness — skill-rich
  programmes lean symbolic, sparse programmes lean semantic.

These are formula changes, not model changes. They are cheap to try,
testable offline, and target the diagnosed failure mode directly.

## What is left on disk

- `data/embeddings/e5_large/` — generated embeddings + dataset snapshot
  (untracked, regeneratable in ~70 s with one command).
- `experiments/results/evaluation/asymmetric_encoder/summary.json` — full
  metric table and the 40-row top-1 churn list.

This document is the experiment record. Regenerate with:

```bash
.venv/bin/python -m src.embeddings.generator \
    --model intfloat/e5-large-v2 \
    --programme-prefix "query: " \
    --job-prefix "passage: " \
    --suffix e5_large
```

then build the dataset against `data/embeddings/e5_large/`, run
`align_hybrid`, and diff against the baseline.
