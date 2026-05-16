# Step 43 — Adaptive alpha · FINDINGS

## TL;DR

**Not adopted.** Adaptive α with an absolute corpus-median-IDF generalist
metric degenerates into a global α reduction because nearly every programme
scores as >0.8 generalist by this metric. Aggregate diversity regresses by
3 unique top-1 matches across all positive decay values; the targeted
generalist failures from the domain-expert review remained in place; the
worst absolute misfires (Cyber Systems → building-management programmer,
IS Engineering @ Vilniaus kolegija → cleaning-business automation manager)
became *more* confident at lower α rather than being demoted.

## Setup

- Hybrid formula unchanged. Only modulation: replace global α with
  `α_p = max(α_floor, α − decay · generalist_score(p))`.
- `generalist_score(p) = 1 − (weight on URIs with corpus IDF ≥ median) / total weight`,
  range [0, 1]. Specialist → 0, generalist → 1.
- Sweep: `decay ∈ {0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}`.
- Base α = 0.55, floor = 0.35.
- Corpus: 45 programmes × 520 job ads. Median URI IDF = 4.27.

## Results

| decay | unique top-1 | top-5 generalists | gap<0.02 | top-1 mean | top-1 max | mean α | n at floor |
|------:|------:|------:|------:|------:|------:|------:|------:|
| 0.00 (baseline) | **39** | **0** | 8 | 0.308 | 0.677 | 0.550 | – |
| 0.10 | 38 (−1) | 1 (+1) | 9 (+1) | 0.339 | 0.675 | 0.462 | 0 |
| 0.15 | 36 (−3) | 1 (+1) | 6 (−2) | 0.357 | 0.674 | 0.418 | 0 |
| 0.20 | 36 (−3) | 2 (+2) | 7 (−1) | 0.372 | 0.683 | 0.374 | 4 |
| 0.25 | 36 (−3) | 2 (+2) | 6 (−2) | 0.381 | **0.701** | 0.354 | 38 |
| 0.30 | 36 (−3) | 1 (+1) | 6 (−2) | 0.382 | 0.701 | 0.352 | 43 |
| 0.40 | 36 (−3) | 1 (+1) | 6 (−2) | 0.382 | 0.701 | 0.351 | 44 |

## Diagnosis — the metric saturates

By decay=0.25, **38 of 45 programmes hit the α floor**. By decay=0.40,
44 of 45 do. The "adaptive" α is therefore not adaptive — it is a global
α reduction with a tiny exception for one or two niche programmes.

Median URI IDF = 4.27 in our 565-document corpus. Only ~50% of URIs by
count clear that bar (by definition — it's the median), but they tend to
be the rarer ones that appear in fewer programmes. Most programmes'
weighted-skill mass sits on the *frequent* URIs — basic IT vocabulary
that everybody has. So the generalist score saturates near 1.0 for
almost everyone.

That means the metric does **not** distinguish "Informatics" (genuinely
broad — covers everything a bit) from "Cybersecurity Technologies"
(genuinely focused — narrow but rare skills). Both score >0.85 generalist
by this definition, and both get pushed to α=0.35.

## Suspect-programme tracking at decay=0.20

Of 16 known-failing generalist programmes, **only 3 changed top-1**:

| programme @ institution | baseline → decay=0.20 | verdict |
|---|---|---|
| IS Engineering @ Utena | IT On-site engineer → IT engineer | lateral |
| **Software Engineering @ Vilniaus kolegija** | Full-Stack Remote → **AI Engineer** | clear win |
| Multimedia Tech @ Šiaulių | Junior social media designer → IT Technikas | likely worse |

Net: 1 win, 1 wash, 1 loss across the targeted set. The big regressions
the experiment was designed to fix all *persisted*:

- Cyber Systems @ Kauno kolegija → **still** "Programmer (building management)" (score 0.26 → 0.45, more confident wrong)
- IS Engineering @ Vilniaus kolegija → **still** "Robotization manager (cleaning business)" (score 0.53 → 0.68, more confident wrong)
- Informatics @ all 4 institutions → **all** still admin / supporter / BI

The score inflation is the giveaway: lowering α amplifies the symbolic /
recall side, and these misfires *win* on shallow lexical/skill overlap
("system", "process", "integration" appear in both job and programme
text). The high-IDF blend (Steps 41 + 42) was supposed to anchor this on
rare skills, but for thin-curriculum programmes there are no rare skills
to anchor on — exactly the case where the generalist metric maxes out.

## What this rules out

- An α modulation driven by *the same IDF signal* the high-IDF blend
  already uses cannot fix programmes the high-IDF blend already failed
  to fix. Both depend on the programme having distinctive high-IDF
  skills; both fail when it doesn't.

## What to try instead

1. **Percentile-rank breadth** — rank programmes by generalist score and
   modulate α by rank percentile, guaranteeing real spread. Doesn't fix
   the underlying issue (thin curricula remain thin), but the modulation
   would at least be non-trivial.
2. **External structure (Step 44 — cluster prior)** — use the programme
   cluster × job cluster contingency from Step 18 as a prior on which
   jobs are appropriate for which programmes. Independent signal,
   doesn't rely on the same IDF source.
3. **Curriculum data improvement** — many failures are programmes whose
   description is thin (single paragraph, no subjects list). No
   formula change can synthesise specificity that was never extracted.
4. **Confidence cue in the UI** — for top-1 scores below a threshold,
   present "low confidence" instead of a bare number. Honest disclosure
   rather than algorithmic fix.

## Artifacts

- `summary.json` — full metric table + per-programme alpha distribution
  + suspect changes per decay
- `rankings_d{0_00..0_40}.parquet` — full hybrid rankings per decay setting

## Next step

Move to (2) above: Step 44, programme-cluster prior. The cluster
contingency table is already built (Step 18) and is structurally
independent of the IDF channel that adaptive α tried — and failed — to
reuse.
