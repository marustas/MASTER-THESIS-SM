# Step 45 — Title-aware re-ranking · FINDINGS

## TL;DR

**Mixed — produces the marquee Cyber Systems → SOC analyst fix at
title_alpha = 0.20, but at aggregate cost (−2 unique top-1, +4 top-5
generalists, +8 head-tied programmes).**  Useful for thesis story
("here is the algorithmic fix for the documented Cyber misfire"); not
adoptable as a canonical default.  Lower alphas preserve aggregate
metrics but don't surface the marquee fixes either.

## Setup

- Add a third blend channel to the hybrid formula:
  `hybrid = α · cos_norm + title_alpha · title_norm
           + (1 − α − title_alpha) · recall_norm`
  where `title_norm = per-programme min-max of cosine(name, job_title)`.
- Title encoder: `all-MiniLM-L6-v2` (same model as body embeddings),
  with titles **pre-translated** to English via Google Translate (cached
  to `data/dataset/title_translations.parquet`).  Multilingual encoders
  would handle Lithuanian natively but couldn't be downloaded in this
  environment; the translation-then-encode approach matches the body
  embedding pipeline and is reproducible.
- Sweep: `title_alpha ∈ {0.00, 0.05, 0.10, 0.15, 0.20}`.
- All other hybrid params at canonical defaults (α=0.55, γ=0.3, IPF
  top-30 two-tier, sqrt implicit, programme IDF, Steps 41 + 42 blends).
- Corpus: 45 programmes × 520 job ads.

## Results

| title_alpha | unique top-1 | top-5 generalists | gap<0.02 | top-1 max | suspects changed |
|---:|---:|---:|---:|---:|---:|
| 0.00 (baseline) | **39** | **0** | 8 | 0.677 | – |
| 0.05 | 39 (0) | 0 (0) | 13 (+5) | 0.664 | 2 / 16 |
| 0.10 | 37 (−2) | 1 (+1) | 16 (+8) | 0.652 | 4 / 16 |
| 0.15 | 38 (−1) | 1 (+1) | 14 (+6) | 0.639 | 7 / 16 |
| 0.20 | 37 (−2) | **4 (+4)** | 16 (+8) | 0.626 | 5 / 16 |

The pattern: every step in title_alpha trades aggregate diversity for
specific suspect movement.  No Pareto-positive operating point.

## The marquee fix — at title_alpha = 0.20

The Cyber Systems and Security failure is the exact misfire the
experiment targeted, and it is fixed cleanly:

```
Cyber Systems and Security @ Kauno kolegija
  baseline → title_alpha=0.20
  "PROGRAMMER (BUILDING MANAGEMENT)" → "SOC analyst for cyber threats"
```

Title-vs-title cosine for the SOC role is much higher than for the
building-management programmer role (which shares only generic IT
vocabulary), so the title channel demotes the wrong match.  This is the
mechanism the experiment was designed to exploit.

Other documented suspect changes at title_alpha = 0.20:

| programme | base → ta=0.20 | verdict |
|---|---|---|
| **Cyber Systems and Security** | BMS programmer → **SOC analyst** | **clear win — marquee fix** |
| Information Systems @ KTU | IT systems tester → Junior System Analyst | lateral |
| Multimedia Tech @ Šiaulių | Junior social media designer → IT Technikas | regression |
| Media Tech @ VBC | Senior IT PM → Junior Social Media Designer | possible win |
| Informatics @ Klaipėdos | BI programmer → MS Power Developer | lateral |

The persistent high-profile failures all *remain*:

- IS Engineering @ Vilniaus kolegija → **still** cleaning-business automation manager
- Informatics @ KTU / Klaipėda / Vilnius University → **still** admin / supporter
- Software Engineering @ Vilnius University → **still** Low-code programuotojas
- Multimedia Tech @ KTU → **still** Gameplay Programmer

The marquee Cyber fix is paid for with the aggregate regressions.

## Why aggregate metrics regress

The title channel adds **noise** for the majority of programmes whose
body-based ranking was already good.  Job titles are short (~30-60
chars) — far less informative than 6000-char curricula or 5000-char job
bodies.  When the title cosine disagrees with the body cosine, the
min-max normalisation amplifies the small title signal into the blend
and can flip rank order without strong evidence.  Head-tied count
(+5 to +8) is the direct symptom — many programmes now have top-1 and
top-2 separated by only the title-channel noise.

## Why some misfires *don't* get fixed by titles

- **IS Engineering @ Vilniaus kolegija → cleaning-business automation
  manager (Robotizacijos ir automatizavimo vadovas)**: the job title
  literally contains "Robotizacijos" and "automatizavimo" — vocabulary
  the IS Engineering curriculum *also* uses (process automation,
  systems integration).  The title channel doesn't help when both
  sides share buzzwords.
- **Informatics → IT admin / supporter**: "Informatics" is a vague
  programme title that doesn't strongly disagree with generic IT job
  titles.  No title-side signal to exploit.

In short — title-aware re-ranking helps when (and only when) the
programme title strongly disagrees with the wrong-match job title.
That's a meaningful subset of failures (Cyber being the clearest
example) but not all of them.

## Suspect change progression

| title_alpha | suspects changed | clear wins | losses |
|---:|---:|---:|---:|
| 0.05 | 2 | 0 | 1 (Multimedia → IT Technikas) |
| 0.10 | 4 | 1 (SE → .NET, lateral) | 1 |
| 0.15 | 7 | mixed | mixed |
| 0.20 | 5 | **1 (Cyber → SOC)** | 1 (Multimedia → IT Technikas) |

The Cyber → SOC fix only emerges at ta=0.20 — anything lower preserves
the BMS programmer misfire.

## Recommendation

**Do not auto-adopt.**  No alpha value is Pareto-positive.  But keep
the parameter available for two use cases:

1. **Thesis demonstration.**  Set `title_alpha=0.20` for a specific
   demo step that shows the Cyber → SOC fix in action — it's a clean
   example of why a multi-channel formula matters.
2. **Future surgical-mode work.**  The right way to use this signal
   would be a *conditional* title channel: apply only when
   `cos(title_p, title_j) > threshold` AND `cos(title_p, title_j) <
   median_title_cos_for_p`, i.e. only when titles strongly *disagree*
   for the current top candidate.  That's a Step 46 candidate if this
   direction is pursued further.

## What this rules out

A uniform title-channel blend cannot simultaneously fix the surgical
misfires and preserve aggregate ranking quality.  The signal is too
sparse — most pairs have neutral title cosine — for an unconditional
blend to be Pareto-positive.

## Artifacts

- `summary.json` — full metric table + deltas + suspect changes per α
- `rankings_t{0_00..0_20}.parquet` — full hybrid rankings per α
- `data/dataset/title_translations.parquet` — cached title translations
  (Lithuanian → English), keyed by row index + field

## Combined Step 43 + 44 + 45 outcome

Three independent algorithmic fixes attempted, none Pareto-positive at
the headline level.  All three either don't move the documented
misfires (43) or move some at non-trivial cost (44, 45).  The honest
read is that the residual failures are not formula-fixable in the
current corpus — the bottleneck is **vocabulary collision between the
programme body text and adjacent-domain job ads**, and that collision
exists at the body level, the cluster level, and the title level
simultaneously.

The right path forward is either (a) better source data (curriculum
enrichment, broader job corpus), (b) external occupation labels (ESCO
occupation matching as Step 46), or (c) honest disclosure (the UI
confidence cue already adopted).
