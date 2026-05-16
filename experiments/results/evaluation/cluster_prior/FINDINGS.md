# Step 44 — Cluster prior · FINDINGS

## TL;DR

**Mild positive — adoption candidate at κ=0.10.**  Diversity goes up
(+3 unique top-1 matches) at the cost of more head-tied programmes (+4
with top-1↔top-2 gap < 0.02).  None of the *specific* generalist
misfires we targeted (Cyber → BMS programmer, IS-Eng → cleaning
automation, all 4 Informatics → admin / supporter) were fixed at any
sweep point, but the algorithm becomes more diverse and finds one clean
new win (Informatics @ Klaipėda: IT Administrator → Mid Software
Engineer .NET/Delphi at κ=0.30).  Decision: present the trade-off to
the user, do not auto-adopt.

## Setup

- Symbolic signal becomes:
  `recall_final = (1 − κ) · programme_recall + κ · cluster_recall(p, j)`
  where `cluster_recall = programme_recall(centroid(p's cluster), j)`
  and the centroid is the mean weighted-skill profile across all
  programmes in the cluster.
- Programme clustering: k-means, k=6, on programme embeddings (Step 7
  defaults, re-run to repopulate `cluster_label` on the dataset).
- Sweep: `κ ∈ {0.00, 0.10, 0.20, 0.30, 0.50, 0.80}`.
- All other hybrid params at canonical defaults
  (α=0.55, γ=0.3, IPF top-30 with two-tier floor, sqrt implicit,
  programme IDF, Step 41 λ=0.25, Step 42 μ=0.20).
- Corpus: 45 programmes × 520 job ads.

## Results

| κ | unique top-1 | top-5 generalists | gap<0.02 | top-1 max | suspect changes |
|---:|---:|---:|---:|---:|---:|
| 0.00 (baseline) | **39** | **0** | 8 | 0.677 | – |
| 0.10 | **42 (+3)** | 1 (+1) | 12 (+4) | 0.689 | 3 / 16 |
| 0.20 | 40 (+1) | 2 (+2) | 11 (+3) | 0.706 | 2 / 16 |
| 0.30 | 41 (+2) | 1 (+1) | 16 (+8) | 0.720 | 3 / 16 |
| 0.50 | 37 (−2) | 3 (+3) | 18 (+10) | 0.741 | 5 / 16 |
| 0.80 | 38 (−1) | 3 (+3) | 22 (+14) | 0.604 | 8 / 16 |

Pareto-optimal sweep point is **κ=0.10**: most diversity gain, smallest
head-tied regression, fewest new generalists.

## Why it works (partially)

Unlike Step 43, the cluster signal is **structurally independent** of
the IDF channel that the high-IDF blend (Steps 41 + 42) already exploits.
The centroid is built from raw weighted skills pooled across the cluster
— it has signal even for programmes whose individual high-IDF skill set
is empty.

For a "thin-curriculum" Informatics programme with only 10 extracted
skills (mostly common), the cluster centroid pools 5-10 sibling
programmes' skill sets → a richer profile that includes the
discipline-typical skills the individual description missed.  The blend
then pulls the recall signal toward this enriched profile and away from
generic IT matches that happen to overlap with the thin per-row skill
set.

## What it does NOT fix

Of 16 known-failing generalist programmes from the domain-expert review,
**only 3 changed top-1 at κ=0.10**:

| programme @ institution | baseline → κ=0.10 | verdict |
|---|---|---|
| Informatics @ Klaipėdos kolegija | BI programmer → MS Power Developer | lateral |
| Information Systems @ KTU | IT systems tester → Business systems & process analyst | mild improvement |
| Multimedia Tech @ Šiaulių | Junior social media designer → IT Technikas | likely worse |

The high-profile regressions all *persist*:

- Cyber Systems @ Kauno kolegija → **still** "Programmer (building management)"
- IS Engineering @ Vilniaus kolegija → **still** "Robotization manager (cleaning business)"
- Informatics @ KTU / Klaipėda / Vilnius University → **still** admin / supporter
- Software Engineering @ all 3 institutions → **still** Low-code / Full-Stack / .NET (1 of 3 is fine)
- Multimedia Tech @ KTU → **still** Gameplay Programmer
- Media Tech @ VBC → **still** Senior IT PM

## A higher κ produces one clean win

At κ=0.30, **Informatics @ Klaipėda flips from "IT ADMINISTRATORIUS" to
"Senior / Mid Software Engineer (.NET / Delphi) — Teltonika Security"**.
This is the kind of fix the whole experiment was designed to produce:
a generalist programme matched to a domain-relevant software engineer
role instead of generic admin work.

But κ=0.30 costs +8 head-tied programmes and one new generalist job in
top-5.  The trade-off is real and asymmetric.  The 1 clean win comes
bundled with several other changes (some lateral, some debatable).

## Comparison to prior adoptions

| step | metric profile | adopted? |
|---|---|---|
| 41 (high-IDF recall blend λ=0.25) | unique +1, head-tied −4, no generalist regression | **yes** |
| 42 (high-IDF F1 blend μ=0.20) | similar pareto-positive pattern | **yes** |
| 43 (adaptive alpha) | unique −3, generalists +2, gap −2 | **no** (Pareto-negative) |
| 44 κ=0.10 (this) | unique +3, head-tied +4, generalists +1 | **TBD** |

44 at κ=0.10 is not the Pareto-positive pattern of Steps 41/42 (those
gave diversity gains AND head-tie reduction).  This trade is the
opposite axis: more diversity at the cost of more head-ties.

## Recommendation

1. **Do not auto-adopt.**  Unlike Steps 41/42, this is not a clear
   Pareto win.  The +3 diversity is real, but +4 head-ties is also
   real, and the headline failures from the domain-expert review are
   untouched.
2. **Present to user.**  The decision is a judgment call about which
   trade-off matters more for the thesis defence.
3. **If adopted**, recommend `cluster_prior_weight = 0.10` (not 0.30) —
   the κ=0.30 win on Informatics @ Klaipėda is not worth the head-tied
   regression and the additional second-order changes.

## What this rules out

The "thin curriculum" diagnosis is partially correct — pooling does
help some programmes (the +3 diversity gain).  But many failures are
not just thin-curriculum issues: they involve fundamental corpus gaps
(no creative-media jobs in CVbankas) or systematic skill-extraction
artefacts (Cyber programmes whose extracted skills overlap with
building-automation programmer text on transversal IT vocabulary).
No single-formula tweak in the alignment layer will fix those — they
need either better data on the input side or a confidence-cue
disclosure on the output side.

## Artifacts

- `summary.json` — full metric table + deltas + suspect changes per κ
- `rankings_k{0_00..0_80}.parquet` — full hybrid rankings per κ

## Combined Step 43 + 44 outcome

A failed.  B is a mild positive at κ=0.10.  Neither solves the headline
generalist misfires we documented.  The honest path forward is the
confidence cue in the UI (cheap, accurate, doesn't pretend to be a
fix it isn't).
