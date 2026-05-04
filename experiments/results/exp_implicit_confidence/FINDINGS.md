# Step 37 — Confidence-weighted Implicit Skills (T2a)

## Setup

Three implicit-skill weighting modes were compared in the hybrid alignment.
All other parameters held fixed at current production values:
α = 0.55, γ = 0.3, semantic_top_n = 50, IPF top-30 (two-tier),
programme IDF on, confidence-aware cosine normalisation on.

| Mode    | Implicit weight formula                                     |
|---------|--------------------------------------------------------------|
| uniform | `0.5` — paper baseline (Gugnani & Misra 2020 E3)            |
| linear  | `0.5 × clip((conf − 0.70) / 0.30, 0, 1)`                    |
| sqrt    | `0.5 × sqrt(clip((conf − 0.70) / 0.30, 0, 1))`              |

Implicit confidence = propagation cosine to the source neighbour, range
0.70–1.00 after Step 4b filtering.

**Distribution of implicit confidence in the dataset (N = 5 932):**

| Statistic | Value |
|-----------|-------|
| min       | 0.700 |
| p10       | 0.707 |
| p25       | 0.717 |
| **p50**   | **0.744** |
| p75       | 0.790 |
| p90       | 0.855 |
| max       | 1.000 |
| mean      | 0.769 |

The distribution is heavily concentrated near the 0.70 floor, so the choice
of decay shape near the floor matters a lot:

| Confidence | linear factor | sqrt factor |
|------------|--------------|-------------|
| 0.70       | 0.000        | 0.000       |
| 0.74 (p50) | 0.067        | 0.182       |
| 0.79 (p75) | 0.150        | 0.274       |
| 0.86 (p90) | 0.267        | 0.366       |
| 1.00       | 0.500        | 0.500       |

At the median confidence, linear gives the implicit skill ≈ 1/8 of its
uniform weight; sqrt gives ≈ 1/3. Both modes compress the implicit channel,
but sqrt preserves substantially more signal mass.

## Results

| Metric                    | uniform (baseline) | linear  | sqrt    |
|---------------------------|--------------------|---------|---------|
| Top-1 unique              | 40                 | 40      | 40      |
| Top-1 diversity           | 0.889              | 0.889   | 0.889   |
| Top-1 max repeat          | 3                  | 2       | 2       |
| Top-5 generalists (>5)    | 1                  | 0       | 1       |
| Top-1 score mean          | 0.3095             | 0.3014  | 0.3046  |
| Top-1 score max           | 0.6771             | 0.6313  | 0.6771  |
| Top-1 score CoV           | 0.391              | 0.360   | 0.372   |
| Gap top-1↔top-2 mean      | 0.0700             | 0.0698  | 0.0694  |
| Programmes with gap<0.02  | 15                 | 13      | **12**  |
| Programmes with gap<0.05  | 23                 | 24      | 24      |
| Spearman sym↔hyb (mean)   | 0.319              | 0.253   | 0.263   |
| Spearman sem↔hyb (mean)   | -0.033             | -0.001  | -0.022  |

## Deltas vs uniform baseline

| Mode   | Δ unique_top1 | Δ top1_max   | Δ gap<0.02 | Δ top1_mean | Δ Spearman sym↔hyb |
|--------|---------------|--------------|-----------|-------------|--------------------|
| linear | 0             | -0.046 (-7%) | **-2**    | -0.008      | -0.066             |
| sqrt   | 0             | **0.000**    | **-3**    | -0.005      | -0.056             |

## Reading the numbers

* **Diversity is preserved.** All three modes produce the same 40/45 unique
  top-1 jobs. Confidence weighting does not break the top-1 distribution.

* **Head discrimination improves.** The number of programmes whose top-1 vs
  top-2 gap is below 0.02 drops 15 → 12 under sqrt (a 20% reduction of
  fragile heads) and 15 → 13 under linear. This is the principal metric we
  designed the experiment to move.

* **Top-1 max score is preserved by sqrt, lost by linear.** Sqrt keeps the
  best-matched programme at the same 0.677. Linear drops it to 0.631
  (−7%) — too aggressive a suppression of the implicit channel for the
  programmes that lean heavily on implicit evidence.

* **Generalist concentration improves under both.** Top-1 max repeat drops
  3 → 2 (no job dominates more than two programmes); linear additionally
  eliminates the lone top-5 generalist (1 → 0).

* **Cross-strategy correlations drop modestly.** Sym↔hyb falls 0.32 → 0.26
  under sqrt and 0.32 → 0.25 under linear. The hybrid becomes slightly more
  independent of the symbolic signal it depends on. Methodologically this
  is a soft positive (less duplication of signal) but should be reported,
  not buried.

* **Score mean compression is small.** Sqrt costs 0.005 of mean top-1 score,
  linear costs 0.008. Negligible against the 0.31 baseline.

## Decision

**Adopt sqrt as the recommended implicit-confidence mode.**

Rationale:

1. **It moves the metric we designed the experiment to move.** Sqrt removes
   3 programmes from the head-tied bucket (gap < 0.02), where the choice of
   top-1 is essentially noise. Linear moves only 2 and at higher cost.

2. **It preserves the strongest matches.** Top-1 max score stays at 0.677.
   The well-matched programmes (AI, cybersecurity, software systems) are
   not penalised for ambiguity in implicit propagation that does not affect
   their head match.

3. **It matches the empirical confidence distribution.** Linear is calibrated
   for a uniform confidence distribution; sqrt is appropriate when most
   confidence values cluster near the floor, which is the case here
   (p50 = 0.74, only ~10% above 0.86). Penalising the median implicit skill
   by 5/6 of its weight (linear) is harsh given the floor is at 0.70 by
   construction (Step 4b filter), not by extraction failure.

4. **It is methodologically defensible.** "Weight implicit skills by their
   propagation confidence" is a straightforward upgrade over a flat 0.5;
   the sqrt curve is a standard choice for confidence-weighted retrieval
   and avoids the harshness of linear scaling near the floor.

## Action items

* **Default in code: not changed yet.** The `implicit_confidence_mode`
  parameter defaults to `"uniform"` in `build_weighted_skills`,
  `align_symbolic_weighted`, and `align_hybrid` to keep the canonical
  pipeline output reproducible until the user approves the default flip.

* **Recommended call site:**
  `align_hybrid(df, implicit_confidence_mode="sqrt", ...)`

* **If the default is flipped, refresh:**
  `experiments/results/exp1_symbolic*/`, `experiments/results/exp3_hybrid/`,
  and downstream artefacts (`evaluation/`, `recommendations/`).

## Files written

* `experiments/results/exp_implicit_confidence/summary.json`
* `experiments/results/exp_implicit_confidence/rankings_uniform.parquet`
* `experiments/results/exp_implicit_confidence/rankings_linear.parquet`
* `experiments/results/exp_implicit_confidence/rankings_sqrt.parquet`
* `experiments/results/exp_implicit_confidence/FINDINGS.md` (this file)
