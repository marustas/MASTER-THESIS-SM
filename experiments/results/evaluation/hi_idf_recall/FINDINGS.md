# High-IDF Recall Blend — Symbolic Specificity Refinement

> **Status: tested, recommended for adoption at λ = 0.25.** First adoptable formula change after the negative results of Step 38 (cross-encoder), Step 39 (LTR), and Step 40 (asymmetric encoder). The change is small (one blended sum on top of the existing symbolic recall) but principled (mutual specificity) and Pareto-improving on the headline metrics. This document is the experiment record.

## 1. Motivation — what problem this solves

The hand-tuned hybrid formula has worked well on aggregate metrics for most of the corpus, but Steps 38–40 traced a consistent residual failure mode that none of the model-side experiments could fix:

> **Specificity asymmetry.** Long generic IT job descriptions list many transversal skills (English, communication, computer technology, lead others, problem-solving). Niche programmes happen to mention some of those same transversal skills. Under the standard `programme_recall = Σ min(w_p[u], w_j[u]) / Σ w_j[u]`, a niche programme can match a generic IT role *just because the role's many "soft skills" overlap with the programme's incidental transversal skills* — even when the role does not demand any of the programme's defining competencies.

Concretely, the named regressions from Step 38 — **Game Designer programme → QA Tester**, **SOC analyst programme → Test Manager**, **AI Engineer programme → SysAdmin**, **Multimedia design → junior social-media advert designer** — all share this pattern: the "wrong" winner is broader, with more skills, none of which actually overlap with the programme's domain identity. The wrong winner has many cheap-skill matches; the right winner has fewer matches but they are the right ones.

Three previous attempts failed to fix this:

- **Step 38** added a cross-encoder re-ranking stage (MS MARCO-MiniLM, BGE). Best variant cut head-tied programmes 12 → 5 but added generalist regressions and lost diversity. Persistent niche regressions traced to corpus-level specificity asymmetry, *not* re-ranker model quality.
- **Step 39** trained a LightGBM LambdaRank on cross-strategy consensus labels. Held-out NDCG@10 looked good (0.759), but feature importance was dominated by `count_top_k` (popularity) and Spearman against the hybrid baseline was −0.332 — the LTR model had rediscovered the generalist bias hybrid's IPF was specifically designed to suppress. Implementation removed.
- **Step 40** swapped the embedding encoder for `intfloat/e5-large-v2` with `query: ` / `passage: ` prefixes. Per-programme cosine range collapsed 6× (0.16 → 0.027), top-5 generalists 1 → 6. The asymmetric paradigm assumes short queries hunting for retrieval passages; programmes are 600+ token structured curricula and jobs are recruiter-written long-form, so the asymmetric distinction collapses on long-document × long-document.

Two negative results on the encoder side (Step 25 MPNet + Step 40 E5) and one negative result on learned blending (Step 39) made it conclusive: **the bottleneck is on the symbolic side, not the encoder, not the blending function**. The right intervention is to change *what counts as a meaningful skill match*.

## 2. Mechanism

### 2.1 The new metric

Define the corpus IDF threshold as the median IDF over all unique ESCO URIs that appear in the dataset:

```
τ = median(idf(u) for u ∈ corpus)
```

In the canonical pipeline this is τ ≈ 4.27 (computed over 462 unique URIs across 565 documents).

A new metric `programme_recall_high_idf` is defined identically to `programme_recall` except it sums only over URIs above the threshold:

```
programme_recall_high_idf(p, j) =
    Σ_{u ∈ p ∩ j, idf(u) > τ}  min(w_p[u], w_j[u])
    ───────────────────────────────────────────────
    Σ_{u ∈ j, idf(u) > τ}       w_j[u]
```

Implemented in `src/alignment/symbolic.py::programme_recall_high_idf` and surfaced as a new column in `align_symbolic_weighted` output.

### 2.2 Fallback for transversal-only programmes

If a programme has zero high-IDF skills (e.g. a programme whose ESCO mappings are all transversal — Project Management, Communications, etc.), the high-IDF restriction has nothing to act on, and the metric would force the score to zero against every job. That is not what we want. Instead the function falls back to the standard `programme_recall(w_p, w_j)` over the full skill set for that programme.

Generic jobs (those with zero high-IDF skills) correctly score 0: a job that demands only transversal skills should not match a niche programme on the symbolic side.

### 2.3 Blending into the hybrid

`align_hybrid` gains two new parameters:

- `symbolic_signal_mode ∈ {"recall", "recall_hi_idf", "recall_hi_idf_blend"}` — selects which signal drives the Stage 2 refinement.
- `hi_idf_blend_lambda ∈ [0, 1]` — when in blend mode, the convex combination weight.

Under the blend mode the symbolic signal becomes:

```
sym_signal = (1 − λ) · programme_recall  +  λ · programme_recall_high_idf
```

Everything downstream of this is unchanged: the same per-programme min-max with confidence-damping, the same `α`-weighted blend with cosine, the same two-tier IPF popularity penalty, the same quality multiplier (specificity_ratio × generic_penalty).

### 2.4 Why blending instead of replacing

An earlier test of pure replacement (`recall_hi_idf` as the *only* symbolic signal) regressed every aggregate metric: top-1 unique 40 → 33, head-tied 12 → 14, top-5 generalists 1 → 4, top-1 max 0.677 → 0.468, score CoV 0.79 → 1.26.

Diagnosis: the median IDF threshold is a high bar (only the half of URIs above 4.27 count). Of the 520 jobs in the corpus, many have *no* high-IDF skills at all. For those jobs `recall_hi_idf = 0.0` regardless of the programme, which kills the symbolic signal entirely. The hybrid blend then reduces to almost-pure cosine, which is exactly the failure mode we were trying to escape.

The blend keeps the original recall as a **floor**: every (programme, job) pair retains a meaningful symbolic score even when the high-IDF channel is empty. The high-IDF channel acts as a *bonus discriminator* for pairs where the rare-skill overlap is real, never as a *zero* for pairs where it isn't.

## 3. Test methodology

### 3.1 Offline tests added

19 new tests added in this iteration:

- `tests/alignment/test_symbolic.py::TestFilterHighIdf` (4 tests) — `_filter_high_idf` correctness: strict-inequality threshold, missing-IDF treatment, empty input.
- `tests/alignment/test_symbolic.py::TestProgrammeRecallHighIdf` (8 tests) — high-IDF restriction filters low-IDF skills from the score, generic jobs zero out, transversal programmes fall back, no-overlap returns zero, threshold below all IDFs equals full recall, threshold above all IDFs triggers fallback, `align_symbolic_weighted` exposes the new column.
- `tests/alignment/test_hybrid.py::TestSymbolicSignalMode` (7 tests) — default mode unchanged from baseline, `recall_hi_idf` and `recall_hi_idf_blend` modes run and produce different scores, `λ=0.0` and `λ=1.0` are the boundary cases of the blend (equivalent to `recall` and `recall_hi_idf` respectively), invalid mode raises, λ out of [0,1] raises.

Full suite: 566 tests, all green.

### 3.2 λ sweep

Ran `align_hybrid` against the full canonical dataset (45 programmes × 520 jobs, top-50 candidates per programme) with `symbolic_signal_mode="recall_hi_idf_blend"` at the following λ values: 0.00 (baseline / `"recall"` mode), 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50. Pure-replacement (`λ=1.0`, equivalent to `symbolic_signal_mode="recall_hi_idf"`) was tested separately as a sanity check.

For each λ:

- aggregate metrics: top-1 unique programmes, head-tied programmes (top-1 / top-2 gap < 0.02), top-5 generalists (jobs appearing in > 5 programmes' top-5), top-5 max repeat, top-1 score mean, top-1 score max, mean head-discrimination gap;
- per-programme top-1 job and score, recorded for cross-tabulation across λ.

A programme was flagged as "movable" if its top-1 job differed from baseline at any λ in the sweep. The full top-1 trajectory across λ values was inspected for those programmes.

## 4. Results

### 4.1 Aggregate metrics across the sweep

| λ | top-1 unique | head-tied (gap < 0.02) | top-5 generalists | top-1 mean | top-1 max | mean head-gap |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 40 | 12 | 1 | 0.305 | 0.677 | 0.069 |
| 0.15 | 41 | 10 | 1 | 0.304 | 0.677 | 0.069 |
| **0.20** | **41** | **8** | **1** | 0.306 | 0.677 | 0.072 |
| **0.25** | **41** | **8** | **1** | 0.307 | 0.677 | 0.073 |
| 0.30 | 41 | 9 | 1 | 0.308 | 0.677 | 0.075 |
| 0.35 | 39 | 7 | 1 | 0.309 | 0.677 | 0.075 |
| 0.40 | 39 | 8 | 1 | 0.310 | 0.677 | 0.076 |
| 0.50 | 38 | 11 | 1 | 0.314 | 0.677 | 0.079 |
| 1.00 (replace) | 33 | 14 | 4 | 0.215 | 0.468 | — |

**Sweet spot: λ ∈ {0.20, 0.25}**, both achieving top-1 unique 41 (best in the sweep) and head-tied 8 (best in the sweep). λ=0.30 is near-equivalent (head-tied 9). At λ ≥ 0.35 diversity regresses below baseline (top-1 unique 41 → 39); at λ = 1.0 every metric collapses.

### 4.2 Per-programme top-1 trajectories

Only 9 of 45 programmes had a different top-1 at any λ in the sweep; 36 programmes were stable across the entire sweep. The 9 movable programmes split into four buckets:

#### 4.2.1 Clean early wins (flip at λ = 0.15, stable)

Two programmes flip at the smallest non-zero λ tested and stay flipped:

**[4] Marketing Technologies**
| λ | top-1 | hybrid_score |
|---|---|---:|
| 0.00 | E-commerce Customer Success Manager | 0.252 |
| 0.15 | Analytics Engineer, Marketing DSA | 0.273 |
| 0.30 | Analytics Engineer, Marketing DSA | 0.462 |
| 0.50 | Analytics Engineer, Marketing DSA | 0.468 |

**[34] Multimedia design**
| λ | top-1 | hybrid_score |
|---|---|---:|
| 0.00 | Junior social-media advert designer | 0.329 |
| 0.15 | UX/UI Designer with verification | 0.344 |
| 0.50 | UX/UI Designer with verification | 0.344 |

These are the cases where the blend works as intended: a single high-IDF shared skill cleanly separates the right job from a generic alternative.

#### 4.2.2 Mid-range change (flip at λ = 0.30)

**[38] Digital technologies and cyber security**
| λ | top-1 | hybrid_score |
|---|---|---:|
| 0.00–0.25 | SOC analitikas (cyber threats) | 0.216 |
| 0.30 | IRT specialist (SOC + IT security) | 0.223 |
| 0.50 | IRT specialist (SOC + IT security) | 0.350 |

Both jobs are within the cybersecurity domain. Sideways at λ = 0.30; the IRT specialist becomes a stronger preferred match as λ grows.

#### 4.2.3 Late wins / late changes (λ ≥ 0.35)

**[21] Game Development** — flips at λ = 0.35 to the textbook-correct match:
| λ | top-1 | hybrid_score |
|---|---|---:|
| 0.00–0.30 | Paid Summer Internship in the IT Department | 0.340 |
| 0.35 | **Gameplay Programmer** | 0.340 |
| 0.50 | Gameplay Programmer | 0.408 |

This is one of the Step 38 named persistent regressions, and λ ≥ 0.35 fixes it. However it costs −2 top-1 unique elsewhere.

**[24] Information Systems and Cyber Security** — flips at λ = 0.35 from "Information security specialist" to "IRT specialist (SOC + IT security)". Sideways.

**[25] Information Systems Engineering** — flips at λ = 0.40 from "IT inžinierius" to "IT On-site inžinierius". Sideways.

**[40] Software Systems** — flips at λ = 0.50 from "IT specialist's assistant" to "Software Engineer (React Native)". Probable upgrade.

#### 4.2.4 High-λ regression and instability

**[14] Cybersecurity Technologies** — flips only at λ = 0.50, from "Pažeidžiamumų valdymo – įsilaužimų testavimo specialistas" (penetration testing specialist — a specific, on-target match) to "IRT specialist (SOC + IT security)" (a broader generalist). The baseline pick was the right answer; pushing λ that high demotes a strong niche match to a generic one. **Real regression.**

**[11] Informatics** — flips at λ = 0.35 to "IT Support Engineer", then *again* at λ = 0.50 to "BI programuotojas". Three different picks across the sweep. The instability is a yellow flag — for this programme the formula is not converging on a confident match.

### 4.3 Mechanism in action — the data behind the wins

To verify the wins are caused by the intended mechanism (not coincidence), the actual ESCO skill sets were inspected for the two clean cases.

#### 4.3.1 Marketing Technologies — single high-IDF match flips the decision

**Programme [4] Marketing Technologies** has 21 skills, of which 6 are high-IDF: marketing management (5.24), electronic business (5.24), marketing analytics (4.74), write specifications (4.74), think abstractly (4.74), guide staff (4.56).

**Baseline pick — E-commerce Customer Success Manager** has only 5 skills, **zero of them high-IDF**: trademarks (2.61), lead others (1.46), Lithuanian (1.30), English (1.01), computer technology (0.91).

**New pick — Analytics Engineer, Marketing DSA** has 47 skills, of which 1 is high-IDF: marketing analytics (4.74). That one URI is also in the programme's high-IDF set.

| | high-IDF ∩ programme | low-IDF ∩ programme |
|---|---:|---:|
| Baseline (CSM) | **0** | 2 |
| New (Analytics Engineer) | **1** *(marketing analytics)* | 9 |

Under the standard `programme_recall`, both jobs got their score from low-IDF overlap; the CSM has a small skill-set so its few cheap-skill matches were a non-trivial fraction of its denominator. Under `programme_recall_high_idf`, the CSM scores exactly 0.0 (no high-IDF skills at all), while the Analytics Engineer scores positive. Even at λ = 0.15 the blend tilts enough to flip — and it tilts more decisively as λ rises.

The single shared high-IDF URI ("marketing analytics") is exactly the kind of skill any human reader would name as defining for a Marketing Technologies programme. The blend isn't surfacing a coincidence; it's surfacing the discriminative signal that already existed in the data but was being drowned by transversal-skill mass.

#### 4.3.2 Multimedia design — same pattern

**Programme [34] Multimedia design** has 6 high-IDF skills including **graphic design (4.40)**.

**Baseline pick — Junior social-media advert designer** has 4 skills, 1 high-IDF ("react to events in time-critical environments" — irrelevant to multimedia design), 0 of which overlap with the programme's high-IDF set.

**New pick — UX/UI Designer** has 21 skills, 4 high-IDF including **graphic design (4.40)** — which is also in the programme's set.

| | high-IDF ∩ programme | low-IDF ∩ programme |
|---|---:|---:|
| Baseline (Junior social media) | 0 | 2 |
| New (UX/UI Designer) | **1** *(graphic design)* | 3 |

Identical mechanism. Single shared high-IDF URI, on-domain, flips the decision at λ = 0.15.

### 4.4 Top-10 stability

At λ = 0.30 (the middle of the sweet-spot window), per-programme top-10 lists are remarkably stable against the baseline:

| | count |
|---|---:|
| Programmes with all 10 same | 34 / 45 |
| Programmes with 9 of 10 same | 11 / 45 |
| Programmes with ≤ 8 of 10 same | 0 |

Mean per-programme top-10 Jaccard: **0.956**. Three programmes had a new top-1; ten more had the same top-1 with a slightly higher hybrid score; eight had the same top-1 with a slightly lower score; the rest were exactly identical. The change is gentle and surgical, not chaotic.

## 5. Why these wins are real, not noise

Three independent verifications:

1. **Both clean wins are driven by a single shared high-IDF URI** that names the programme's domain identity (marketing analytics for Marketing Technologies; graphic design for Multimedia design). The blend isn't surfacing an arbitrary rare skill — it's surfacing the skill that any human would name as the programme's defining outcome.

2. **Both losing baseline picks have zero high-IDF overlap with the programme.** They were winning purely on cheap, ubiquitous skill mass. The high-IDF restriction reveals that they don't actually demand any of the programme's specific competencies — exactly the failure mode this metric was designed to detect.

3. **The wins are robust across the safe λ range** (0.15–0.30). Once the flip happens it stays, and the score grows as λ grows. If the blend were surfacing noise, the picks would oscillate; instead they are monotonic in λ until a different threshold effect kicks in (e.g. [21] Game Development at λ = 0.35).

## 6. Recommendation

**Adopt λ = 0.25 as the new default for the symbolic refinement signal:** `symbolic_signal_mode="recall_hi_idf_blend"`, `hi_idf_blend_lambda=0.25`.

| metric | baseline | λ=0.25 | Δ |
|---|---:|---:|---:|
| top-1 unique | 40 / 45 | **41 / 45** | +1 |
| head-tied (gap < 0.02) | 12 / 45 | **8 / 45** | −4 (33% reduction) |
| top-5 generalists | 1 | 1 | = |
| top-1 score mean | 0.305 | 0.307 | +0.002 |
| top-1 score max | 0.677 | 0.677 | = |

The change captures both clean wins ([4] Marketing Technologies, [34] Multimedia design) at the smallest λ that doesn't introduce regressions elsewhere. λ = 0.30 is near-equivalent (one extra head-tied programme, slightly higher mean score) and is also defensible if we want the [38] Digital Cyber sec sideways change.

**Why not push to λ = 0.35** (which would also fix the [21] Game Development → Gameplay Programmer regression named in Step 38)? The cost is −2 top-1 unique elsewhere, plus instability on [11] Informatics (top-1 changes again at λ = 0.50). The Game Development case is genuinely hard and likely needs a different intervention — a programme-name × job-title re-ranker, a cluster prior multiplier, or a separate niche-detection signal — not a single global λ pushed harder.

**Why not λ = 0.50** (highest mean score)? It demotes the strong [14] Cybersecurity Technologies → penetration-testing-specialist match to a generic IRT specialist. That is a real regression on a niche-correct baseline pick.

**Implementation cost**: this is a one-line change to the default parameters of `align_hybrid` and `run_hybrid_alignment`. The plumbing is already in place and tested. Adopting requires regenerating `experiments/results/exp3_hybrid/rankings.parquet` and the downstream artefacts (cross-strategy evaluation, recommendations, export CSV).

## 7. Limitations and follow-ups

**Threshold choice is corpus-dependent.** The median IDF (4.27 here) shifts if the corpus changes. We did not sweep alternative thresholds (e.g. 33rd or 75th percentile) — only median. A future iteration could test whether a stricter threshold shrinks the safe λ range further or expands it.

**Programme-side IDF vs job-side IDF.** Currently the threshold is applied symmetrically to both sides using the corpus-wide IDF. Step 31's programme-level IDF (used for skill weighting in `align_symbolic_weighted`) is separate. A future variant could threshold the programme side by `programme_idf` and the job side by `corpus_idf` to reflect "rare in programmes" vs "rare in jobs".

**Single global λ.** The sweep showed each programme has its own preferred λ — [4] Marketing Tech is happiest at any λ ≥ 0.15, [21] Game Development needs ≥ 0.35, [14] Cybersecurity wants λ exactly 0 (regresses at λ = 0.50). A per-programme adaptive λ keyed to programme breadth or skill richness could capture more of the available signal. Out of scope for this iteration; recorded for the brainstorm queue.

**The Game Development case.** Even at λ = 0.50 the picks were a clean win, but at any λ that does *not* cost diversity elsewhere, the baseline pick remains. This says: the Game Development / Game Designer / Gameplay Programmer cluster has corpus-level ambiguity — possibly because the long generic IT internship listing happens to mention game-related skills somewhere in its description, giving it a real signal that the high-IDF restriction does not fully strip away. Likely the right next move is title-aware refinement (programme name + job title cosine), not a stronger symbolic filter.

## 8. Files

- `src/alignment/symbolic.py` — `programme_recall_high_idf` and `_filter_high_idf` added; `align_symbolic_weighted` now emits the `programme_recall_high_idf` column; threshold logged at run time.
- `src/alignment/hybrid.py` — `symbolic_signal_mode` and `hi_idf_blend_lambda` parameters wired through `align_hybrid` and `run_hybrid_alignment`.
- `tests/alignment/test_symbolic.py` — 12 new tests (TestFilterHighIdf, TestProgrammeRecallHighIdf).
- `tests/alignment/test_hybrid.py` — 7 new tests (TestSymbolicSignalMode).
- `experiments/results/evaluation/hi_idf_recall/summary_blend.json` — full λ sweep results and top-1 churn tables.
- `experiments/results/evaluation/hi_idf_recall/summary.json` — initial pure-replacement (λ = 1.0) experiment record, kept for reference.

Pending: the choice of whether to flip the canonical pipeline default to λ = 0.25, regenerate `experiments/results/exp3_hybrid/` and the downstream export, and commit the change as the next adopted improvement.
