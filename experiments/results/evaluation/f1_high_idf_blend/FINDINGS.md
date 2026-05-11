# Step 42 — High-IDF F1 blend (mutual specificity, both directions)

## Hypothesis

Step 41 attacked specificity asymmetry from the **recall** side: a niche programme that doesn't cover a job's specific demands gets demoted (`programme_recall_high_idf`). The dual problem — a programme whose specialisation isn't demanded by the job — is invisible to `recall_high_idf` because that metric only ever asks "does the programme cover the job?". The motivating diagnostic in this session identified six "wrong-vertical" Bad matches (cybersec→BMS programmer, IS-tech→sales engineering, design→PM) where transversal IT vocabulary inflates recall but the high-IDF skill sets are disjoint.

The dual is `programme_precision_high_idf` (fraction of the *programme's* high-IDF skills the job actually demands). The harmonic mean of the two is `F1_high_idf`, which collapses to zero on either-side high-IDF gaps. Blended into the Stage 2 symbolic signal as

```
final_signal = (1 − μ) · recall_blend + μ · F1_high_idf
```

where `recall_blend` is the Step 41 output. μ = 0 reduces to current Step 41 behaviour; μ = 1 fully replaces recall with F1.

## Sweep

| μ | top-1 unique | head-tied (gap<0.02) | top-5 generalists (freq>5) | top-5 max freq | top-1 score mean | top-1 score max | top-10 Jaccard vs μ=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (baseline) | **41** | 8 | 1 | 6 | 0.307 | 0.677 | 1.000 |
| 0.10 | 40 | 9 | 1 | 6 | 0.307 | 0.677 | 0.951 |
| 0.15 | 39 | 11 | 1 | 6 | 0.308 | 0.677 | 0.948 |
| **0.20** | 39 | 8 | **0** | **5** | 0.308 | 0.677 | 0.948 |
| 0.25 | 39 | 9 | 1 | 6 | 0.309 | 0.677 | 0.932 |
| 0.30 | 39 | 9 | 1 | 6 | 0.310 | 0.677 | 0.925 |
| 0.40 | 40 | 9 | 1 | 6 | 0.313 | 0.677 | 0.902 |
| 0.50 | 39 | 10 | 1 | 6 | 0.315 | 0.677 | 0.876 |
| 0.75 | 38 | **7** | **0** | **5** | 0.319 | 0.547 | 0.803 |
| 1.00 | 33 | 14 | 3 | 7 | 0.215 | 0.468 | 0.431 |

No setting strictly dominates the baseline on aggregate metrics — the gain is *qualitative*, not aggregate-numerical.

## Per-programme top-1 changes (vs μ=0)

The diagnostic from earlier in this session classified the 45 baseline top-1 matches as Good (G), Neutral (N), or Bad (B). The table below tracks each flip's quality direction.

### μ = 0.20 (5 changes)

| pid | Programme | Old top-1 | New top-1 | Old verdict | New verdict | Direction |
|---:|---|---|---|:---:|:---:|:---:|
| 11 | Informatics (Klaipėdos kol.) | MS Power Developer | BI programuotojas | B | N | **B → N** ↑ |
| 21 | Game Development (VVK) | Paid Summer IT Internship | Gameplay Programmer | B | G | **B → G** ↑↑ |
| 24 | IS and Cyber Security (VU) | Informacijos saugos specialistas | IRT/SOC IT security specialist | G | G | = |
| 25 | IS Engineering (Utena College) | IT inžinierius | IT On-site inžinierius | N | N | = |
| 38 | Digital tech & cybersec (MRU) | SOC analitikas | IRT/SOC IT security specialist | G | G | = |

**Net at μ = 0.20: 2 quality improvements, 0 regressions, 3 lateral changes.** Removes the only top-5 generalist (1 → 0). Costs −2 unique top-1 (41 → 39). Top-10 Jaccard 0.948 (gentle).

### μ = 0.40 (9 changes)

Adds to the μ=0.20 changes:

| pid | Programme | Old top-1 | New top-1 | Direction |
|---:|---|---|---|:---:|
| 2 | Information Systems (KTU) | IT sistemų testuotojas | Verslo sistemų ir procesų analitikas | **N → G** ↑ |
| 29 | Informatics Engineering (KTU) | AI Engineer (Applied AI) | IT systems administrator | **N → B** ↓ |
| 30 | Informatics engineering (KU) | IT inžinierius | Experienced AI Engineer/Data Scientist | N → ? (better vertical, wrong level) |
| 40 | Software Systems (Kauno kol.) | IT specialist assistant | Experienced AI Engineer/Data Scientist | B → ? (better vertical, wrong level) |

**Net at μ = 0.40: 3 clear improvements (#2, #11, #21), 1 regression (#29), 2 borderline level-mismatch flips, 3 lateral changes.** Quality gain ~+2.

### μ = 0.75 (10 changes)

Adds to the μ=0.40 changes:

| pid | Programme | Old top-1 | New top-1 | Direction |
|---:|---|---|---|:---:|
| 14 | Cybersecurity Technologies (VVK) | Pažeidžiamumų valdymo (pentester) | IRT/SOC IT security specialist | G → G (lateral, both cybersec) |
| 33 | Cyber Systems and Security (Kauno kol.) | PROGRAMUOTOJAS (PASTATŲ VALDYMAS) — BMS programmer | Cyber Security Specialist | **B → G** ↑↑ (the marquee fix) |

**Net at μ = 0.75: 4 clear improvements (#2, #11, #21, #33), 1 regression (#29), 5 lateral/borderline.** The #33 fix is the only configuration that resolves the cybersec→BMS-programmer wrong-vertical regression that motivated this experiment. Cost: −3 unique top-1, top-1 score max degrades 0.677 → 0.547 (Bioinformatics top-1 score collapses because it has near-zero F1_hi against any candidate), top-10 Jaccard drops to 0.803.

### μ = 1.00 — confirmed unstable

Same lesson as Step 41's pure-replacement (λ=1) test: 33 unique top-1, 14 head-tied, 3 generalists. Many programmes (transversal-only, generic informatics) have F1_hi = 0 against most candidates because their high-IDF skill sets are sparse or empty; the symbolic signal collapses and the ranking is driven entirely by cosine, which reintroduces all the generalist bias the earlier hybrid steps fought to remove.

## Wrong-vertical Bad matches (the experiment's primary target)

The diagnostic listed six wrong-vertical Bad matches as candidates for F1 to fix:

| pid | Programme | Bad top-1 | Fixed at μ=0.20? | μ=0.40? | μ=0.75? |
|---:|---|---|:---:|:---:|:---:|
| 16 | Media Technologies (VVK) | Senior IT PM | no | no | no |
| 18 | Programming & Internet Tech (VVK) | D365 PM | no | no | no |
| 19 | Digital Design Technologies (LBC) | Senior IT PM | no | no | no |
| 26 | IS Engineering (Vilniaus kol.) | Robotization manager (cleaning industry) | no | no | no |
| 28 | IS Technology (Šiaulių kol.) | Sales engineer | no | no | no |
| **33** | **Cyber Systems and Security (Kauno kol.)** | **BMS programmer** | **no** | **no** | **YES** |

Only 1 of the 6 wrong-vertical Bad matches is fixed by F1, and only at μ = 0.75. The other five are **role-level mismatches** (Senior PM, Manager, Sales engineer matched to fresh-grad curricula), not high-IDF vertical mismatches in the precision sense. Their failure mode is that the senior/manager job's skill list is dominated by transversal management vocabulary (low IDF), so it sits *below* the median IDF threshold — F1_high_idf has nothing to penalise. These cases need the **job-title seniority filter** (Phase 2 of the proposed roadmap), not a symbolic-side fix.

## Decision

**Adopt μ = 0.20 as the canonical default.** Rationale:

- **Cleanly Pareto-positive on quality**: 2 Bad matches improved (#11 Informatics → BI programuotojas; #21 Game Development → Gameplay Programmer). Zero regressions.
- **Removes the last top-5 generalist** (1 → 0; max freq 6 → 5).
- **Preserves head-tied count** (8, identical to baseline).
- **Gentle change footprint**: top-10 Jaccard 0.948, only 5 of 45 top-1 picks flip, score_max preserved at 0.677.
- **Diversity cost** −2 unique top-1 (41 → 39) is a *price for correctness*: the 2 lateral changes (#24, #38) converge two cybersec programmes to similar SOC-flavoured matches, which the earlier IPF diagnostic showed is legitimate convergence (both programmes are >0.85 cosine similar).

**μ = 0.40 and μ = 0.75 are noted as escalation options** for cases where the curriculum mapping needs more aggressive specificity weighting at the cost of more lateral movement and (for μ=0.75) score quality. The marquee #33 BMS-programmer regression is *only* fixable via μ ≥ 0.75 — that single match is not worth the broader degradation.

**μ = 1.00 confirmed unstable** and is rejected, mirroring Step 41's λ=1 finding.

## What this experiment leaves untouched

The five remaining wrong-vertical Bad matches (#16, #18, #19, #26, #28) are *role-level* mismatches between fresh-graduate curricula and senior/manager job titles. They do not respond to high-IDF F1 because senior/manager job descriptions are dominated by transversal vocabulary that sits below the median IDF threshold. The F1 metric correctly does not fire on these jobs because they have no high-IDF skills to compare against. The right intervention is a **job-title seniority filter** (Phase 2), not further symbolic-formula refinement.

## Implementation summary

- New metric `programme_precision_high_idf` in `src/alignment/symbolic.py:184`, with the same transversal-fallback semantics as `programme_recall_high_idf`.
- New parameter `hi_idf_f1_lambda` (default 0.20) in `align_hybrid` and `run_hybrid_alignment`.
- Stage 2 symbolic signal blends `recall_blend` (Step 41 output) with `F1_high_idf = 2·P·R / (P + R)` using safe division when `P + R = 0`.
- 8 new tests added (5 in `tests/alignment/test_symbolic.py`, 3 in `tests/alignment/test_hybrid.py`); full suite now 569 passing.

## Artefacts

- `experiments/results/evaluation/f1_high_idf_blend/sweep.json` — per-μ aggregate metrics
- `experiments/results/evaluation/f1_high_idf_blend/top1_per_mu.parquet` — every programme's top-1 pick at every μ
- `experiments/scripts/f1_high_idf_sweep.py` — reproducer
