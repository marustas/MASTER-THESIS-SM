# Implementation Progress

## Step 36 — Replace LinkedIn Data Source (Commercial Viability) [ ]

**Deferred — not required for thesis defence. Only relevant if project is productised.**

LinkedIn ToS explicitly prohibits scraping; *hiQ v. LinkedIn* (2022) settled in LinkedIn's favour. Commercial use of the existing LinkedIn-derived corpus is legally exposed (ToS breach + GDPR risk on personal data in postings). Research/thesis use is a tolerated grey area and does not require action now.

Replace LinkedIn ingestion with licensed / open sources before any paid pilot or commercial release:

1. **EURES** (EU Employment Services API) — free, official, EU-wide coverage, pristine provenance
2. **Adzuna API** — licensed commercial aggregator, ~€500–€2k/mo, strong UK/DE/FR/NL/PL
3. **CVbankas.lt partnership** — negotiate commercial licence for existing LT coverage
4. **Optional fallbacks:** Jooble / Careerjet partner feeds, Cedefop Skills-OVATE, SerpAPI / JobsPikr (legal risk outsourced by contract)

Rebuild `all_jobs.json` without LinkedIn rows, re-run steps 3–11, re-validate ranking quality against the existing expert-reviewed baseline. Dropping LinkedIn likely improves procurement story ("fully licensed data, auditable provenance, GDPR-compliant") — not a loss for the commercial pitch.

**Rationale:** keeps the LinkedIn-inclusive pipeline intact for the thesis; isolates the replacement as a commercialisation concern to be handled only when needed.

**Output:** replacement job corpus in `data/raw/job_ads/` (licensed sources only), re-run evaluation results
**Module:** new `src/scraping/eures.py`, new `src/scraping/adzuna.py`, updated `src/scraping/job_ads.py`

---

## Step 41 — Symbolic Specificity Refinement (high-IDF recall blend) [x] adopted

After three negative results from Steps 38 (cross-encoder), 39 (LTR), and 40 (asymmetric encoder), the bottleneck was clearly on the symbolic side, not the encoder or the blending function. Added a second symbolic recall signal `programme_recall_high_idf` restricted to ESCO URIs whose corpus IDF exceeds the median (4.27 in the canonical corpus), with a fallback for transversal-only programmes that uses the standard recall. Blended into Stage 2 of `align_hybrid` as `(1 − λ) · programme_recall + λ · programme_recall_high_idf`. Pure replacement (λ=1) regressed every metric — the median IDF is a high bar and many jobs zero out. The blend keeps the original recall as a floor while adding a specificity-bonus channel.

**Sweep over λ ∈ {0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.00}** identified λ = 0.25 as the Pareto-optimal setting:

| metric | baseline (λ=0) | adopted (λ=0.25) | Δ |
|---|---:|---:|---:|
| top-1 unique | 40 / 45 | **41 / 45** | +1 |
| head-tied (gap < 0.02) | 12 / 45 | **8 / 45** | −4 (33%) |
| top-5 generalists (freq > 5) | 1 | 1 | = |
| top-1 score mean | 0.305 | 0.307 | +0.002 |
| top-1 score max | 0.677 | 0.677 | = |
| top-10 Jaccard vs baseline (mean) | 1.0 | 0.956 | — |

Two programmes flip top-1, both clean wins driven by a single shared on-domain high-IDF URI: **[4] Marketing Technologies** (E-commerce CSM → Analytics Engineer Marketing DSA via shared "marketing analytics") and **[34] Multimedia design** (junior social media → UX/UI Designer via shared "graphic design"). 34 of 45 programmes have *identical* top-10; mean per-programme top-10 Jaccard 0.956 — the change is gentle and surgical.

**Adopted as the canonical hybrid default.** `align_hybrid` and `run_hybrid_alignment` now blend by default with `hi_idf_blend_lambda = 0.25`; the legacy `symbolic_signal_mode` dispatch and pure-replacement code path were removed, since the experiment confirmed the blend strictly dominates the alternatives. Canonical pipeline outputs (`exp3_hybrid/rankings.parquet`, cross-strategy summary, recommendations, exports) regenerated under the new default.

**Output:** `experiments/results/evaluation/hi_idf_recall/{FINDINGS.md, summary.json, summary_blend.json}`, `experiments/results/exp3_hybrid/{rankings.parquet, summary.json}` regenerated, `experiments/results/exports/programme_job_mapping.csv` regenerated.
**Module:** `src/alignment/symbolic.py` (added `programme_recall_high_idf`, `_filter_high_idf`, `programme_recall_high_idf` column in `align_symbolic_weighted`), `src/alignment/hybrid.py` (`hi_idf_blend_lambda` parameter, blend inlined as the only path), `tests/alignment/test_symbolic.py` (+12 tests), `tests/alignment/test_hybrid.py` (+2 tests for the blend lambda parameter). Suite now 561 tests.

---

## Step 42 — High-IDF F1 Blend (mutual specificity, both directions) [x] adopted

Step 41 attacked specificity asymmetry from the **recall** side. The dual problem — a programme whose specialisation is *not* demanded by the job — is invisible to `programme_recall_high_idf`. The diagnostic identified six wrong-vertical Bad matches (cybersec→BMS programmer, IS-tech→sales engineering, design→PM) where transversal IT vocabulary inflates recall but the high-IDF skill sets are disjoint.

Added `programme_precision_high_idf` (fraction of the programme's high-IDF skills the job demands) with the same transversal-fallback semantics as `programme_recall_high_idf`. Combined as `F1_high_idf = 2·P·R / (P + R)`. Blended into the Stage 2 symbolic signal:

```
final_signal = (1 − μ) · recall_blend + μ · F1_high_idf
```

where `recall_blend` is the Step 41 output. μ = 0 reduces to Step 41 behaviour; μ = 1 fully replaces recall with F1.

**Sweep over μ ∈ {0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00}** identified μ = 0.20 as Pareto-optimal:

| metric | baseline (μ=0) | adopted (μ=0.20) | Δ |
|---|---:|---:|---:|
| top-1 unique | 41 / 45 | 39 / 45 | −2 |
| head-tied (gap < 0.02) | 8 / 45 | 8 / 45 | = |
| top-5 generalists (freq > 5) | 1 | **0** | −1 |
| top-5 max freq | 6 | **5** | −1 |
| top-1 score mean | 0.307 | 0.308 | +0.001 |
| top-1 score max | 0.677 | 0.677 | = |
| top-10 Jaccard vs baseline | 1.0 | 0.948 | — |

**Per-programme net quality at μ=0.20: 2 Bad→better, 0 regressions, 3 lateral changes.** Removes the last top-5 generalist. The 2 quality wins: **[11] Informatics (Klaipėdos kol.)** MS Power Developer → BI programuotojas (B→N); **[21] Game Development (VVK)** Paid Summer IT Internship → Gameplay Programmer (B→G).

**μ = 1.00 confirmed unstable** (mirroring Step 41's λ=1): 33 unique top-1, 14 head-tied, 3 generalists. Many transversal-only programmes have F1_hi = 0 against most candidates, so the symbolic signal collapses and cosine drives the ranking — reintroducing the generalist bias.

**Limitation:** of 6 wrong-vertical Bad matches the experiment targeted, only 1 (#33 Cyber Systems & Security → BMS programmer) is fixable by F1, and only at μ = 0.75 (which costs top-1 score max 0.677 → 0.547 and 3 unique). The other 5 are **role-level mismatches** (Senior PM, Manager, Sales engineer matched to fresh-grad curricula) — the senior-job descriptions are dominated by transversal management vocabulary below the median IDF threshold, so F1_hi has nothing to penalise. These cases need a **job-title seniority filter** (Phase 2), not further symbolic-formula refinement.

**Adopted as the canonical hybrid default.** `align_hybrid` and `run_hybrid_alignment` now blend the Step 41 recall_blend with F1_high_idf at `hi_idf_f1_lambda = 0.20` by default. μ = 0.40 and μ = 0.75 are documented as escalation options.

**Output:** `experiments/results/evaluation/f1_high_idf_blend/{FINDINGS.md, sweep.json, top1_per_mu.parquet}`, `experiments/scripts/f1_high_idf_sweep.py` (reproducer).
**Module:** `src/alignment/symbolic.py` (added `programme_precision_high_idf`), `src/alignment/hybrid.py` (`hi_idf_f1_lambda` parameter, F1 blend inlined into Stage 2), `tests/alignment/test_symbolic.py` (+5 tests), `tests/alignment/test_hybrid.py` (+3 tests). Suite now 569 tests.

---

## Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
