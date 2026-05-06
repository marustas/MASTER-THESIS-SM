# Step 38 — Cross-encoder Re-ranking

## Setup

Hybrid configurations compared on the full dataset
(γ = 0.3, semantic_top_n = 50, IPF on, programme IDF on,
implicit_confidence_mode = "sqrt").

| Config                  | α (cos) | xe_alpha | recall | xe_pool_mode      |
|-------------------------|---------|----------|--------|--------------------|
| baseline                | 0.55    | 0.00     | 0.45   | —                  |
| three_channel           | 0.275   | 0.275    | 0.45   | single             |
| three_channel_secwm     | 0.275   | 0.275    | 0.45   | section_weighted   |
| three_channel_secmax    | 0.275   | 0.275    | 0.45   | section_max        |

Section-aware variants split each programme into the same section groups
used by Step 34's section-weighted embeddings (subjects 0.35, outcomes 0.25,
identity 0.15, specialisations 0.20, _remainder 0.05) and score each
non-empty section against the full job text.  ``section_weighted`` pools
by the section weights; ``section_max`` keeps the strongest-matching
section.  This bypasses the 512-token shared budget of the cross-encoder.

Cross-encoder: `BAAI/bge-reranker-base`

## Results

| Metric | baseline | three_channel | three_channel_secwm | three_channel_secmax |
|---|---|---|---|---|
| Top-1 unique | 40 | 36 | 41 | 41 |
| Top-1 diversity | 0.8889 | 0.8000 | 0.9111 | 0.9111 |
| Top-1 max repeat | 2 | 4 | 2 | 2 |
| Top-5 generalists (>5) | 1 | 1 | 1 | 2 |
| Top-1 score mean | 0.3046 | 0.3204 | 0.3247 | 0.3285 |
| Top-1 score max | 0.6771 | 0.6909 | 0.6403 | 0.6208 |
| Top-1 score CoV | 0.3718 | 0.3492 | 0.3235 | 0.3343 |
| Gap top1↔top2 mean | 0.0694 | 0.0592 | 0.0718 | 0.0633 |
| Programmes with gap<0.02 | 12 | 16 | 14 | 19 |
| Programmes with gap<0.05 | 24 | 29 | 25 | 24 |
| Spearman sym↔hyb | 0.2634 | 0.1703 | 0.1622 | 0.1516 |
| Spearman sem↔hyb | -0.0216 | -0.2209 | -0.2178 | -0.2232 |
| Spearman base↔hyb | — | 0.9061 | 0.9009 | 0.8847 |
| Top-1 agreement w/ base | — | 25 | 27 | 21 |

## Deltas vs baseline

| Config | Δ unique_top1 | Δ top1_score_mean | Δ gap_mean | Δ gap<0.02 | Δ top5_generalists |
|---|---|---|---|---|---|
| three_channel | -4 | +0.0158 | -0.0102 | +4 | +0 |
| three_channel_secwm | +1 | +0.0201 | +0.0024 | +2 | +0 |
| three_channel_secmax | +1 | +0.0239 | -0.0060 | +7 | +1 |

## Notes

* Hypothesis under test: bi-encoder cosine has spent its variance during
  Stage 1 retrieval (cosine top-1 == hybrid top-1 in 0/45 programmes
  pre-Step 38). A cross-encoder re-evaluates each pair from scratch with
  full token-level attention, producing fresh ranking signal in the
  candidate pool.

* Single-pass cross-encoder shares its 512-token budget with the job text,
  giving each side ~256 tokens. The section-aware variants lift this
  ceiling by re-scoring per-section.

* Watch metrics in this order:
  1. **Programmes with gap < 0.02** — primary head-discrimination metric.
     Step 38 is justified iff this drops appreciably.
  2. **Top-1 diversity** — guardrail. Must not regress.
  3. **Top-5 generalists** — guardrail. Must not regress.
  4. **Top-1 score mean / max** — informational; cross-encoder scores live
     on a different scale, so absolute mean shifts are expected.

* Spearman base↔hyb measures how much the new config rearranges the old
  ranking. Low Spearman + better head discrimination = the cross-encoder
  is contributing genuinely new signal, not just redecorating the order.
