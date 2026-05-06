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

Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Results

| Metric | baseline | three_channel | three_channel_secwm | three_channel_secmax | three_channel_jobchunk | three_channel_secxjob |
|---|---|---|---|---|---|---|
| Top-1 unique | 40 | 38 | 43 | 41 | 39 | 39 |
| Top-1 diversity | 0.8889 | 0.8444 | 0.9556 | 0.9111 | 0.8667 | 0.8667 |
| Top-1 max repeat | 2 | 3 | 2 | 2 | 2 | 2 |
| Top-5 generalists (>5) | 1 | 2 | 0 | 1 | 1 | 0 |
| Top-1 score mean | 0.3046 | 0.3661 | 0.3612 | 0.3656 | 0.3513 | 0.3496 |
| Top-1 score max | 0.6771 | 0.5665 | 0.7446 | 0.6551 | 0.5874 | 0.6516 |
| Top-1 score CoV | 0.3718 | 0.2421 | 0.3134 | 0.2993 | 0.2627 | 0.3011 |
| Gap top1↔top2 mean | 0.0694 | 0.0759 | 0.0699 | 0.0678 | 0.0618 | 0.0691 |
| Programmes with gap<0.02 | 12 | 5 | 12 | 13 | 12 | 14 |
| Programmes with gap<0.05 | 24 | 22 | 26 | 29 | 24 | 25 |
| Spearman sym↔hyb | 0.2634 | 0.1239 | 0.1178 | 0.1301 | 0.1373 | 0.1287 |
| Spearman sem↔hyb | -0.0216 | -0.2445 | -0.2228 | -0.2342 | -0.2358 | -0.2138 |
| Spearman base↔hyb | — | 0.8650 | 0.8936 | 0.8880 | 0.8650 | 0.8828 |
| Top-1 agreement w/ base | — | 22 | 22 | 23 | 23 | 26 |

## Deltas vs baseline

| Config | Δ unique_top1 | Δ top1_score_mean | Δ gap_mean | Δ gap<0.02 | Δ top5_generalists |
|---|---|---|---|---|---|
| three_channel | -2 | +0.0615 | +0.0066 | -7 | +1 |
| three_channel_secwm | +3 | +0.0566 | +0.0005 | +0 | -1 |
| three_channel_secmax | +1 | +0.0609 | -0.0016 | +1 | +0 |
| three_channel_jobchunk | -1 | +0.0467 | -0.0076 | +0 | +0 |
| three_channel_secxjob | -1 | +0.0450 | -0.0003 | +2 | -1 |

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
