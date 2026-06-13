# RRF + BM25 Hybrid — Findings

> **Status: tried, not adopted.** Branch `feature/rrf_bm25_alignment` preserved but not merged.
> Expert evaluation: hybrid wins clearly (~32 hybrid better, ~5 RRF better, ~8 draw).

## Setup

Three-strategy Reciprocal Rank Fusion over all 45 × 520 = 23,400 pairs:

| Strategy | Signal | Score column |
|---|---|---|
| Semantic | Cosine similarity on combined embeddings | `cosine_combined` |
| Symbolic | `programme_recall` (ESCO URI weighted overlap) | `programme_recall` |
| BM25 | Okapi BM25 lexical retrieval on `cleaned_text` | `bm25_score` |

RRF formula: `score(d) = Σ_i 1 / (60 + rank_i(d))`, k=60 (Cormack et al. 2009).
IPF applied post-fusion with same two-tier parameters as `align_hybrid`.

## Headline numbers

| Metric | Hybrid (current) | RRF |
|---|---|---|
| n_pairs | 2,250 (top-50 candidates) | 23,400 (all pairs) |
| Top-1 mean score | 0.308* | 0.026† |
| Top-1 unique / 45 | **39** | 37 |
| Top-1 diversity | **87%** | 82% |
| Top-1 changed vs hybrid | — | 82% of matches |

*Hybrid score is per-programme min-max normalised — not raw.
†RRF score upper bound ≈ 0.049 (rank 1 in all 3 strategies). Not comparable to hybrid score.

## Why it failed

### 1. BM25 full-text convergence collapse

BM25 on `cleaned_text` rewards high-frequency generic IT vocabulary
("informacinės sistemos", "programavimas", "duomenų bazės") shared by all IT
job postings.  One generic job with high TF-IDF across that vocabulary gets
pushed to rank 1 for every programme in a family:

- All 4 Informatics programmes → *ERP konsultantas (SCM)*
- All 3 Software Engineering programmes → *.NET Software Engineer*
- All 3 Informatics Engineering programmes → *IT INFRASTRUKTŪROS ADMINISTRATORIUS*

The semantic stage had already learned to discriminate at domain level.
Adding BM25 as an equal-weight signal corrupts that discrimination.

### 2. Seniority mismatch

*IT Vadovas (-ė)* (IT Manager, typically 5–10 YOE) became top-1 for three
programmes.  The hybrid's IPF + symbolic recall suppresses senior generalist
roles better than RRF rank fusion.

### 3. Wrong-vertical lexical pull

*EUROPE FINANCIAL CONTROLLER • TELTONIKA IoT GROUP* appeared as top-1 for
**Information and Communication Technologies (VGTU)** — presumably because
"Teltonika IoT" terminology overlaps with programme vocabulary.  A financial
controller is entirely off-domain for an ICT graduate.

### 4. Domain-specific regression

**Game Development (Vilnius Business College)**: hybrid correctly identifies
*Gameplay Programmer* (exact match); RRF returns *Associate, Digital Review
Team Analyst (German speaker)* — requires German, off-domain entirely.

**Cybersecurity Technologies**: hybrid returns *Pažeidžiamumų valdymo –
įsilaužimų testavimo specialistas* (pen testing — correct); RRF returns
*Naudotojų Technologijų vadovas* (user technology manager — wrong level,
wrong domain).

## Where RRF was better

Two known hybrid wrong-vertical bugs were fixed:

- **Cyber Systems and Security (Kauno kolegija)**: hybrid had *PROGRAMUOTOJAS
  (PASTATŲ VALDYMAS)* (building management systems — flagged previously);
  RRF at least returns an IT-domain job.
- **Information Systems Technology**: hybrid had *Pardavimų inžinierius*
  (sales engineer); RRF returns *IT sistemų ir skaitmenizacijos vadovas*.

Both RRF improvements are still seniority-mismatched (manager-level roles),
so they are better but not correct.

## Stable matches (both approaches agree)

These are the high-confidence pairs unaffected by fusion method:

| Programme | Top-1 job |
|---|---|
| Bioinformatics | Data Analyst SQL Python |
| Computer games and animation | Game Designer |
| Digital Design Technologies | Vyr. IT projektų vadovas |
| Game Development and Digital Animation | Game Designer |
| Information System Engineering | Data Scientist |
| Multimedia and Internet Technologies | Techninio aptarnavimo specialistas (student) |

## What could salvage BM25

Restrict BM25 to extracted ESCO skill label strings only — not full
`cleaned_text`.  A BM25 index over preferred-label strings would measure
lexical skill overlap without picking up document-level generic vocabulary.
This is a fundamentally different use of BM25 (skill-level, not text-level)
and would not suffer from the convergence collapse observed here.

Alternatively, use BM25 only as a Stage 1 retrieval filter (candidate
generation) rather than as a scoring signal in fusion.  Many IR systems use
BM25 to expand the candidate pool before a neural re-ranker; BM25-as-signal
in a rank fusion is a different and weaker use case for this corpus.

## Decision

Default rankings remain with `align_hybrid` (α=0.55, IPF two-tier,
hi-IDF F1 blend).  The RRF implementation is preserved on branch
`feature/rrf_bm25_alignment` for reference.  The module
`src/alignment/rrf.py` and `tests/alignment/test_rrf.py` are not on main.
