# Extraction-pilot protocol (intermediate defence)

## Purpose

Quantify how accurately the explicit + implicit ESCO skill extractor
(`src/skills/explicit_extractor.py`, `src/skills/implicit_extractor.py`,
following Gugnani & Misra, 2020) reproduces a human reading of the same
document. The pilot is intentionally small-scale (N = 5 programmes +
5 job advertisements, single annotator) and is presented as the
*pilot* validation. The locked 6-expert × 45-programme protocol
(`experiments/expert_eval/`) is the planned *Phase 2* validation and is
framed as future work in the next three months.

## Sample selection

- **Programmes:** 5 of the 45 in the canonical dataset, stratified by
  `cluster_label`. One programme per cluster, picked in
  descending-cluster-size order; deterministic given a fixed seed.
- **Job advertisements:** 5 of the 520 in the canonical dataset,
  stratified the same way.
- **Seed:** 42 (recorded in `selection.json` together with the picked
  IDs for reproducibility).
- **Exclusions:** rows with `cluster_label` NaN or `< 0` (HDBSCAN noise)
  are excluded so every sampled document sits inside a labelled
  cluster.

To produce the sample:

```bash
.venv/bin/python -m experiments.scripts.extraction_pilot_sample
```

The script writes four artefacts to
`experiments/validation/extraction/`:

| Artefact | Purpose |
|---|---|
| `programmes_template.csv` | Blank annotation sheet, 5 docs × 8 rows |
| `jobs_template.csv` | Same for the 5 job ads |
| `DOCS_TO_READ.md` | Full text of each sampled document for the annotator |
| `selection.json` | Seed + picked doc IDs (reproducibility record) |

## Annotation procedure (blind)

The annotator (researcher, for the intermediate defence) must follow
this order strictly. Reading the extractor's output before annotating
biases the annotation toward the extractor.

1. Open `DOCS_TO_READ.md` and read the first document end to end.
2. For each skill you identify, open the ESCO taxonomy browser
   (https://esco.ec.europa.eu/) and search for the canonical concept.
3. Record one row per skill in `programmes_template.csv` or
   `jobs_template.csv`, copy the file to a `_filled.csv` suffix:
   - `esco_uri` — the canonical URI from ESCO
     (e.g. `http://data.europa.eu/esco/skill/3a01a37b-fcfc-...`)
   - `preferred_label` — the ESCO preferred label
   - `annotation_type` — `explicit` if you can point at the surface
     form in the text, `implicit` if you inferred it from domain
     context
   - `annotator_confidence` — `high`, `medium`, or `low`
   - `notes` — free text for difficult cases
4. Delete unused blank rows; add more rows if a document needs more
   than the eight slots that the template ships with.
5. Only after all 10 documents are annotated, run the comparison.

**Do not** look at the dataset's `skill_details` column or any
existing `experiments/results/...` artefact until the annotations are
saved.

## Comparison

```bash
.venv/bin/python -m experiments.scripts.extraction_pilot_compare
```

Reads `programmes_filled.csv` and `jobs_filled.csv` from the same
directory, plus the canonical `data/dataset/dataset.parquet` for the
extractor output. Writes:

| Artefact | Contents |
|---|---|
| `metrics.csv` | Per-document precision / recall / F1 / Jaccard plus a `micro-average` row |
| `diff.csv` | One row per URI per document, tagged TP / FP / FN / NEAR_MISS |
| `REPORT.md` | Defence-ready Markdown table + per-document error block |

## Metrics and error taxonomy

For each document the annotator's URI set is the **gold standard**;
the extractor's URI set is the prediction.

- **TP** — URI present in both.
- **FP** — URI predicted by the extractor, absent from gold.
- **FN** — URI in gold, missed by the extractor.
- **NEAR_MISS** — sub-category of FP. An FP URI whose preferred-label
  tokens overlap with a gold URI's preferred-label tokens at
  Jaccard ≥ 0.5 (token Jaccard over case-folded labels, stop-tokens
  ≤ 3 chars dropped). Captures sibling-concept errors (e.g.
  `Python (computer programming)` vs `program in Python`); these are
  not strictly wrong, just non-canonical. **NEAR_MISS counts are
  reported but still contribute to FP in the headline precision
  figure.** The protocol's review step below decides whether to
  re-label them post-hoc.

Per document the script computes:

- precision $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FP})$
- recall $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$
- $F_1 = 2 \cdot \mathrm{precision} \cdot \mathrm{recall} /
  (\mathrm{precision} + \mathrm{recall})$
- set Jaccard $= |\text{gold} \cap \text{pred}| / |\text{gold} \cup \text{pred}|$

The micro-average row aggregates the four counters across all 10
documents before computing the same metrics. Macro averages can be
read off the `metrics.csv` by averaging the per-document precision /
recall / F1 columns.

Explicit vs implicit split: report two sub-totals separately, since the
modules have different failure modes — explicit relies on
PhraseMatcher and produces FPs in over-rich documents; implicit
propagates from corpus neighbours and produces FNs when the document
sits in a sparse neighbourhood. Both are interesting for the
defence narrative.

## Review step

After the comparison runs, inspect `REPORT.md` and:

1. Re-check every NEAR_MISS — confirm whether the extractor's pick is
   defensible as the same concept under a different surface form. Move
   any that you accept as "good enough" out of FP (record the decision
   in `notes` for transparency).
2. Skim the FP list. Common causes to call out in the chapter:
   over-broad PhraseMatcher hits, repeated noise terms (e.g. months,
   article words mapped to spurious ESCO concepts), implicit
   propagation from off-topic neighbours.
3. Skim the FN list. Common causes: terms outside the ESCO vocabulary,
   terms present in the text but split across lines, Lithuanian
   surface forms that did not normalise to an English ESCO label.

## Reporting in the thesis chapter

Recommended structure for the defence section (≤ 2 pages):

1. **Sample and procedure** — copy the relevant parts of this
   protocol (one paragraph).
2. **Headline metrics** — the micro-average row from `metrics.csv` for
   programmes and jobs separately.
3. **Per-document table** — `metrics.csv` rendered as a table.
4. **Qualitative error analysis** — 3–5 representative FP and FN
   examples from `diff.csv`, with the conjectured cause for each.
5. **Limitations** — N = 1 annotator, N = 10 documents, no
   inter-annotator agreement available; framed against the planned
   Phase-2 expert evaluation.

## Limitations (defence-honest)

- **Single annotator.** No inter-annotator agreement statistic. The
  expert-evaluation protocol (Phase 2) addresses this with 6 experts
  and Cohen / Fleiss κ on a Stage 1 calibration set.
- **Sample size.** 5 + 5 documents is sufficient for exploratory
  error analysis but not for stable precision / recall estimates.
  Variance bars on the per-document numbers will be wide.
- **Researcher bias.** The annotator built the extractor. Even with
  blind annotation the procedure does not fully control for the
  annotator's knowledge of which surface forms the extractor matches
  well. The phrasing in the chapter should say so explicitly.
- **Near-miss heuristic.** Label-token Jaccard at 0.5 is a coarse
  proxy for "same conceptual skill". A more rigorous variant would
  use the ESCO `broaderConcept` relation, but that is not present in
  the local CSV loader and would require an additional ESCO download.
  Recorded as follow-up work.

## File layout

```
experiments/validation/extraction/
├── PROTOCOL.md             (this file)
├── DOCS_TO_READ.md         (generated by sampler)
├── selection.json          (generated by sampler)
├── programmes_template.csv (generated by sampler — blank)
├── jobs_template.csv       (generated by sampler — blank)
├── programmes_filled.csv   (manual, gitignored)
├── jobs_filled.csv         (manual, gitignored)
├── metrics.csv             (generated by compare)
├── diff.csv                (generated by compare)
└── REPORT.md               (generated by compare)
```

## References

- Gugnani, A., & Misra, H. (2020). Implicit skills extraction using
  document embedding and its use in job recommendation.
  *Proceedings of the AAAI Conference on Artificial Intelligence,
  34*(08), 13286–13293. https://doi.org/10.1609/aaai.v34i08.7038
- European Commission. (2022). *ESCO: European Skills, Competences,
  Qualifications and Occupations* (Version 1.2.1).
  https://esco.ec.europa.eu/
