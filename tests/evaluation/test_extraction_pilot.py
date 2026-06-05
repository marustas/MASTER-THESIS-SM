"""Offline tests for src/evaluation/extraction_pilot.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.extraction_pilot import (
    DocumentComparison,
    aggregate_metrics,
    compare_document,
    compare_sample,
    stratified_sample_jobs,
    stratified_sample_programmes,
    write_annotation_template,
)


@pytest.fixture
def dataset() -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(10):
        rows.append({
            "source_type": "programme",
            "name": f"Programme {i}",
            "institution": f"Inst {i % 3}",
            "extended_description": "x" * (1000 + i * 100),
            "cluster_label": float(i % 5),
            "skill_uris": [f"uri:p{i}:a", f"uri:p{i}:b"],
            "skill_details": [
                {"esco_uri": f"uri:p{i}:a", "preferred_label": f"prog skill a {i}",
                 "explicit": True, "confidence": 0.9},
                {"esco_uri": f"uri:p{i}:b", "preferred_label": f"prog skill b {i}",
                 "explicit": False, "confidence": 0.8},
            ],
        })
    for j in range(20):
        rows.append({
            "source_type": "job_ad",
            "job_title": f"Job {j}",
            "employer_sector": "tech",
            "description": "y" * (500 + j * 50),
            "cluster_label": float(j % 6),
            "skill_uris": [f"uri:j{j}:a"],
            "skill_details": [
                {"esco_uri": f"uri:j{j}:a", "preferred_label": f"job skill a {j}",
                 "explicit": True, "confidence": 0.95},
            ],
        })
    return pd.DataFrame(rows)


class TestStratifiedSample:
    def test_programmes_returns_n_rows(self, dataset):
        sample = stratified_sample_programmes(dataset, n=5, seed=7)
        assert len(sample) == 5
        assert "programme_id" in sample.columns
        assert "name" in sample.columns

    def test_programmes_covers_distinct_clusters(self, dataset):
        sample = stratified_sample_programmes(dataset, n=5, seed=7)
        assert sample["cluster_label"].nunique() == 5

    def test_jobs_returns_n_rows(self, dataset):
        sample = stratified_sample_jobs(dataset, n=5, seed=7)
        assert len(sample) == 5
        assert "job_id" in sample.columns

    def test_jobs_covers_distinct_clusters(self, dataset):
        sample = stratified_sample_jobs(dataset, n=5, seed=7)
        assert sample["cluster_label"].nunique() == 5

    def test_sample_is_deterministic_given_seed(self, dataset):
        a = stratified_sample_programmes(dataset, n=5, seed=11)
        b = stratified_sample_programmes(dataset, n=5, seed=11)
        assert a["programme_id"].tolist() == b["programme_id"].tolist()

    def test_different_seeds_can_differ(self, dataset):
        a = stratified_sample_jobs(dataset, n=5, seed=1)
        b = stratified_sample_jobs(dataset, n=5, seed=99)
        assert a["job_id"].tolist() != b["job_id"].tolist()


class TestWriteAnnotationTemplate:
    def test_template_has_eight_rows_per_doc(self, tmp_path, dataset):
        sample = stratified_sample_programmes(dataset, n=3, seed=7)
        out = write_annotation_template(
            sample, tmp_path / "tpl.csv", doc_kind="programme"
        )
        df = pd.read_csv(out)
        assert len(df) == 3 * 8
        assert set(df["doc_id"].unique()) == set(sample["programme_id"].tolist())
        for col in ("doc_kind", "doc_id", "doc_title", "esco_uri", "preferred_label",
                    "annotation_type", "annotator_confidence", "notes"):
            assert col in df.columns

    def test_template_does_not_leak_extractor_output(self, tmp_path, dataset):
        sample = stratified_sample_programmes(dataset, n=2, seed=7)
        out = write_annotation_template(
            sample, tmp_path / "tpl.csv", doc_kind="programme"
        )
        df = pd.read_csv(out)
        assert df["esco_uri"].fillna("").str.strip().eq("").all()
        assert df["preferred_label"].fillna("").str.strip().eq("").all()

    def test_invalid_doc_kind_raises(self, tmp_path, dataset):
        sample = stratified_sample_programmes(dataset, n=2, seed=7)
        with pytest.raises(ValueError):
            write_annotation_template(sample, tmp_path / "tpl.csv", doc_kind="bogus")


class TestCompareDocument:
    def test_perfect_match(self):
        cmp = compare_document(
            doc_kind="programme",
            doc_id=1,
            doc_title="X",
            gold_uris=["a", "b"],
            extracted=[
                {"esco_uri": "a", "preferred_label": "skill a"},
                {"esco_uri": "b", "preferred_label": "skill b"},
            ],
        )
        assert cmp.precision == 1.0
        assert cmp.recall == 1.0
        assert cmp.f1 == 1.0
        assert cmp.jaccard == 1.0

    def test_missing_skill_yields_false_negative(self):
        cmp = compare_document(
            doc_kind="programme",
            doc_id=1,
            doc_title="X",
            gold_uris=["a", "b", "c"],
            extracted=[{"esco_uri": "a", "preferred_label": "skill a"}],
        )
        assert cmp.precision == 1.0
        assert cmp.recall == pytest.approx(1 / 3)
        assert cmp.fn == frozenset({"b", "c"})

    def test_extra_skill_yields_false_positive(self):
        cmp = compare_document(
            doc_kind="programme",
            doc_id=1,
            doc_title="X",
            gold_uris=["a"],
            extracted=[
                {"esco_uri": "a", "preferred_label": "skill a"},
                {"esco_uri": "z", "preferred_label": "skill z"},
            ],
        )
        assert cmp.precision == 0.5
        assert cmp.recall == 1.0
        assert cmp.fp == frozenset({"z"})

    def test_near_miss_detected_via_label_overlap(self):
        cmp = compare_document(
            doc_kind="programme",
            doc_id=1,
            doc_title="X",
            gold_uris=["python-prog"],
            extracted=[
                {"esco_uri": "python-comp", "preferred_label": "python computer programming"},
            ],
            uri_to_label={
                "python-prog": "program in python programming",
                "python-comp": "python computer programming",
            },
        )
        assert cmp.fp == frozenset({"python-comp"})
        assert "python-comp" in cmp.near_misses
        target, score = cmp.near_misses["python-comp"]
        assert target == "python-prog"
        assert score >= 0.5

    def test_empty_inputs_zero_metrics(self):
        cmp = compare_document(
            doc_kind="job_ad",
            doc_id=0,
            doc_title="X",
            gold_uris=[],
            extracted=[],
        )
        assert cmp.precision == 0.0
        assert cmp.recall == 0.0
        assert cmp.f1 == 0.0


class TestCompareSample:
    def test_pipeline_end_to_end(self, dataset, tmp_path):
        sample = stratified_sample_programmes(dataset, n=2, seed=7)
        annotation_rows = []
        for _, row in sample.iterrows():
            details = list(row["skill_details"])
            annotation_rows.append({
                "doc_kind": "programme",
                "doc_id": int(row["programme_id"]),
                "doc_title": row["name"],
                "esco_uri": details[0]["esco_uri"],
                "preferred_label": details[0]["preferred_label"],
                "annotation_type": "explicit",
                "annotator_confidence": "high",
                "notes": "",
            })
            annotation_rows.append({
                "doc_kind": "programme",
                "doc_id": int(row["programme_id"]),
                "doc_title": row["name"],
                "esco_uri": "uri:not-extracted",
                "preferred_label": "missed skill",
                "annotation_type": "explicit",
                "annotator_confidence": "high",
                "notes": "",
            })
        annotations = pd.DataFrame(annotation_rows)
        cmps, diff = compare_sample(annotations, dataset)
        assert len(cmps) == 2
        for cmp in cmps:
            assert "uri:not-extracted" in cmp.fn
        verdicts = set(diff["verdict"].unique())
        assert "TP" in verdicts
        assert "FN" in verdicts
        assert "FP" in verdicts or "NEAR_MISS" in verdicts

    def test_skips_empty_uri_rows(self, dataset):
        annotations = pd.DataFrame([
            {"doc_kind": "programme", "doc_id": 0, "doc_title": "Programme 0",
             "esco_uri": "", "preferred_label": "", "annotation_type": "",
             "annotator_confidence": "", "notes": ""},
        ])
        cmps, diff = compare_sample(annotations, dataset)
        assert cmps == []
        assert diff.empty


class TestAggregateMetrics:
    def test_aggregate_appends_micro_row(self):
        cmps = [
            DocumentComparison(
                doc_kind="programme",
                doc_id=1,
                doc_title="X",
                gold_uris=frozenset({"a", "b"}),
                extracted_uris=frozenset({"a", "c"}),
                tp=frozenset({"a"}),
                fp=frozenset({"c"}),
                fn=frozenset({"b"}),
            ),
            DocumentComparison(
                doc_kind="programme",
                doc_id=2,
                doc_title="Y",
                gold_uris=frozenset({"a"}),
                extracted_uris=frozenset({"a"}),
                tp=frozenset({"a"}),
                fp=frozenset(),
                fn=frozenset(),
            ),
        ]
        agg = aggregate_metrics(cmps)
        assert len(agg) == 3
        last = agg.iloc[-1]
        assert last["doc_title"] == "micro-average"
        assert last["tp"] == 2
        assert last["fp"] == 1
        assert last["fn"] == 1
        assert last["precision"] == pytest.approx(2 / 3, abs=1e-3)
        assert last["recall"] == pytest.approx(2 / 3, abs=1e-3)

    def test_aggregate_empty_returns_empty(self):
        agg = aggregate_metrics([])
        assert agg.empty
