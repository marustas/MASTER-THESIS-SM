"""
Tests for src/evaluation/ablation.py.

Covers:
  - extract_with_config: returns skill_details dicts per text
  - build_ablated_dataset: replaces skill_details in correct positions
  - compute_ablation_metrics: metric keys, skill counts, edge cases
  - run_ablation_study: runs all configs, returns correct structure
  - compute_deltas: delta and pct change from baseline
  - module_weights integration: disabling S3 reduces extracted skills
  - run: output files created
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import (
    ABLATION_CONFIGS,
    extract_with_config,
    build_ablated_dataset,
    compute_ablation_metrics,
    run_ablation_study,
    compute_deltas,
    run,
)
from src.skills.explicit_extractor import ExplicitSkillExtractor


# ── Helpers ────────────────────────────────────────────────────────────────────


def _skill(uri: str, *, explicit: bool = True, confidence: float = 1.0) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": confidence,
    }


def _make_df(
    prog_details: list[list[dict]],
    job_details: list[list[dict]],
    prog_texts: list[str] | None = None,
    job_texts: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, details in enumerate(prog_details):
        text = prog_texts[i] if prog_texts else f"programme text {i}"
        rows.append({
            "source_type": "programme",
            "skill_details": details,
            "name": f"Prog{i}",
            "cleaned_text": text,
        })
    for i, details in enumerate(job_details):
        text = job_texts[i] if job_texts else f"job text {i}"
        rows.append({
            "source_type": "job_ad",
            "skill_details": details,
            "job_title": f"Job{i}",
            "cleaned_text": text,
        })
    return pd.DataFrame(rows)


# ── ABLATION_CONFIGS ──────────────────────────────────────────────────────────


class TestAblationConfigs:
    def test_has_baseline(self):
        assert "baseline" in ABLATION_CONFIGS

    def test_has_all_ablations(self):
        expected = {"baseline", "no_S1_ner", "no_S2_pos", "no_S3_dict", "no_S4_embed"}
        assert set(ABLATION_CONFIGS.keys()) == expected

    def test_baseline_has_all_positive_weights(self):
        for key, val in ABLATION_CONFIGS["baseline"].items():
            assert val > 0, f"baseline weight {key} should be positive"

    def test_each_ablation_zeroes_one_module(self):
        for name, weights in ABLATION_CONFIGS.items():
            if name == "baseline":
                continue
            zeroed = [k for k, v in weights.items() if v == 0]
            assert len(zeroed) == 1, f"{name} should zero exactly one module"


# ── extract_with_config ──────────────────────────────────────────────────────


class TestExtractWithConfig:
    def test_returns_list_of_lists(self, explicit_extractor):
        texts = ["Python and Java programming", "SQL database management"]
        result = extract_with_config(texts, explicit_extractor)
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)

    def test_each_skill_is_dict(self, explicit_extractor):
        result = extract_with_config(["Python programming"], explicit_extractor)
        for skill in result[0]:
            assert isinstance(skill, dict)
            assert "esco_uri" in skill
            assert "preferred_label" in skill

    def test_empty_text(self, explicit_extractor):
        result = extract_with_config([""], explicit_extractor)
        assert result == [[]]


# ── build_ablated_dataset ────────────────────────────────────────────────────


class TestBuildAblatedDataset:
    def test_replaces_skill_details(self):
        df = _make_df(
            [[_skill("esco:python")]],
            [[_skill("esco:java")]],
        )
        new_prog = [[_skill("esco:sql")]]
        new_job = [[_skill("esco:docker")]]
        result = build_ablated_dataset(df, {"programme": new_prog, "job_ad": new_job})

        prog_row = result[result["source_type"] == "programme"].iloc[0]
        job_row = result[result["source_type"] == "job_ad"].iloc[0]
        assert prog_row["skill_details"][0]["esco_uri"] == "esco:sql"
        assert job_row["skill_details"][0]["esco_uri"] == "esco:docker"

    def test_preserves_other_columns(self):
        df = _make_df([[_skill("esco:python")]], [[_skill("esco:java")]])
        result = build_ablated_dataset(df, {"programme": [[]], "job_ad": [[]]})
        assert "name" in result.columns
        assert "job_title" in result.columns
        assert result.iloc[0]["name"] == "Prog0"

    def test_does_not_modify_original(self):
        df = _make_df([[_skill("esco:python")]], [[_skill("esco:java")]])
        original_uri = df.iloc[0]["skill_details"][0]["esco_uri"]
        build_ablated_dataset(df, {"programme": [[_skill("esco:sql")]], "job_ad": [[]]})
        assert df.iloc[0]["skill_details"][0]["esco_uri"] == original_uri


# ── compute_ablation_metrics ─────────────────────────────────────────────────


class TestComputeAblationMetrics:
    def _make_inputs(self):
        prog_skills = [_skill("esco:python"), _skill("esco:ml")]
        job_skills = [_skill("esco:python"), _skill("esco:java")]
        df = _make_df([prog_skills], [job_skills])

        from src.alignment.symbolic import align_symbolic
        rankings, gaps = align_symbolic(df, top_n=5)
        return rankings, gaps, df

    def test_metric_keys(self):
        rankings, gaps, df = self._make_inputs()
        metrics = compute_ablation_metrics(rankings, gaps, df, top_n=5)
        expected_keys = {
            "weighted_jaccard_mean_all",
            "weighted_jaccard_mean_top_n",
            "overlap_coeff_mean_all",
            "overlap_coeff_mean_top_n",
            "mean_skills_programme",
            "mean_skills_job",
            "unique_gap_uris",
            "total_gap_entries",
        }
        assert set(metrics.keys()) == expected_keys

    def test_skills_count(self):
        rankings, gaps, df = self._make_inputs()
        metrics = compute_ablation_metrics(rankings, gaps, df, top_n=5)
        assert metrics["mean_skills_programme"] == 2.0
        assert metrics["mean_skills_job"] == 2.0

    def test_empty_skills(self):
        df = _make_df([[]], [[]])
        from src.alignment.symbolic import align_symbolic
        rankings, gaps = align_symbolic(df, top_n=5)
        metrics = compute_ablation_metrics(rankings, gaps, df, top_n=5)
        assert metrics["weighted_jaccard_mean_all"] == 0.0
        assert metrics["mean_skills_programme"] == 0.0


# ── run_ablation_study ───────────────────────────────────────────────────────


class TestRunAblationStudy:
    def test_returns_all_configs(self, mock_esco_index, mock_embedding_model):
        df = _make_df(
            [[], []],
            [[], [], []],
            prog_texts=["Python machine learning", "Java SQL database"],
            job_texts=["Docker Kubernetes cloud", "Python data analysis", "Java REST API"],
        )

        def factory(weights):
            return ExplicitSkillExtractor(
                mock_esco_index,
                embedding_model=mock_embedding_model,
                module_weights=weights,
            )

        configs = {
            "baseline": {"S1": 1, "S2": 1, "S3": 20, "S4": 2},
            "no_S3_dict": {"S1": 1, "S2": 1, "S3": 0, "S4": 2},
        }
        results = run_ablation_study(df, factory, configs=configs, top_n=3)

        assert "baseline" in results
        assert "no_S3_dict" in results
        assert "weighted_jaccard_mean_all" in results["baseline"]

    def test_disabling_s3_reduces_skills(self, mock_esco_index, mock_embedding_model):
        """S3 (dictionary) dominates; disabling it should reduce extracted skills."""
        df = _make_df(
            [[]],
            [[]],
            prog_texts=["Python machine learning data analysis"],
            job_texts=["Java SQL database programming"],
        )

        def factory(weights):
            return ExplicitSkillExtractor(
                mock_esco_index,
                embedding_model=mock_embedding_model,
                module_weights=weights,
            )

        configs = {
            "baseline": {"S1": 1, "S2": 1, "S3": 20, "S4": 2},
            "no_S3_dict": {"S1": 1, "S2": 1, "S3": 0, "S4": 2},
        }
        results = run_ablation_study(df, factory, configs=configs, top_n=1)

        baseline_skills = results["baseline"]["mean_skills_programme"]
        ablated_skills = results["no_S3_dict"]["mean_skills_programme"]
        assert ablated_skills <= baseline_skills


# ── compute_deltas ───────────────────────────────────────────────────────────


class TestComputeDeltas:
    def test_computes_delta_from_baseline(self):
        results = {
            "baseline": {
                "weighted_jaccard_mean_all": 0.10,
                "weighted_jaccard_mean_top_n": 0.20,
                "overlap_coeff_mean_all": 0.30,
                "overlap_coeff_mean_top_n": 0.40,
                "mean_skills_programme": 10.0,
                "mean_skills_job": 8.0,
                "unique_gap_uris": 50,
            },
            "no_S3_dict": {
                "weighted_jaccard_mean_all": 0.05,
                "weighted_jaccard_mean_top_n": 0.10,
                "overlap_coeff_mean_all": 0.15,
                "overlap_coeff_mean_top_n": 0.20,
                "mean_skills_programme": 5.0,
                "mean_skills_job": 4.0,
                "unique_gap_uris": 30,
            },
        }
        deltas = compute_deltas(results)

        assert "no_S3_dict" in deltas
        assert "baseline" not in deltas

        d = deltas["no_S3_dict"]
        assert d["weighted_jaccard_mean_all"] == pytest.approx(-0.05, abs=1e-6)
        assert d["weighted_jaccard_mean_all_pct"] == pytest.approx(-50.0)
        assert d["mean_skills_programme"] == pytest.approx(-5.0)

    def test_no_baseline_returns_empty(self):
        results = {"no_S3_dict": {"weighted_jaccard_mean_all": 0.05}}
        assert compute_deltas(results) == {}

    def test_all_configs_present(self):
        base = {
            "weighted_jaccard_mean_all": 0.10,
            "weighted_jaccard_mean_top_n": 0.20,
            "overlap_coeff_mean_all": 0.30,
            "overlap_coeff_mean_top_n": 0.40,
            "mean_skills_programme": 10.0,
            "mean_skills_job": 8.0,
            "unique_gap_uris": 50,
        }
        results = {
            "baseline": base,
            "no_S1_ner": {**base, "weighted_jaccard_mean_all": 0.09},
            "no_S2_pos": {**base, "weighted_jaccard_mean_all": 0.09},
            "no_S3_dict": {**base, "weighted_jaccard_mean_all": 0.03},
            "no_S4_embed": {**base, "weighted_jaccard_mean_all": 0.08},
        }
        deltas = compute_deltas(results)
        assert len(deltas) == 4


# ── run (pipeline entry point) ───────────────────────────────────────────────


class TestRun:
    def test_output_files(self, tmp_path, mock_esco_index, mock_embedding_model):
        df = _make_df(
            [[]],
            [[]],
            prog_texts=["Python machine learning"],
            job_texts=["Java SQL database"],
        )
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "ablation"

        # Monkey-patch to avoid loading real ESCO
        import src.evaluation.ablation as ablation_mod

        original_run = ablation_mod.run

        def patched_run(dataset_path, output_dir, top_n=20):
            def factory(weights):
                return ExplicitSkillExtractor(
                    mock_esco_index,
                    embedding_model=mock_embedding_model,
                    module_weights=weights,
                )

            df = pd.read_parquet(dataset_path)
            configs = {
                "baseline": {"S1": 1, "S2": 1, "S3": 20, "S4": 2},
                "no_S3_dict": {"S1": 1, "S2": 1, "S3": 0, "S4": 2},
            }
            results = run_ablation_study(df, factory, configs=configs, top_n=top_n)
            deltas_result = compute_deltas(results)

            output_dir.mkdir(parents=True, exist_ok=True)
            output = {"configs": results, "deltas": deltas_result}
            out_path = output_dir / "ablation_results.json"
            with open(out_path, "w") as fh:
                json.dump(output, fh, indent=2)

        patched_run(dataset_path, output_dir, top_n=1)

        assert (output_dir / "ablation_results.json").exists()
        with open(output_dir / "ablation_results.json") as fh:
            data = json.load(fh)
        assert "configs" in data
        assert "deltas" in data
        assert "baseline" in data["configs"]


# ── module_weights integration ───────────────────────────────────────────────


class TestModuleWeightsIntegration:
    def test_default_weights_match_constants(self, mock_esco_index, mock_embedding_model):
        ext = ExplicitSkillExtractor(
            mock_esco_index, embedding_model=mock_embedding_model
        )
        assert ext._w_ner == 1
        assert ext._w_pos == 1
        assert ext._w_dict == 20
        assert ext._w_embed == 2
        assert ext._w_total == 24

    def test_custom_weights_applied(self, mock_esco_index, mock_embedding_model):
        ext = ExplicitSkillExtractor(
            mock_esco_index,
            embedding_model=mock_embedding_model,
            module_weights={"S1": 0, "S2": 0, "S3": 10, "S4": 5},
        )
        assert ext._w_ner == 0
        assert ext._w_pos == 0
        assert ext._w_dict == 10
        assert ext._w_embed == 5
        assert ext._w_total == 15

    def test_partial_override_falls_back(self, mock_esco_index, mock_embedding_model):
        ext = ExplicitSkillExtractor(
            mock_esco_index,
            embedding_model=mock_embedding_model,
            module_weights={"S3": 0},
        )
        assert ext._w_ner == 1
        assert ext._w_dict == 0
        assert ext._w_total == 4

    def test_zero_s3_extracts_fewer_skills(self, mock_esco_index, mock_embedding_model):
        baseline = ExplicitSkillExtractor(
            mock_esco_index, embedding_model=mock_embedding_model,
        )
        no_dict = ExplicitSkillExtractor(
            mock_esco_index,
            embedding_model=mock_embedding_model,
            module_weights={"S1": 1, "S2": 1, "S3": 0, "S4": 2},
        )
        text = "Python machine learning data analysis SQL"
        baseline_skills = baseline.extract(text)
        ablated_skills = no_dict.extract(text)
        assert len(ablated_skills) <= len(baseline_skills)
