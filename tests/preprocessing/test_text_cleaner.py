"""Tests for LinkedIn boilerplate stripping in text_cleaner.py."""

from __future__ import annotations

import pytest

from src.preprocessing.text_cleaner import strip_linkedin_boilerplate


class TestStripLinkedinBoilerplate:
    def test_removes_about_the_job_header(self):
        text = "About the job\n\nWe need a Python developer."
        result = strip_linkedin_boilerplate(text)
        assert "About the job" not in result
        assert "Python developer" in result

    def test_truncates_at_what_we_offer(self):
        text = (
            "Build scalable APIs with Python.\n"
            "Requirements:\n"
            "3+ years experience.\n"
            "What We Offer\n"
            "Competitive salary.\n"
            "Health insurance.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "scalable APIs" in result
        assert "3+ years" in result
        assert "Competitive salary" not in result
        assert "Health insurance" not in result

    def test_truncates_at_benefits(self):
        text = (
            "Design ML pipelines.\n"
            "Benefits\n"
            "Free lunches and team events.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "ML pipelines" in result
        assert "Free lunches" not in result

    def test_truncates_at_equal_opportunity(self):
        text = (
            "Senior Java developer needed.\n"
            "Equal Opportunity\n"
            "We are committed to diversity.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "Java developer" in result
        assert "diversity" not in result

    def test_truncates_at_we_are_proud(self):
        text = (
            "Full-stack engineer role.\n"
            "We are proud to foster a workplace free from discrimination.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "Full-stack" in result
        assert "discrimination" not in result

    def test_truncates_at_salary_line(self):
        text = (
            "DevOps engineer.\n"
            "Salary: 4000-6000 EUR\n"
            "Apply now.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "DevOps" in result
        assert "4000" not in result

    def test_truncates_at_lithuanian_data_protection(self):
        text = (
            "Backend developer.\n"
            "Siųsdami savo gyvenimo aprašymą\n"
            "sutinkate su duomenų tvarkymu.\n"
        )
        result = strip_linkedin_boilerplate(text)
        assert "Backend" in result
        assert "duomenų" not in result

    def test_preserves_text_without_boilerplate(self):
        text = "Python developer with ML experience.\n3+ years required."
        result = strip_linkedin_boilerplate(text)
        assert result == text

    def test_empty_text(self):
        assert strip_linkedin_boilerplate("") == ""

    def test_only_boilerplate(self):
        text = "About the job\nWhat We Offer\nFree snacks."
        result = strip_linkedin_boilerplate(text)
        assert result.strip() == ""

    def test_case_insensitive(self):
        text = "Code stuff.\nWHAT WE OFFER\nPerks."
        result = strip_linkedin_boilerplate(text)
        assert "Code stuff" in result
        assert "Perks" not in result

    def test_we_offer_variant(self):
        text = "Build APIs.\nWe Offer\nGreat culture.\n"
        result = strip_linkedin_boilerplate(text)
        assert "Build APIs" in result
        assert "Great culture" not in result

    def test_our_offer_variant(self):
        text = "Design systems.\nOur Offer\nRemote work.\n"
        result = strip_linkedin_boilerplate(text)
        assert "Design systems" in result
        assert "Remote work" not in result

    def test_what_youll_get_variant(self):
        text = "Write code.\nWhat you'll get\nStock options.\n"
        result = strip_linkedin_boilerplate(text)
        assert "Write code" in result
        assert "Stock options" not in result

    def test_joinrs_ai_removed(self):
        text = "DevOps role.\nJoinrs AI\nSummary of the opportunity.\n"
        result = strip_linkedin_boilerplate(text)
        assert "DevOps" in result
        assert "Summary" not in result

    def test_does_not_match_inline_offer(self):
        """'we offer' inside a sentence should NOT trigger cutoff."""
        text = "We offer competitive rates and flexible hours.\nRequirements: Python."
        # This has 'We offer' but not on its own line — the regex requires
        # the pattern to be the entire line content (^\s*...\s*$).
        result = strip_linkedin_boilerplate(text)
        assert "competitive rates" in result
        assert "Requirements" in result

    def test_we_ensure_equal_variant(self):
        text = "QA engineer.\nWe ensure equal opportunity to all.\n"
        result = strip_linkedin_boilerplate(text)
        assert "QA engineer" in result
        assert "equal opportunity" not in result

    def test_no_false_positive_on_committed_to_creating(self):
        """'We are committed to creating scalable solutions' is legit content."""
        text = "We are committed to creating scalable data solutions.\nRequirements: Python."
        result = strip_linkedin_boilerplate(text)
        assert "scalable data" in result
        assert "Requirements" in result

    def test_compensation_cutoff(self):
        text = "Data engineer.\nCompensation:\n5000 EUR.\n"
        result = strip_linkedin_boilerplate(text)
        assert "Data engineer" in result
        assert "5000" not in result
