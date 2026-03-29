from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Programme(BaseModel):
    """A study programme entry scraped from LAMA BPO and/or university websites."""

    # Core fields from LAMA BPO registry
    name: str
    institution: str
    city: str | None = None
    field_group: str | None = None  # Krypčių grupė
    field: str | None = None  # Kryptis
    study_mode: str | None = None  # Studijų forma (full-time / part-time)
    degree: str | None = None  # e.g. "Bakalauras"
    duration_years: float | None = None
    credits_ects: int | None = None
    language: str | None = None
    state_funded_places: int | None = None
    fee_funded_places: int | None = None

    # Links
    lama_bpo_url: str | None = None
    university_url: str | None = None
    aikos_url: str | None = None

    # Descriptions (dual representation per thesis methodology)
    brief_description: str | None = None       # From LAMA BPO registry
    extended_description: str | None = None    # Scraped from university website

    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "lama_bpo"


class JobAd(BaseModel):
    """A job advertisement scraped from an online job platform."""

    # Core fields (per thesis methodology Step 2)
    job_title: str
    description: str | None = None
    required_skills: list[str] = Field(default_factory=list)  # explicitly listed skills
    employer_sector: str | None = None
    location: str | None = None
    country: str | None = None
    employment_type: str | None = None   # full-time / part-time / contract
    remote: bool | None = None
    posting_date: str | None = None      # raw string; normalised in preprocessing

    # Source metadata
    url: str | None = None
    source: str                             # e.g. "cvbankas", "linkedin"
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class CourseModule(BaseModel):
    """A single course or module within a programme curriculum."""

    programme_name: str
    institution: str
    module_name: str
    credits_ects: int | None = None
    description: str | None = None
    semester: int | None = None
    module_type: str | None = None  # compulsory / elective
