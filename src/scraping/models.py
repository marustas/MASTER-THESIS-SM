from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Programme(BaseModel):
    """A study programme entry scraped from LAMA BPO and/or university websites."""

    # Core fields from LAMA BPO registry
    name: str
    institution: str
    city: Optional[str] = None
    field_group: Optional[str] = None  # Krypčių grupė
    field: Optional[str] = None  # Kryptis
    study_mode: Optional[str] = None  # Studijų forma (full-time / part-time)
    degree: Optional[str] = None  # e.g. "Bakalauras"
    duration_years: Optional[float] = None
    credits_ects: Optional[int] = None
    language: Optional[str] = None
    state_funded_places: Optional[int] = None
    fee_funded_places: Optional[int] = None

    # Links
    lama_bpo_url: Optional[str] = None
    university_url: Optional[str] = None

    # Descriptions (dual representation per thesis methodology)
    brief_description: Optional[str] = None       # From LAMA BPO registry
    extended_description: Optional[str] = None    # Scraped from university website

    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "lama_bpo"


class JobAd(BaseModel):
    """A job advertisement scraped from an online job platform."""

    # Core fields (per thesis methodology Step 2)
    job_title: str
    company: Optional[str] = None
    description: Optional[str] = None
    required_skills: list[str] = Field(default_factory=list)  # explicitly listed skills
    employer_sector: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    employment_type: Optional[str] = None   # full-time / part-time / contract
    remote: Optional[bool] = None
    posting_date: Optional[str] = None      # raw string; normalised in preprocessing

    # Source metadata
    url: Optional[str] = None
    source: str                             # e.g. "cvbankas", "linkedin"
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class CourseModule(BaseModel):
    """A single course or module within a programme curriculum."""

    programme_name: str
    institution: str
    module_name: str
    credits_ects: Optional[int] = None
    description: Optional[str] = None
    semester: Optional[int] = None
    module_type: Optional[str] = None  # compulsory / elective
