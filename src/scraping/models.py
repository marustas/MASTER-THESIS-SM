from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


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


class CourseModule(BaseModel):
    """A single course or module within a programme curriculum."""

    programme_name: str
    institution: str
    module_name: str
    credits_ects: Optional[int] = None
    description: Optional[str] = None
    semester: Optional[int] = None
    module_type: Optional[str] = None  # compulsory / elective
