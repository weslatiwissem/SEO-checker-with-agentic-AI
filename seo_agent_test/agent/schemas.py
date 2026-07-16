"""
Validation layer. LLM-generated JSON is untrusted-shaped by default -- this
enforces the contract so a malformed synthesizer/critic output fails loudly
and predictably instead of breaking downstream consumers (PDF export, DB
storage, CLI rendering).
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError


class Finding(BaseModel):
    severity: str = Field(pattern="^(good|warning|critical)$")
    issue: str
    recommendation: str = ""


class Category(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    findings: list[Finding] = []


class Report(BaseModel):
    url: str
    overall_score: float = Field(ge=0, le=100)
    grade: str = Field(pattern="^[A-F]$")
    summary: str
    categories: list[Category]
    quick_wins: list[str] = []
    data_limitations: str = ""
    trend: dict | None = None


def validate_report(data: dict) -> dict:
    """Raise ValidationError with a clear message if the shape is wrong;
    otherwise return the normalized dict."""
    report = Report(**data)
    return report.model_dump()


__all__ = ["Report", "Category", "Finding", "validate_report", "ValidationError"]
