"""Structured job description requirements."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JDRequirements(BaseModel):
    """JD normalized for retrieval and matching."""

    role_title: str | None = Field(default=None, description="Target role title")
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float | None = Field(
        default=None, description="Minimum years if stated"
    )
    education_requirement: str | None = Field(
        default=None, description="Degree level or field if stated"
    )
    domain_keywords: list[str] = Field(default_factory=list)
    core_responsibilities: list[str] = Field(default_factory=list)

    def as_retrieval_query(self) -> str:
        """Single string for embedding similarity search."""
        parts = [
            self.role_title or "",
            "Required: " + ", ".join(self.required_skills),
            "Preferred: " + ", ".join(self.preferred_skills),
            "Domain: " + ", ".join(self.domain_keywords),
            "Responsibilities: " + "; ".join(self.core_responsibilities[:12]),
        ]
        return "\n".join(p for p in parts if p.strip())
