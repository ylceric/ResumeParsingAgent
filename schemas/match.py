"""JD–candidate match analysis schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchAnalysis(BaseModel):
    """Explainable match output for one candidate."""

    total_match_score: float = Field(ge=0.0, le=100.0)
    skill_match_score: float = Field(ge=0.0, le=100.0)
    experience_match_score: float = Field(ge=0.0, le=100.0)
    project_relevance_score: float = Field(ge=0.0, le=100.0)
    domain_relevance_score: float = Field(ge=0.0, le=100.0)
    matched_evidence: list[str] = Field(
        default_factory=list,
        description="Bullet evidence from resume aligned with JD",
    )
    missing_or_weak_evidence: list[str] = Field(
        default_factory=list,
        description="Gaps or weak signals vs JD",
    )
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(
        default_factory=list,
        description="Suggested interview questions for this candidate",
    )


class CandidateMatchResult(BaseModel):
    candidate_id: str
    name: str | None
    analysis: MatchAnalysis
