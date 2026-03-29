"""Pydantic models for standardized candidate profiles."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EducationEntry(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field: str | None = None
    start_year: str | None = None
    end_year: str | None = None


class WorkEntry(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None


class ProjectEntry(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """Structured profile produced by the resume parsing LLM."""

    name: str | None = None
    email: str | None = None
    phone: str | None = Field(
        default=None,
        description="Mobile phone number as shown on resume (do not invent)",
    )
    wechat: str | None = Field(
        default=None,
        description="WeChat ID / 微信号 if explicitly stated",
    )
    location: str | None = Field(
        default=None,
        description="Current or preferred city/region, 地理地点",
    )
    job_intent: str | None = Field(
        default=None,
        description="Job search intent: role type, industry, full-time/part-time, 求职意向",
    )
    birth_year: int | None = Field(
        default=None,
        description="Birth year as integer if inferable from resume (出生年份); null if unknown",
    )
    latest_graduation_date: str | None = Field(
        default=None,
        description="Most recent graduation time as written (e.g. 2024-06 or 2024年6月), 最近毕业时间",
    )
    highest_education: str | None = Field(
        default=None,
        description="Highest completed degree level (最高学历), e.g. 博士/硕士/本科/专科; may include major if concise",
    )
    education: list[EducationEntry] = Field(default_factory=list)
    work_experience: list[WorkEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_of_experience: float | None = Field(
        default=None, description="Estimated total years of professional experience"
    )
    summary: str = Field(default="", description="Concise candidate summary for HR")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Potential concerns e.g. job hopping, gaps, unclear dates",
    )
    embedding_ready_text: str = Field(
        default="",
        description="Clean narrative for chunking and embeddings",
    )


class CandidateRecord(BaseModel):
    """Full persisted candidate row including identifiers and raw text."""

    candidate_id: str
    profile: CandidateProfile
    raw_text: str
    source_file: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_chroma_documents(self) -> list[tuple[str, dict[str, Any]]]:
        """Return list of (text, metadata) pairs for vector indexing."""
        meta_base: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "name": self.profile.name or "",
            "source_file": self.source_file,
            "years_of_experience": self.profile.years_of_experience or 0.0,
            "skills": ",".join(self.profile.skills[:40]),
            "education_level": (
                self.profile.highest_education
                or (
                    self.profile.education[0].degree
                    if self.profile.education
                    else ""
                )
            ),
            "location": self.profile.location or "",
            "job_intent": (self.profile.job_intent or "")[:500],
        }
        chunks: list[tuple[str, dict[str, Any]]] = []

        contact_bits = " | ".join(
            filter(
                None,
                [
                    f"手机: {self.profile.phone}" if self.profile.phone else None,
                    f"微信: {self.profile.wechat}" if self.profile.wechat else None,
                    f"地点: {self.profile.location}" if self.profile.location else None,
                    f"求职意向: {self.profile.job_intent}" if self.profile.job_intent else None,
                    f"出生年份: {self.profile.birth_year}" if self.profile.birth_year else None,
                    f"最近毕业: {self.profile.latest_graduation_date}"
                    if self.profile.latest_graduation_date
                    else None,
                    f"最高学历: {self.profile.highest_education}"
                    if self.profile.highest_education
                    else None,
                ],
            )
        )
        if contact_bits.strip():
            chunks.append(
                (
                    f"[Contact & intent]\n{contact_bits}",
                    {**meta_base, "chunk_type": "contact_intent"},
                )
            )

        if self.profile.summary:
            chunks.append(
                (
                    f"[Summary]\n{self.profile.summary}",
                    {**meta_base, "chunk_type": "summary"},
                )
            )
        for w in self.profile.work_experience:
            line = " | ".join(
                filter(
                    None,
                    [
                        w.company,
                        w.title,
                        w.start_date,
                        w.end_date,
                        w.description,
                    ],
                )
            )
            if line.strip():
                chunks.append(
                    (
                        f"[Work]\n{line}",
                        {**meta_base, "chunk_type": "work"},
                    )
                )
        for p in self.profile.projects:
            tech = ", ".join(p.technologies) if p.technologies else ""
            body = " | ".join(filter(None, [p.name, p.description, tech]))
            if body.strip():
                chunks.append(
                    (
                        f"[Project]\n{body}",
                        {**meta_base, "chunk_type": "project"},
                    )
                )
        if self.profile.skills:
            chunks.append(
                (
                    f"[Skills]\n{', '.join(self.profile.skills)}",
                    {**meta_base, "chunk_type": "skills"},
                )
            )
        if self.profile.embedding_ready_text.strip():
            chunks.append(
                (
                    self.profile.embedding_ready_text.strip(),
                    {**meta_base, "chunk_type": "embedding_ready"},
                )
            )
        # Raw text chunks (lightweight split) — repository may further split
        if self.raw_text.strip():
            chunks.append(
                (
                    f"[Resume excerpt]\n{self.raw_text[:2000]}",
                    {**meta_base, "chunk_type": "raw_excerpt"},
                )
            )
        return chunks
