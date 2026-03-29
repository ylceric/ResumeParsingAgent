"""JD extraction -> vector recall -> per-candidate LLM match analysis."""

from __future__ import annotations

from chains.jd_extraction import extract_jd_requirements
from chains.match_analysis import analyze_match
from repositories.candidate_repository import CandidateRepository
from repositories.vector_repository import VectorRepository
from schemas.jd import JDRequirements
from schemas.match import CandidateMatchResult, MatchAnalysis
from utils.config import (
    DEFAULT_JD_RETRIEVAL_TOP_K,
    DEFAULT_MATCH_CANDIDATE_TOP_N,
    AppConfig,
)


def _dedupe_candidates_by_best_score(
    scored_docs: list[tuple[object, float]],
) -> list[str]:
    """Return candidate_id list ordered by best (lowest) distance score first."""
    best: dict[str, float] = {}
    for doc, score in scored_docs:
        meta = getattr(doc, "metadata", {}) or {}
        cid = meta.get("candidate_id")
        if not cid:
            continue
        prev = best.get(cid)
        if prev is None or score < prev:
            best[cid] = score
    return sorted(best.keys(), key=lambda k: best[k])


class MatchingService:
    def __init__(
        self,
        config: AppConfig,
        candidate_repo: CandidateRepository,
        vector_repo: VectorRepository | None,
    ) -> None:
        self._config = config
        self._candidates = candidate_repo
        self._vectors = vector_repo

    def extract_jd(self, jd_text: str) -> JDRequirements:
        if not self._config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return extract_jd_requirements(jd_text, self._config)

    def match(
        self,
        jd_text: str,
        keyword_fallback_query: str | None = None,
        *,
        jd_retrieval_top_k: int = DEFAULT_JD_RETRIEVAL_TOP_K,
        match_candidate_top_n: int = DEFAULT_MATCH_CANDIDATE_TOP_N,
    ) -> tuple[JDRequirements, list[CandidateMatchResult]]:
        top_k = max(1, jd_retrieval_top_k)
        top_n = max(1, match_candidate_top_n)
        jd = self.extract_jd(jd_text)
        query = jd.as_retrieval_query()
        if not query.strip():
            query = jd_text[:2000]

        candidate_ids: list[str] = []
        if self._vectors:
            try:
                raw = self._vectors.similarity_search_with_scores(
                    query,
                    k=top_k,
                )
                candidate_ids = _dedupe_candidates_by_best_score(raw)
            except Exception:
                candidate_ids = []

        if not candidate_ids and keyword_fallback_query:
            records = self._candidates.keyword_search(
                keyword_fallback_query, limit=max(top_k, 30)
            )
            candidate_ids = [r.candidate_id for r in records]

        if not candidate_ids:
            records = self._candidates.list_all(limit=max(top_n, 100))
            candidate_ids = [r.candidate_id for r in records]

        top_ids = candidate_ids[:top_n]
        by_id = self._candidates.get_many(top_ids)
        ordered = [by_id[i] for i in top_ids if i in by_id]

        results: list[CandidateMatchResult] = []
        for rec in ordered:
            try:
                analysis = analyze_match(jd, rec, self._config)
            except Exception as exc:  # noqa: BLE001
                analysis = MatchAnalysis(
                    total_match_score=0.0,
                    skill_match_score=0.0,
                    experience_match_score=0.0,
                    project_relevance_score=0.0,
                    domain_relevance_score=0.0,
                    matched_evidence=[],
                    missing_or_weak_evidence=[f"Match analysis error: {exc}"],
                    strengths=[],
                    concerns=["Could not complete automated match"],
                    interview_questions=[],
                )
            results.append(
                CandidateMatchResult(
                    candidate_id=rec.candidate_id,
                    name=rec.profile.name,
                    analysis=analysis,
                )
            )

        results.sort(key=lambda r: r.analysis.total_match_score, reverse=True)
        return jd, results
