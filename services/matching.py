"""JD extraction -> vector recall -> per-candidate LLM match analysis."""

from __future__ import annotations

from collections.abc import Callable

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
        on_step: Callable[[str], None] | None = None,
    ) -> tuple[JDRequirements, list[CandidateMatchResult]]:
        def emit(msg: str) -> None:
            if on_step:
                on_step(msg)

        top_k = max(1, jd_retrieval_top_k)
        top_n = max(1, match_candidate_top_n)
        emit("步骤 1/4：LLM 抽取 JD 结构化需求（岗位、技能、经验等）…")
        jd = self.extract_jd(jd_text)
        emit(
            f"JD 抽取完成：岗位「{jd.role_title or '—'}」，"
            f"必备技能 {len(jd.required_skills)} 项，加分技能 {len(jd.preferred_skills)} 项"
        )
        query = jd.as_retrieval_query()
        if not query.strip():
            query = jd_text[:2000]
        emit(f"检索 query 已组装（约 {len(query)} 字）")

        candidate_ids: list[str] = []
        emit(f"步骤 2/4：召回候选人（向量 top_k={top_k}，最终最多分析 {top_n} 人）…")
        if self._vectors:
            try:
                emit("正在 Chroma 向量相似度检索…")
                raw = self._vectors.similarity_search_with_scores(
                    query,
                    k=top_k,
                )
                candidate_ids = _dedupe_candidates_by_best_score(raw)
                emit(f"向量召回：去重后 {len(candidate_ids)} 位候选人")
            except Exception as exc:  # noqa: BLE001
                emit(f"向量检索异常，将尝试关键词/全库兜底：{type(exc).__name__}: {exc}")
                candidate_ids = []
        else:
            emit("向量库不可用，跳过向量召回")

        if not candidate_ids and keyword_fallback_query:
            emit("向量无结果：使用 JD 片段做 SQLite 关键词召回…")
            records = self._candidates.keyword_search(
                keyword_fallback_query, limit=max(top_k, 30)
            )
            candidate_ids = [r.candidate_id for r in records]
            emit(f"关键词召回：{len(candidate_ids)} 位候选人")

        if not candidate_ids:
            emit("仍无候选：从人才库按时间倒序取一批…")
            records = self._candidates.list_all(limit=max(top_n, 100))
            candidate_ids = [r.candidate_id for r in records]
            emit(f"全库兜底：{len(candidate_ids)} 位候选人")

        top_ids = candidate_ids[:top_n]
        emit(f"步骤 3/4：加载 {len(top_ids)} 位候选人完整画像…")
        by_id = self._candidates.get_many(top_ids)
        ordered = [by_id[i] for i in top_ids if i in by_id]
        emit(f"已加载 {len(ordered)} 条记录，开始逐人 LLM 匹配分析")

        results: list[CandidateMatchResult] = []
        emit("步骤 4/4：对每位候选人调用匹配分析链（可解释分数与证据）…")
        for i, rec in enumerate(ordered, start=1):
            label = rec.profile.name or rec.candidate_id[:8] + "…"
            emit(f"  ({i}/{len(ordered)}) 分析：{label} …")
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
            emit(
                f"  ({i}/{len(ordered)}) 完成：总分 {analysis.total_match_score:.1f}"
            )

        results.sort(key=lambda r: r.analysis.total_match_score, reverse=True)
        emit("全部完成：已按总分重新排序")
        return jd, results
