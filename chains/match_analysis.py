"""LangChain: single candidate + JD -> MatchAnalysis."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from chains.llm_factory import chat_llm
from prompts.templates import MATCH_ANALYSIS_SYSTEM, MATCH_ANALYSIS_USER
from schemas.candidate import CandidateRecord
from schemas.jd import JDRequirements
from schemas.match import MatchAnalysis
from utils.config import AppConfig


def build_match_analysis_runnable(config: AppConfig):
    llm = chat_llm(config)
    structured = llm.with_structured_output(MatchAnalysis)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MATCH_ANALYSIS_SYSTEM),
            ("human", MATCH_ANALYSIS_USER),
        ]
    )
    return prompt | structured


def analyze_match(
    jd: JDRequirements,
    candidate: CandidateRecord,
    config: AppConfig,
) -> MatchAnalysis:
    chain = build_match_analysis_runnable(config)
    jd_json = json.dumps(jd.model_dump(), ensure_ascii=False, indent=2)
    cand = candidate.profile.model_dump()
    candidate_json = json.dumps(cand, ensure_ascii=False, indent=2)
    excerpt = candidate.raw_text[:6000]
    out = chain.invoke(
        {
            "jd_json": jd_json,
            "candidate_json": candidate_json,
            "raw_excerpt": excerpt,
        }
    )
    if isinstance(out, dict):
        return MatchAnalysis.model_validate(out)
    return out
