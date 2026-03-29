"""Interactive Q&A over JD matching results."""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate

from chains.llm_factory import chat_llm
from schemas.jd import JDRequirements
from schemas.match import CandidateMatchResult
from utils.config import AppConfig


class ChatTurn(TypedDict):
    role: str
    content: str


_SYSTEM = """You are an HR copilot answering questions based on one JD and its current ranked matching results.
Rules:
- Use only provided JD text, structured JD fields, and match result data.
- Do not fabricate candidate facts beyond provided evidence.
- If information is missing, state uncertainty and suggest what to verify.
- Provide concise, practical HR-oriented answers.
- Answer in the same language as the user.
"""

_USER = """Original JD text:
{jd_text}

Structured JD (JSON):
{jd_structured_json}

Current ranked match results (JSON):
{match_results_json}

Recent chat history:
{history_text}

User question:
{question}
"""


def _history_to_text(history: list[ChatTurn], limit: int = 12) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{h['role']}: {h['content']}" for h in history[-limit:])


def answer_jd_match_question(
    *,
    jd_text: str,
    jd_structured: JDRequirements,
    results: list[CandidateMatchResult],
    question: str,
    config: AppConfig,
    history: list[ChatTurn] | None = None,
) -> str:
    llm = chat_llm(config, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _USER)])
    chain = prompt | llm

    compact_results = [
        {
            "rank": idx + 1,
            "candidate_id": r.candidate_id,
            "name": r.name,
            "analysis": r.analysis.model_dump(),
        }
        for idx, r in enumerate(results)
    ]
    payload = {
        "jd_text": jd_text[:6000],
        "jd_structured_json": json.dumps(
            jd_structured.model_dump(), ensure_ascii=False, indent=2
        ),
        "match_results_json": json.dumps(compact_results, ensure_ascii=False, indent=2),
        "history_text": _history_to_text(history or []),
        "question": question,
    }
    out = chain.invoke(payload)
    return str(getattr(out, "content", out)).strip()
