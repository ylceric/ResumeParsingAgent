"""Candidate-focused chat assistant for HR Q&A."""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate

from chains.llm_factory import chat_llm
from schemas.candidate import CandidateRecord
from utils.config import AppConfig


class ChatTurn(TypedDict):
    role: str
    content: str


_CHAT_SYSTEM = """You are an HR copilot assistant answering questions about ONE selected candidate.
Rules:
- Ground answers strictly in the provided candidate profile and extracted resume text.
- If evidence is missing, clearly state uncertainty and suggest what to verify in interview.
- Be concise and practical for HR screening.
- Answer in the same language as the user's question.
- Do not fabricate dates, companies, skills, or certificates.
"""

_CHAT_USER = """Candidate context (JSON):
{candidate_json}

Extracted resume text (may be truncated):
{raw_text}

Recent conversation history:
{history_text}

User question:
{question}
"""


def _history_to_text(history: list[ChatTurn], limit: int = 10) -> str:
    if not history:
        return "(none)"
    clipped = history[-limit:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in clipped)


def answer_candidate_question(
    candidate: CandidateRecord,
    question: str,
    config: AppConfig,
    history: list[ChatTurn] | None = None,
) -> str:
    """Answer one question for the selected candidate."""
    llm = chat_llm(config, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _CHAT_SYSTEM),
            ("human", _CHAT_USER),
        ]
    )
    chain = prompt | llm
    payload = {
        "candidate_json": json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "source_file": candidate.source_file,
                "profile": candidate.profile.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "raw_text": (candidate.raw_text or "")[:12000],
        "history_text": _history_to_text(history or []),
        "question": question,
    }
    out = chain.invoke(payload)
    return str(getattr(out, "content", out)).strip()
