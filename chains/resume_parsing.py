"""LangChain: resume text -> CandidateProfile."""

from __future__ import annotations

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from chains.llm_factory import chat_llm
from prompts.templates import RESUME_PARSE_SYSTEM, RESUME_PARSE_USER
from schemas.candidate import CandidateProfile
from utils.config import AppConfig


def build_resume_parsing_runnable(config: AppConfig):
    llm = chat_llm(config)
    structured = llm.with_structured_output(CandidateProfile)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESUME_PARSE_SYSTEM),
            ("human", RESUME_PARSE_USER),
        ]
    )
    return prompt | structured


def parse_resume_text(raw_text: str, config: AppConfig) -> CandidateProfile:
    chain = build_resume_parsing_runnable(config)
    now = datetime.now()
    out = chain.invoke(
        {
            "raw_text": raw_text,
            "current_date": now.strftime("%Y-%m-%d"),
            "current_year": now.year,
        }
    )
    if isinstance(out, dict):
        return CandidateProfile.model_validate(out)
    return out
