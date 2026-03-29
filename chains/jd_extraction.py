"""LangChain: JD text -> JDRequirements."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from chains.llm_factory import chat_llm
from prompts.templates import JD_EXTRACTION_SYSTEM, JD_EXTRACTION_USER
from schemas.jd import JDRequirements
from utils.config import AppConfig


def build_jd_extraction_runnable(config: AppConfig):
    llm = chat_llm(config)
    structured = llm.with_structured_output(JDRequirements)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JD_EXTRACTION_SYSTEM),
            ("human", JD_EXTRACTION_USER),
        ]
    )
    return prompt | structured


def extract_jd_requirements(jd_text: str, config: AppConfig) -> JDRequirements:
    chain = build_jd_extraction_runnable(config)
    out = chain.invoke({"jd_text": jd_text})
    if isinstance(out, dict):
        return JDRequirements.model_validate(out)
    return out


def jd_requirements_to_json(req: JDRequirements) -> str:
    return json.dumps(req.model_dump(), ensure_ascii=False, indent=2)
