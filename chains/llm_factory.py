"""Shared LLM construction."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from utils.config import AppConfig


def chat_llm(config: AppConfig, temperature: float = 0.1) -> ChatOpenAI:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        temperature=temperature,
    )
