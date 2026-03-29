"""Optional multimodal fallback: image -> resume plain text."""

from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts.templates import VISION_RESUME_SYSTEM, VISION_RESUME_USER
from utils.config import AppConfig


def _image_mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def extract_text_from_image_vision(image_path: str, config: AppConfig) -> str:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    mime = _image_mime(image_path)
    url = f"data:{mime};base64,{b64}"
    llm = ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        temperature=0,
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": VISION_RESUME_USER},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    resp = llm.invoke(
        [
            SystemMessage(content=VISION_RESUME_SYSTEM),
            msg,
        ]
    )
    return (resp.content or "").strip()
