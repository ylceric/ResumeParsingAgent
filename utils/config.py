"""Application configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "app.db"

# JD matching page defaults (overridden in Streamlit UI)
DEFAULT_JD_RETRIEVAL_TOP_K = 24
DEFAULT_MATCH_CANDIDATE_TOP_N = 8


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration."""

    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    vision_fallback_enabled: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            vision_fallback_enabled=os.getenv("OPENAI_VISION_FALLBACK", "true").lower()
            in ("1", "true", "yes"),
        )


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_stored_resume_path(candidate_id: str, source_filename: str) -> Path | None:
    """Return path to copied original resume under data/uploads/, or first file in folder."""
    folder = UPLOADS_DIR / candidate_id
    if not folder.is_dir():
        return None
    exact = folder / Path(source_filename).name
    if exact.is_file():
        return exact
    for child in sorted(folder.iterdir()):
        if child.is_file():
            return child
    return None
