"""Orchestrate parse -> LLM profile -> SQLite + Chroma."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from chains.resume_parsing import parse_resume_text
from repositories.candidate_repository import CandidateRepository
from repositories.vector_repository import VectorRepository
from schemas.candidate import CandidateProfile
from services.document_parser import ParseOutcome, extract_text
from utils.config import UPLOADS_DIR, AppConfig, ensure_data_dirs


@dataclass
class IngestResult:
    source_file: str
    candidate_id: str | None
    success: bool
    message: str
    profile: CandidateProfile | None = None
    parse_outcome: ParseOutcome | None = None


class IngestionService:
    def __init__(
        self,
        config: AppConfig,
        candidate_repo: CandidateRepository,
        vector_repo: VectorRepository | None,
    ) -> None:
        self._config = config
        self._candidates = candidate_repo
        self._vectors = vector_repo

    def ingest_file(self, uploaded_path: str, original_name: str) -> IngestResult:
        ensure_data_dirs()
        parse_outcome = extract_text(uploaded_path, self._config)
        if parse_outcome.error and not parse_outcome.text.strip():
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message=parse_outcome.error or "Parse failed",
                parse_outcome=parse_outcome,
            )

        if not self._config.openai_api_key:
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message="OPENAI_API_KEY is not set; cannot parse resume with LLM",
                parse_outcome=parse_outcome,
            )

        try:
            profile = parse_resume_text(parse_outcome.text, self._config)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message=f"LLM parsing failed: {type(exc).__name__}: {exc}",
                parse_outcome=parse_outcome,
            )

        cid = str(uuid.uuid4())
        dest_dir = UPLOADS_DIR / cid
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / Path(original_name).name
        try:
            shutil.copy2(uploaded_path, dest_path)
        except Exception:
            dest_path = Path(uploaded_path)

        record = self._candidates.insert(
            profile=profile,
            raw_text=parse_outcome.text,
            source_file=original_name,
            candidate_id=cid,
        )

        indexed = 0
        if self._vectors:
            try:
                indexed = self._vectors.index_candidate(record)
            except Exception as exc:  # noqa: BLE001
                return IngestResult(
                    source_file=original_name,
                    candidate_id=cid,
                    success=True,
                    message=f"Saved to DB but vector index failed: {exc}",
                    profile=profile,
                    parse_outcome=parse_outcome,
                )

        warn = []
        if parse_outcome.error:
            warn.append(parse_outcome.error)
        msg = f"Ingested OK; vector chunks: {indexed}"
        if warn:
            msg += " | Warnings: " + "; ".join(warn)
        return IngestResult(
            source_file=original_name,
            candidate_id=cid,
            success=True,
            message=msg,
            profile=profile,
            parse_outcome=parse_outcome,
        )
