"""Orchestrate parse -> LLM profile -> SQLite + Chroma."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
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

    def ingest_file(
        self,
        uploaded_path: str,
        original_name: str,
        *,
        on_step: Callable[[str], None] | None = None,
    ) -> IngestResult:
        def emit(msg: str) -> None:
            if on_step:
                on_step(msg)

        ensure_data_dirs()
        emit(f"「{original_name}」开始提取文本…")
        parse_outcome = extract_text(uploaded_path, self._config)
        nchars = len(parse_outcome.text or "")
        if parse_outcome.error:
            emit(
                f"「{original_name}」提取结束：方式={parse_outcome.method}，"
                f"字符数≈{nchars}，提示：{parse_outcome.error}"
            )
        else:
            emit(
                f"「{original_name}」提取完成：方式={parse_outcome.method}，字符数≈{nchars}"
            )
        if parse_outcome.error and not parse_outcome.text.strip():
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message=parse_outcome.error or "Parse failed",
                parse_outcome=parse_outcome,
            )

        if not self._config.openai_api_key:
            emit("跳过 LLM：未配置 OPENAI_API_KEY")
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message="OPENAI_API_KEY is not set; cannot parse resume with LLM",
                parse_outcome=parse_outcome,
            )

        emit("调用 LLM 解析为结构化候选人画像…")
        try:
            profile = parse_resume_text(parse_outcome.text, self._config)
        except Exception as exc:  # noqa: BLE001
            emit(f"LLM 解析失败：{type(exc).__name__}: {exc}")
            return IngestResult(
                source_file=original_name,
                candidate_id=None,
                success=False,
                message=f"LLM parsing failed: {type(exc).__name__}: {exc}",
                parse_outcome=parse_outcome,
            )

        emit(f"画像解析成功：{profile.name or '（未识别姓名）'}")
        cid = str(uuid.uuid4())
        dest_dir = UPLOADS_DIR / cid
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / Path(original_name).name
        try:
            shutil.copy2(uploaded_path, dest_path)
        except Exception:
            dest_path = Path(uploaded_path)

        emit(f"写入 SQLite（candidate_id={cid[:8]}…）")
        record = self._candidates.insert(
            profile=profile,
            raw_text=parse_outcome.text,
            source_file=original_name,
            candidate_id=cid,
        )

        indexed = 0
        if self._vectors:
            emit("写入 Chroma 向量索引（分块 + embedding）…")
            try:
                indexed = self._vectors.index_candidate(record)
                emit(f"向量索引完成：共 {indexed} 条 chunk")
            except Exception as exc:  # noqa: BLE001
                emit(f"向量索引失败：{type(exc).__name__}: {exc}")
                return IngestResult(
                    source_file=original_name,
                    candidate_id=cid,
                    success=True,
                    message=f"Saved to DB but vector index failed: {exc}",
                    profile=profile,
                    parse_outcome=parse_outcome,
                )
        else:
            emit("跳过向量索引：向量库未就绪或未配置 API Key")

        warn = []
        if parse_outcome.error:
            warn.append(parse_outcome.error)
        msg = f"Ingested OK; vector chunks: {indexed}"
        if warn:
            msg += " | Warnings: " + "; ".join(warn)
        emit("本文件处理流程结束（成功）")
        return IngestResult(
            source_file=original_name,
            candidate_id=cid,
            success=True,
            message=msg,
            profile=profile,
            parse_outcome=parse_outcome,
        )
