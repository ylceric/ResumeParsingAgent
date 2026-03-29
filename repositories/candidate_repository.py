"""SQLite persistence for candidate records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable

from schemas.candidate import (
    CandidateProfile,
    CandidateRecord,
    EducationEntry,
    ProjectEntry,
    WorkEntry,
)

_EXTRA_COLUMNS: dict[str, str] = {
    "wechat": "TEXT",
    "location": "TEXT",
    "job_intent": "TEXT",
    "birth_year": "INTEGER",
    "latest_graduation_date": "TEXT",
    "highest_education": "TEXT",
}


def _ensure_extra_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(candidates)").fetchall()
    existing = {r[1] for r in rows}
    for col, sql_type in _EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {sql_type}")
    conn.commit()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            wechat TEXT,
            location TEXT,
            job_intent TEXT,
            birth_year INTEGER,
            latest_graduation_date TEXT,
            highest_education TEXT,
            education_json TEXT NOT NULL DEFAULT '[]',
            work_experience_json TEXT NOT NULL DEFAULT '[]',
            projects_json TEXT NOT NULL DEFAULT '[]',
            skills_json TEXT NOT NULL DEFAULT '[]',
            years_of_experience REAL,
            summary TEXT NOT NULL DEFAULT '',
            risk_flags_json TEXT NOT NULL DEFAULT '[]',
            embedding_ready_text TEXT NOT NULL DEFAULT '',
            raw_text TEXT NOT NULL DEFAULT '',
            source_file TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    _ensure_extra_columns(conn)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidates_created ON candidates(created_at)"
    )
    conn.commit()


class CandidateRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            _init_schema(conn)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert(
        self,
        profile: CandidateProfile,
        raw_text: str,
        source_file: str,
        candidate_id: str | None = None,
    ) -> CandidateRecord:
        cid = candidate_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id, name, email, phone, wechat, location, job_intent,
                    birth_year, latest_graduation_date, highest_education,
                    education_json, work_experience_json, projects_json, skills_json,
                    years_of_experience, summary, risk_flags_json, embedding_ready_text,
                    raw_text, source_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    profile.name,
                    profile.email,
                    profile.phone,
                    profile.wechat,
                    profile.location,
                    profile.job_intent,
                    profile.birth_year,
                    profile.latest_graduation_date,
                    profile.highest_education,
                    json.dumps([e.model_dump() for e in profile.education], ensure_ascii=False),
                    json.dumps(
                        [w.model_dump() for w in profile.work_experience],
                        ensure_ascii=False,
                    ),
                    json.dumps([p.model_dump() for p in profile.projects], ensure_ascii=False),
                    json.dumps(profile.skills, ensure_ascii=False),
                    profile.years_of_experience,
                    profile.summary,
                    json.dumps(profile.risk_flags, ensure_ascii=False),
                    profile.embedding_ready_text,
                    raw_text,
                    source_file,
                    now,
                ),
            )
            conn.commit()
        return self.get_by_id(cid)

    def get_by_id(self, candidate_id: str) -> CandidateRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return self._row_to_record(row)

    def list_all(self, limit: int = 500) -> list[CandidateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def keyword_search(self, query: str, limit: int = 100) -> list[CandidateRecord]:
        q = f"%{query.strip()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM candidates
                WHERE summary LIKE ? OR raw_text LIKE ? OR name LIKE ?
                   OR skills_json LIKE ? OR email LIKE ? OR phone LIKE ?
                   OR IFNULL(wechat, '') LIKE ? OR IFNULL(location, '') LIKE ?
                   OR IFNULL(job_intent, '') LIKE ?
                   OR IFNULL(highest_education, '') LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (q, q, q, q, q, q, q, q, q, q, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_many(self, candidate_ids: Iterable[str]) -> dict[str, CandidateRecord]:
        ids = list(dict.fromkeys(candidate_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM candidates WHERE candidate_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {r["candidate_id"]: self._row_to_record(r) for r in rows}

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM candidates").fetchone()
        return int(row["c"]) if row else 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CandidateRecord:
        education = [
            EducationEntry(**e) for e in json.loads(row["education_json"] or "[]")
        ]
        work = [
            WorkEntry(**w) for w in json.loads(row["work_experience_json"] or "[]")
        ]
        projects = [
            ProjectEntry(**p) for p in json.loads(row["projects_json"] or "[]")
        ]
        skills = json.loads(row["skills_json"] or "[]")
        risk_flags = json.loads(row["risk_flags_json"] or "[]")
        profile = CandidateProfile(
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            wechat=row["wechat"],
            location=row["location"],
            job_intent=row["job_intent"],
            birth_year=row["birth_year"],
            latest_graduation_date=row["latest_graduation_date"],
            highest_education=row["highest_education"],
            education=education,
            work_experience=work,
            projects=projects,
            skills=skills,
            years_of_experience=row["years_of_experience"],
            summary=row["summary"] or "",
            risk_flags=risk_flags,
            embedding_ready_text=row["embedding_ready_text"] or "",
        )
        created_raw = row["created_at"]
        if isinstance(created_raw, str) and created_raw.endswith("Z"):
            created_raw = created_raw[:-1] + "+00:00"
        created = datetime.fromisoformat(created_raw)
        return CandidateRecord(
            candidate_id=row["candidate_id"],
            profile=profile,
            raw_text=row["raw_text"] or "",
            source_file=row["source_file"] or "",
            created_at=created,
        )
