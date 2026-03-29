"""Lightweight smoke checks (no API calls)."""

from __future__ import annotations

import unittest

from utils.bootstrap import ensure_project_on_syspath

ensure_project_on_syspath()

from repositories.candidate_repository import CandidateRepository
from schemas.jd import JDRequirements
from utils.config import SQLITE_PATH, ensure_data_dirs


class TestSmoke(unittest.TestCase):
    def test_sqlite_repository_init(self) -> None:
        ensure_data_dirs()
        repo = CandidateRepository(SQLITE_PATH)
        self.assertIsInstance(repo.count(), int)

    def test_jd_retrieval_query(self) -> None:
        jd = JDRequirements(
            role_title="Backend Engineer",
            required_skills=["Python"],
            preferred_skills=["Kubernetes"],
            domain_keywords=["fintech"],
            core_responsibilities=["API design"],
        )
        q = jd.as_retrieval_query()
        self.assertIn("Python", q)


if __name__ == "__main__":
    unittest.main()
