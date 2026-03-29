"""Chroma vector store for candidate chunk retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from schemas.candidate import CandidateRecord
from utils.config import CHROMA_DIR


COLLECTION_NAME = "candidate_chunks"


class VectorRepository:
    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self._persist = persist_directory or str(CHROMA_DIR)
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=self._persist,
        )

    def index_candidate(self, record: CandidateRecord) -> int:
        """Add chunked documents for one candidate. Returns number of chunks."""
        pairs = record.to_chroma_documents()
        docs: list[Document] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
        )
        for text, meta in pairs:
            if not text or not text.strip():
                continue
            for chunk in splitter.split_text(text.strip()):
                m = dict(meta)
                m["chunk_id"] = str(uuid.uuid4())
                docs.append(Document(page_content=chunk, metadata=m))
        if not docs:
            return 0
        self._store.add_documents(docs)
        return len(docs)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 20,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict,
        )
