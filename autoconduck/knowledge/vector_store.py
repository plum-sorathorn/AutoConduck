"""LanceDB vector store for repository knowledge and API contracts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from autoconduck._compat import lancedb_connect
from autoconduck.knowledge.extractor import _embed_text, extract_repo_symbols
from autoconduck.knowledge.models import CodeChunk, QueryResult

logger = logging.getLogger(__name__)


class KnowledgeVectorStore:
    """Selective LanceDB vector index and context snippet retriever."""

    def __init__(self, db_uri: str = ":memory:", table_name: str = "repo_knowledge") -> None:
        self.db_uri = db_uri
        self.table_name = table_name
        self.db = lancedb_connect(db_uri)
        self._table = None
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                self._table = self.db.create_table(self.table_name, data=[])
        except Exception as exc:
            logger.warning("Failed to initialize LanceDB table: %s", exc)

    def index_repository(self, root_dir: str | Path, max_files: int = 50) -> int:
        """Scan and index symbols from the repository."""
        chunks = extract_repo_symbols(root_dir, max_files=max_files)
        if not chunks:
            return 0

        records = [c.model_dump() for c in chunks]
        try:
            self._table = self.db.create_table(self.table_name, data=records, mode="overwrite")
            return len(records)
        except Exception as exc:
            logger.warning("Failed to write to LanceDB table: %s", exc)
            return 0

    def search(
        self,
        query: str | Sequence[float],
        limit: int = 5,
        where: str | None = None,
    ) -> list[QueryResult]:
        """Search table by embedding vector or text query."""
        if not self._table:
            return []

        if isinstance(query, str):
            vec = _embed_text(query)
        else:
            vec = list(query)

        try:
            q = self._table.search(vec).limit(limit)
            if where:
                q = q.where(where)
            raw_results = q.to_list()
        except Exception as exc:
            logger.warning("Search query failed: %s", exc)
            return []

        results: list[QueryResult] = []
        for r in raw_results:
            dist = float(r.get("_distance", 0.0))
            score = max(0.0, 1.0 - dist)
            chunk = CodeChunk(
                id=str(r.get("id", "")),
                symbol=str(r.get("symbol", "")),
                file=str(r.get("file", "")),
                content=str(r.get("content", "")),
                vector=r.get("vector", []),
                metadata=r.get("metadata", {}),
            )
            results.append(QueryResult(chunk=chunk, score=score, distance=dist))

        return results

    def get_context_snippets(self, query: str, max_tokens: int = 250) -> list[str]:
        """Retrieve top context snippets strictly bounded to <= max_tokens total."""
        results = self.search(query, limit=5)
        snippets: list[str] = []
        total_tokens = 0

        for r in results:
            content = r.chunk.content.strip()
            if not content:
                continue

            words = content.split()
            remaining_budget = max_tokens - total_tokens
            if remaining_budget <= 0:
                break

            if len(words) > remaining_budget:
                truncated_content = " ".join(words[:remaining_budget])
                snippet = f"[{r.chunk.file} : {r.chunk.symbol}]\n{truncated_content}"
                snippets.append(snippet)
                total_tokens += remaining_budget
                break
            else:
                snippet = f"[{r.chunk.file} : {r.chunk.symbol}]\n{content}"
                snippets.append(snippet)
                total_tokens += len(words)

        return snippets
