"""Compatibility fallback for LanceDB when native lancedb/arrow libraries are absent."""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:
    import lancedb
    HAS_LANCEDB = True
except Exception:
    lancedb = None
    HAS_LANCEDB = False


def is_lancedb_available() -> bool:
    """Return True if native lancedb is installed and importable."""
    return HAS_LANCEDB and lancedb is not None


def _cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    # Clamp float rounding
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


class LanceDBFallbackQuery:
    """Query builder emulation for LanceDB table search."""

    def __init__(self, table: LanceDBFallbackTable, query_vector: Sequence[float] | None = None) -> None:
        self.table = table
        self.query_vector = query_vector
        self._limit = 10
        self._where_clause: str | None = None
        self._columns: list[str] | None = None

    def limit(self, n: int) -> LanceDBFallbackQuery:
        self._limit = n
        return self

    def where(self, clause: str) -> LanceDBFallbackQuery:
        self._where_clause = clause
        return self

    def select(self, columns: list[str]) -> LanceDBFallbackQuery:
        self._columns = columns
        return self

    def to_list(self) -> list[dict[str, Any]]:
        rows = list(self.table._rows)
        scored: list[dict[str, Any]] = []

        for row in rows:
            row_copy = dict(row)
            if self.query_vector is not None:
                vec = row_copy.get("vector") or row_copy.get("embedding")
                if isinstance(vec, (list, tuple)):
                    dist = _cosine_distance(self.query_vector, vec)
                else:
                    dist = 1.0
                row_copy["_distance"] = dist
            else:
                row_copy["_distance"] = 0.0

            # Simple where clause filter (e.g. "key == 'val'" or substring)
            if self._where_clause:
                # Basic key equality check support
                if "==" in self._where_clause:
                    k, v = [part.strip().strip("'\"") for part in self._where_clause.split("==", 1)]
                    if str(row_copy.get(k, "")) != v:
                        continue
            scored.append(row_copy)

        if self.query_vector is not None:
            scored.sort(key=lambda r: r.get("_distance", 1.0))

        results = scored[: self._limit]
        if self._columns:
            filtered_results = []
            for r in results:
                filtered = {col: r[col] for col in self._columns if col in r}
                if "_distance" in r:
                    filtered["_distance"] = r["_distance"]
                filtered_results.append(filtered)
            return filtered_results
        return results

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
            return pd.DataFrame(self.to_list())
        except ImportError:
            return self.to_list()


class LanceDBFallbackTable:
    """Pure-Python in-memory table emulator for LanceDB."""

    def __init__(self, name: str, schema: Any = None) -> None:
        self.name = name
        self.schema = schema
        self._rows: list[dict[str, Any]] = []
        self._is_fallback = True

    def add(self, data: list[dict[str, Any]] | list[Any]) -> None:
        for item in data:
            if isinstance(item, dict):
                self._rows.append(dict(item))
            elif hasattr(item, "model_dump"):
                self._rows.append(item.model_dump())
            elif hasattr(item, "dict"):
                self._rows.append(item.dict())
            else:
                self._rows.append(vars(item))

    def search(self, query: Sequence[float] | None = None, vector_column_name: str = "vector") -> LanceDBFallbackQuery:
        return LanceDBFallbackQuery(self, query_vector=query)

    def count_rows(self) -> int:
        return len(self._rows)

    def delete(self, where: str = "") -> None:
        if not where:
            self._rows.clear()
            return
        if "==" in where:
            k, v = [part.strip().strip("'\"") for part in where.split("==", 1)]
            self._rows = [r for r in self._rows if str(r.get(k, "")) != v]

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
            return pd.DataFrame(self._rows)
        except ImportError:
            return self._rows

    def to_arrow(self) -> Any:
        try:
            import pyarrow as pa
            return pa.Table.from_pylist(self._rows)
        except ImportError:
            return self._rows


class LanceDBFallbackConnection:
    """In-memory connection emulator for LanceDB."""

    def __init__(self, uri: str = "", **kwargs: Any) -> None:
        self.uri = uri
        self.kwargs = kwargs
        self._tables: dict[str, LanceDBFallbackTable] = {}
        self._is_fallback = True

    def table_names(self) -> list[str]:
        return list(self._tables.keys())

    def create_table(
        self,
        name: str,
        data: list[dict[str, Any]] | None = None,
        schema: Any = None,
        mode: str = "create",
        exist_ok: bool = True,
    ) -> LanceDBFallbackTable:
        if name in self._tables and not exist_ok and mode == "create":
            raise ValueError(f"Table '{name}' already exists.")
        if name not in self._tables or mode == "overwrite":
            self._tables[name] = LanceDBFallbackTable(name, schema=schema)
        table = self._tables[name]
        if data:
            table.add(data)
        return table

    def open_table(self, name: str) -> LanceDBFallbackTable:
        if name not in self._tables:
            self._tables[name] = LanceDBFallbackTable(name)
        return self._tables[name]

    def drop_table(self, name: str) -> None:
        self._tables.pop(name, None)


def connect(uri: str = "", **kwargs: Any) -> Any:
    """Connect to a LanceDB database or return an in-memory fallback connection."""
    if is_lancedb_available():
        try:
            return lancedb.connect(uri, **kwargs)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Failed to connect to native lancedb at %s (%s). Using fallback.", uri, exc)
            return LanceDBFallbackConnection(uri, **kwargs)
    return LanceDBFallbackConnection(uri, **kwargs)
