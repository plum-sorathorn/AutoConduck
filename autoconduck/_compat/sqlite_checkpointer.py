"""Compatibility fallback for LangGraph SQLite Checkpointer when native package is absent."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from typing import Any, AsyncIterator, Iterator, Sequence

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _NativeSqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _NativeAsyncSqliteSaver
    HAS_SQLITE_CHECKPOINTER = True
except Exception:
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver as _NativeSqliteSaver
        from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver as _NativeAsyncSqliteSaver
        HAS_SQLITE_CHECKPOINTER = True
    except Exception:
        _NativeSqliteSaver = None
        _NativeAsyncSqliteSaver = None
        HAS_SQLITE_CHECKPOINTER = False


try:
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
except Exception:
    BaseCheckpointSaver = object
    CheckpointTuple = None  # type: ignore[assignment, misc]


def is_sqlite_checkpointer_available() -> bool:
    """Return True if native LangGraph SQLite checkpointer is installed and importable."""
    return HAS_SQLITE_CHECKPOINTER and _NativeSqliteSaver is not None


class CheckpointTupleFallback:
    """Fallback representation of a LangGraph CheckpointTuple."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        parent_config: dict[str, Any] | None = None,
        pending_writes: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.metadata = metadata or {}
        self.parent_config = parent_config
        self.pending_writes = pending_writes or []


class SqliteSaverFallback(BaseCheckpointSaver):
    """Pure-Python SQLite Checkpoint Saver fallback."""

    def __init__(self, conn: sqlite3.Connection | str = ":memory:", *, serde: Any = None) -> None:
        if BaseCheckpointSaver is not object:
            super().__init__(serde=serde)
        self._lock = threading.Lock()
        if isinstance(conn, str):
            self.conn = sqlite3.connect(conn, check_same_thread=False)
        else:
            self.conn = conn
        self._is_fallback = True
        self.setup()

    def setup(self) -> None:
        """Initialize checkpoint schema tables."""
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB NOT NULL,
                        metadata_type TEXT,
                        metadata BLOB NOT NULL,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoint_writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        blob BLOB NOT NULL,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """
                )
                try:
                    self.conn.execute("ALTER TABLE checkpoints ADD COLUMN metadata_type TEXT")
                except Exception:
                    pass

    @classmethod
    def from_conn_string(cls, conn_string: str) -> SqliteSaverFallback:
        return cls(conn_string)

    def _deserialize(self, type_hint: str | None, blob: Any) -> Any:
        """Robustly deserialize blob from serde or fallback json."""
        if not blob:
            return {}
        if hasattr(self, "serde") and self.serde:
            if type_hint:
                try:
                    return self.serde.loads_typed((type_hint, blob))
                except Exception:
                    pass
            for th in ("msgpack", "json"):
                try:
                    return self.serde.loads_typed((th, blob))
                except Exception:
                    pass
        try:
            raw = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else blob
            return json.loads(raw)
        except Exception:
            return {}

    def get_tuple(self, config: dict[str, Any]) -> Any | None:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        with self._lock:
            cur = self.conn.cursor()
            if checkpoint_id:
                cur.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cur.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                )
            row = cur.fetchone()
        if not row:
            return None

        c_id, parent_id, c_type, c_data, m_type, meta_data = row
        parsed_checkpoint = self._deserialize(c_type, c_data)
        parsed_metadata = self._deserialize(m_type, meta_data)

        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        tuple_cls = CheckpointTuple if CheckpointTuple is not None else CheckpointTupleFallback
        return tuple_cls(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": c_id,
                }
            },
            checkpoint=parsed_checkpoint,
            metadata=parsed_metadata,
            parent_config=parent_config,
            pending_writes=[],
        )

    def list(
        self,
        config: dict[str, Any] | None,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[Any]:
        configurable = (config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        query = "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata FROM checkpoints"
        params: list[Any] = []
        conditions: list[str] = []

        if thread_id:
            conditions.append("thread_id = ?")
            params.append(thread_id)
            conditions.append("checkpoint_ns = ?")
            params.append(checkpoint_ns)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._lock:
            cur = self.conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        tuple_cls = CheckpointTuple if CheckpointTuple is not None else CheckpointTupleFallback
        for row in rows:
            t_id, ns, c_id, parent_id, c_type, c_data, m_type, meta_data = row
            parsed_checkpoint = self._deserialize(c_type, c_data)
            parsed_metadata = self._deserialize(m_type, meta_data)

            p_cfg = {
                "configurable": {
                    "thread_id": t_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": parent_id,
                }
            } if parent_id else None

            yield tuple_cls(
                config={"configurable": {"thread_id": t_id, "checkpoint_ns": ns, "checkpoint_id": c_id}},
                checkpoint=parsed_checkpoint,
                metadata=parsed_metadata,
                parent_config=p_cfg,
                pending_writes=[],
            )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = checkpoint.get("id") or configurable.get("checkpoint_id", "default")
        parent_id = configurable.get("checkpoint_id")

        c_type, c_blob = "json", b""
        if hasattr(self, "serde") and self.serde:
            try:
                c_type, c_blob = self.serde.dumps_typed(checkpoint)
            except Exception:
                c_type, c_blob = "json", json.dumps(checkpoint, default=str).encode("utf-8")
        else:
            c_type, c_blob = "json", json.dumps(checkpoint, default=str).encode("utf-8")

        m_type, m_blob = "json", b""
        if hasattr(self, "serde") and self.serde:
            try:
                m_type, m_blob = self.serde.dumps_typed(metadata)
            except Exception:
                m_type, m_blob = "json", json.dumps(metadata, default=str).encode("utf-8")
        else:
            m_type, m_blob = "json", json.dumps(metadata, default=str).encode("utf-8")

        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id, parent_id, c_type, c_blob, m_type, m_blob),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id", "")

        serialized_writes = []
        for idx, (channel, val) in enumerate(writes):
            w_type, blob = "json", b""
            if hasattr(self, "serde") and self.serde:
                try:
                    w_type, blob = self.serde.dumps_typed(val)
                except Exception:
                    w_type = "json"
                    blob = json.dumps(val, default=str).encode("utf-8")
            else:
                w_type = "json"
                blob = json.dumps(val, default=str).encode("utf-8")
            serialized_writes.append((str(thread_id), str(checkpoint_ns), str(checkpoint_id), str(task_id), int(idx), str(channel), str(w_type), blob))

        with self._lock:
            with self.conn:
                for row_params in serialized_writes:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row_params,
                    )

    # Async variants
    async def aget_tuple(self, config: dict[str, Any]) -> Any | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: dict[str, Any] | None,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Any]:
        items = list(self.list(config, filter=filter, before=before, limit=limit))
        for item in items:
            yield item

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)


def get_sqlite_checkpointer(conn_string: str = ":memory:") -> Any:
    """Return a unified SqliteSaver compatible with sync and async LangGraph runners."""
    return SqliteSaverFallback.from_conn_string(conn_string)
