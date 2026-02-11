from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    file_path: str
    file_name: str
    file_hash: str
    file_size: int
    modified_time: float
    indexed_at: str
    num_chunks: int


class SQLiteDocumentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    indexed_at TEXT NOT NULL,
                    num_chunks INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(file_path)"
            )

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, file_path, file_name, file_hash, file_size, modified_time, indexed_at, num_chunks "
                "FROM documents WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return DocumentRecord(*row) if row else None

    def get_by_path(self, file_path: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, file_path, file_name, file_hash, file_size, modified_time, indexed_at, num_chunks "
                "FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        return DocumentRecord(*row) if row else None

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, file_path, file_name, file_hash, file_size, modified_time,
                    indexed_at, num_chunks
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_name = excluded.file_name,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    modified_time = excluded.modified_time,
                    indexed_at = excluded.indexed_at,
                    num_chunks = excluded.num_chunks
                """,
                (
                    record.id,
                    record.file_path,
                    record.file_name,
                    record.file_hash,
                    record.file_size,
                    record.modified_time,
                    record.indexed_at,
                    record.num_chunks,
                ),
            )

    def upsert_documents(self, records: Iterable[DocumentRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO documents (
                    id, file_path, file_name, file_hash, file_size, modified_time,
                    indexed_at, num_chunks
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_name = excluded.file_name,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    modified_time = excluded.modified_time,
                    indexed_at = excluded.indexed_at,
                    num_chunks = excluded.num_chunks
                """,
                [
                    (
                        record.id,
                        record.file_path,
                        record.file_name,
                        record.file_hash,
                        record.file_size,
                        record.modified_time,
                        record.indexed_at,
                        record.num_chunks,
                    )
                    for record in records
                ],
            )

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
