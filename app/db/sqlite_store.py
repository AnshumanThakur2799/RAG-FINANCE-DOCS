from __future__ import annotations

import sqlite3
import re
import json
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


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    doc_id: str
    chunk_id: int
    text: str
    source_path: str
    source_name: str


@dataclass(frozen=True)
class TenderTextRecord:
    tender_id: str
    doc_id: str
    source_path: str
    source_name: str
    full_text: str
    metadata: dict[str, str | int]
    indexed_at: str


@dataclass(frozen=True)
class ChunkMetadataRecord:
    chunk_uid: str
    doc_id: str
    raw_table_json: str


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
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    id UNINDEXED,
                    doc_id UNINDEXED,
                    chunk_id UNINDEXED,
                    source_path UNINDEXED,
                    source_name UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tender_texts (
                    tender_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (tender_id, source_path)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tender_texts_tender_id ON tender_texts(tender_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    chunk_uid TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    raw_table_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_metadata_doc_id ON chunk_metadata(doc_id)"
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

    def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_fts WHERE doc_id = ?", (doc_id,))

    def replace_chunks(self, doc_id: str, chunks: Iterable[ChunkRecord]) -> None:
        chunk_records = list(chunks)
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_fts WHERE doc_id = ?", (doc_id,))
            if chunk_records:
                conn.executemany(
                    """
                    INSERT INTO chunk_fts (
                        id, doc_id, chunk_id, source_path, source_name, text
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.doc_id,
                            chunk.chunk_id,
                            chunk.source_path,
                            chunk.source_name,
                            chunk.text,
                        )
                        for chunk in chunk_records
                    ],
                )

    def replace_chunk_metadata(
        self, doc_id: str, records: Iterable[ChunkMetadataRecord]
    ) -> None:
        metadata_records = list(records)
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_metadata WHERE doc_id = ?", (doc_id,))
            if metadata_records:
                conn.executemany(
                    """
                    INSERT INTO chunk_metadata (chunk_uid, doc_id, raw_table_json)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            record.chunk_uid,
                            record.doc_id,
                            record.raw_table_json,
                        )
                        for record in metadata_records
                    ],
                )

    def attach_chunk_metadata(self, results: list[dict]) -> list[dict]:
        chunk_ids = [
            str(result.get("id"))
            for result in results
            if isinstance(result.get("id"), str) and result.get("id")
        ]
        if not chunk_ids:
            return results

        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_uid, raw_table_json
                FROM chunk_metadata
                WHERE chunk_uid IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        table_json_by_chunk_id = {row[0]: row[1] for row in rows}
        for result in results:
            chunk_uid = result.get("id")
            if not isinstance(chunk_uid, str):
                continue
            table_json = table_json_by_chunk_id.get(chunk_uid)
            if table_json:
                result["table_json"] = table_json
        return results

    def upsert_tender_text(self, record: TenderTextRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tender_texts (
                    tender_id, doc_id, source_path, source_name,
                    full_text, metadata_json, indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tender_id, source_path) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    source_name = excluded.source_name,
                    full_text = excluded.full_text,
                    metadata_json = excluded.metadata_json,
                    indexed_at = excluded.indexed_at
                """,
                (
                    record.tender_id,
                    record.doc_id,
                    record.source_path,
                    record.source_name,
                    record.full_text,
                    json.dumps(record.metadata),
                    record.indexed_at,
                ),
            )

    def get_tender_full_text(self, tender_id: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT full_text
                FROM tender_texts
                WHERE tender_id = ?
                ORDER BY source_name ASC, source_path ASC
                """,
                (tender_id,),
            ).fetchall()
        return "\n\n".join(row[0] for row in rows if row and row[0])

    @staticmethod
    def _to_fts_query(raw_query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\.\-_]*", raw_query.lower())
        if not tokens:
            return ""
        unique_tokens = list(dict.fromkeys(tokens))
        return " OR ".join(f'"{token}"' for token in unique_tokens[:20])

    def search_chunks(self, query: str, *, top_k: int = 5) -> list[dict]:
        fts_query = self._to_fts_query(query)
        if not fts_query:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    doc_id,
                    chunk_id,
                    text,
                    source_path,
                    source_name,
                    bm25(chunk_fts) AS bm25_score
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (fts_query, max(1, top_k)),
            ).fetchall()

        return [
            {
                "id": row[0],
                "doc_id": row[1],
                "chunk_id": row[2],
                "text": row[3],
                "source_path": row[4],
                "source_name": row[5],
                "_lexical_score": row[6],
            }
            for row in rows
        ]

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
