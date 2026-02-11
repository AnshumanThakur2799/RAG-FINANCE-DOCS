from __future__ import annotations

from pathlib import Path
from typing import Iterable

import lancedb


class LanceDBVectorStore:
    def __init__(self, db_dir: Path, table_name: str = "document_chunks") -> None:
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self._db = lancedb.connect(str(self.db_dir))

    def _table_exists(self) -> bool:
        return self.table_name in self._db.table_names()

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._table_exists():
            return
        table = self._db.open_table(self.table_name)
        table.delete(f"doc_id = '{doc_id}'")

    def upsert_chunks(self, records: Iterable[dict]) -> None:
        records = list(records)
        if not records:
            return
        if self._table_exists():
            table = self._db.open_table(self.table_name)
            table.add(records)
        else:
            self._db.create_table(self.table_name, data=records)

    def search(self, vector: list[float], *, top_k: int = 5) -> list[dict]:
        if not self._table_exists():
            return []
        table = self._db.open_table(self.table_name)
        return table.search(vector).limit(top_k).to_list()
