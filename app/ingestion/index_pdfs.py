from __future__ import annotations

import argparse
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.config import Settings
from app.db.sqlite_store import ChunkRecord, DocumentRecord, SQLiteDocumentStore
from app.db.vector_store import LanceDBVectorStore
from app.embeddings.client import build_embedding_client
from app.ingestion.chunker import TokenChunker
from app.ingestion.pdf_reader import extract_text_from_pdf


@dataclass(frozen=True)
class IndexResult:
    indexed: bool
    num_chunks: int
    status: str


def iter_pdf_paths(input_dir: Path) -> Iterator[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    for path in sorted(input_dir.rglob("*.pdf")):
        if path.is_file():
            yield path


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def batch_items(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def index_pdf(
    path: Path,
    *,
    chunker: TokenChunker,
    embed_client,
    vector_store: LanceDBVectorStore,
    store: SQLiteDocumentStore,
    reindex: bool,
    embed_batch_size: int,
    recreate_table_on_dim_mismatch: bool,
) -> IndexResult:
    file_hash = compute_file_hash(path)
    existing = store.get_by_hash(file_hash)
    if existing and not reindex:
        return IndexResult(indexed=False, num_chunks=0, status="already_indexed")

    text = extract_text_from_pdf(path)
    if not text:
        return IndexResult(indexed=False, num_chunks=0, status="empty_pdf")

    chunks = chunker.chunk(text)
    embeddings: list[list[float]] = []
    for batch in batch_items(chunks, embed_batch_size):
        embeddings.extend(embed_client.embed(batch, input_type="document"))

    if len(chunks) != len(embeddings):
        raise RuntimeError("Embedding count mismatch.")

    doc_id = file_hash
    if existing:
        vector_store.delete_by_doc_id(doc_id)

    records = []
    lexical_records: list[ChunkRecord] = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_uid = f"{doc_id}::chunk_{idx}"
        records.append(
            {
                "id": chunk_uid,
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk,
                "vector": embedding,
                "source_path": str(path.resolve()),
                "source_name": path.name,
            }
        )
        lexical_records.append(
            ChunkRecord(
                id=chunk_uid,
                doc_id=doc_id,
                chunk_id=idx,
                text=chunk,
                source_path=str(path.resolve()),
                source_name=path.name,
            )
        )

    vector_store.upsert_chunks(
        records, recreate_on_dim_mismatch=recreate_table_on_dim_mismatch
    )
    store.replace_chunks(doc_id, lexical_records)

    stat = path.stat()
    store.upsert_document(
        DocumentRecord(
            id=doc_id,
            file_path=str(path.resolve()),
            file_name=path.name,
            file_hash=file_hash,
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            indexed_at=store.now_iso(),
            num_chunks=len(chunks),
        )
    )
    return IndexResult(indexed=True, num_chunks=len(chunks), status="indexed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index PDFs into LanceDB and track metadata in SQLite."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing PDFs to index.",
    )
    parser.add_argument(
        "--table",
        default="document_chunks",
        help="LanceDB table name.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex even if the PDF hash already exists.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-table-recreate-on-dim-mismatch",
        action="store_true",
        help=(
            "Do not recreate LanceDB table when embedding dimension mismatches "
            "(advanced/debug use)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings = Settings.from_env()
    input_dir = Path(args.input)
    store = SQLiteDocumentStore(settings.sqlite_db_path)
    vector_store = LanceDBVectorStore(settings.lancedb_dir, table_name=args.table)
    embed_client = build_embedding_client(
        settings.embedding_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_embedding_model,
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embeddings_deployment,
        local_model=settings.local_embedding_model,
        local_normalize=settings.local_embedding_normalize,
        local_prompt_style=settings.local_embedding_prompt_style,
        local_device=settings.local_embedding_device,
    )
    chunker = TokenChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )

    all_pdfs = list(iter_pdf_paths(input_dir))
    run_start = time.perf_counter()
    wall_start = store.now_iso()

    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks_indexed = 0
    total_file_processing_seconds = 0.0

    logging.info(
        "Starting indexing run | input=%s | total_files=%s | reindex=%s | embed_batch_size=%s",
        str(input_dir.resolve()),
        len(all_pdfs),
        args.reindex,
        args.embed_batch_size,
    )

    for file_number, pdf_path in enumerate(all_pdfs, start=1):
        file_start = time.perf_counter()
        try:
            result = index_pdf(
                pdf_path,
                chunker=chunker,
                embed_client=embed_client,
                vector_store=vector_store,
                store=store,
                reindex=args.reindex,
                embed_batch_size=args.embed_batch_size,
                recreate_table_on_dim_mismatch=(
                    args.reindex and not args.no_table_recreate_on_dim_mismatch
                ),
            )
            file_elapsed = time.perf_counter() - file_start
            total_file_processing_seconds += file_elapsed

            if result.indexed:
                indexed_count += 1
                total_chunks_indexed += result.num_chunks
                logging.info(
                    "[%s/%s] Indexed %s | chunks=%s | time=%.2fs",
                    file_number,
                    len(all_pdfs),
                    pdf_path.name,
                    result.num_chunks,
                    file_elapsed,
                )
            else:
                skipped_count += 1
                logging.info(
                    "[%s/%s] Skipped %s | reason=%s | time=%.2fs",
                    file_number,
                    len(all_pdfs),
                    pdf_path.name,
                    result.status,
                    file_elapsed,
                )
        except Exception as exc:
            file_elapsed = time.perf_counter() - file_start
            total_file_processing_seconds += file_elapsed
            failed_count += 1
            logging.exception(
                "[%s/%s] Failed %s | time=%.2fs | error=%s",
                file_number,
                len(all_pdfs),
                pdf_path.name,
                file_elapsed,
                str(exc),
            )

    run_elapsed = time.perf_counter() - run_start
    avg_file_seconds = (
        total_file_processing_seconds / len(all_pdfs) if all_pdfs else 0.0
    )
    avg_chunks_per_indexed = (
        total_chunks_indexed / indexed_count if indexed_count else 0.0
    )

    logging.info(
        "Indexing complete | started_at=%s | finished_at=%s | total_files=%s | indexed=%s | skipped=%s | failed=%s | total_chunks_indexed=%s | avg_chunks_per_indexed=%.2f | processing_time_seconds=%.2f | avg_time_per_file_seconds=%.2f",
        wall_start,
        store.now_iso(),
        len(all_pdfs),
        indexed_count,
        skipped_count,
        failed_count,
        total_chunks_indexed,
        avg_chunks_per_indexed,
        run_elapsed,
        avg_file_seconds,
    )


if __name__ == "__main__":
    main()
