from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

from app.config import Settings
from app.db.sqlite_store import DocumentRecord, SQLiteDocumentStore
from app.db.vector_store import LanceDBVectorStore
from app.embeddings.client import build_embedding_client
from app.ingestion.chunker import TokenChunker
from app.ingestion.pdf_reader import extract_text_from_pdf


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
) -> bool:
    file_hash = compute_file_hash(path)
    existing = store.get_by_hash(file_hash)
    if existing and not reindex:
        return False

    text = extract_text_from_pdf(path)
    if not text:
        print(f"Skipping empty PDF: {path}")
        return False

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
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        records.append(
            {
                "id": f"{doc_id}::chunk_{idx}",
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk,
                "vector": embedding,
                "source_path": str(path.resolve()),
                "source_name": path.name,
            }
        )

    vector_store.upsert_chunks(records)

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
    return True


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
    args = parser.parse_args()

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

    indexed_count = 0
    skipped_count = 0
    for pdf_path in iter_pdf_paths(input_dir):
        try:
            indexed = index_pdf(
                pdf_path,
                chunker=chunker,
                embed_client=embed_client,
                vector_store=vector_store,
                store=store,
                reindex=args.reindex,
                embed_batch_size=args.embed_batch_size,
            )
            if indexed:
                indexed_count += 1
                print(f"Indexed: {pdf_path.name}")
            else:
                skipped_count += 1
                print(f"Skipped: {pdf_path.name}")
        except Exception as exc:
            print(f"Failed: {pdf_path.name} -> {exc}")

    print(f"Done. Indexed: {indexed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
