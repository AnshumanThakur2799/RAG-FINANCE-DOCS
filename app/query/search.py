from __future__ import annotations

import argparse

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import LanceDBVectorStore
from app.embeddings.client import build_embedding_client
from app.retrieval import RETRIEVAL_MODES, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LanceDB for matching chunks.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument("--table", default="document_chunks", help="LanceDB table name.")
    parser.add_argument(
        "--retrieval-mode",
        default="",
        help=f"Retrieval mode: {', '.join(RETRIEVAL_MODES)}.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
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
    document_store = SQLiteDocumentStore(settings.sqlite_db_path)
    vector_store = LanceDBVectorStore(settings.lancedb_dir, table_name=args.table)
    retrieval_mode = args.retrieval_mode.strip().lower() or settings.retrieval_mode
    retriever = build_retriever(
        mode=retrieval_mode,
        embed_client=embed_client,
        vector_store=vector_store,
        document_store=document_store,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        hybrid_candidate_multiplier=settings.hybrid_candidate_multiplier,
    )

    results = retriever.retrieve(args.query, top_k=max(1, args.top_k))

    if not results:
        print("No results found. Have you indexed PDFs yet?")
        return

    print("Top matches:")
    for idx, result in enumerate(results, start=1):
        source = result.get("source_name") or result.get("source_path", "unknown")
        chunk_id = result.get("chunk_id", "n/a")
        score = result.get(
            "_rrf_score",
            result.get("_distance", result.get("_lexical_score", result.get("score", "n/a"))),
        )
        if isinstance(score, float):
            score_display = f"{score:.4f}"
        else:
            score_display = str(score)
        text = (result.get("text") or "").replace("\n", " ").strip()
        if len(text) > 400:
            text = f"{text[:400]}..."
        print(f"{idx}. {source} (chunk {chunk_id}, score {score_display})")
        print(f"   {text}")


if __name__ == "__main__":
    main()
