from __future__ import annotations

import argparse

from app.config import Settings
from app.db.vector_store import LanceDBVectorStore
from app.embeddings.client import build_embedding_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LanceDB for matching chunks.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument("--table", default="document_chunks", help="LanceDB table name.")
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
    vector_store = LanceDBVectorStore(settings.lancedb_dir, table_name=args.table)

    embedding = embed_client.embed([args.query], input_type="query")[0]
    results = vector_store.search(embedding, top_k=max(1, args.top_k))

    if not results:
        print("No results found. Have you indexed PDFs yet?")
        return

    print("Top matches:")
    for idx, result in enumerate(results, start=1):
        source = result.get("source_name") or result.get("source_path", "unknown")
        chunk_id = result.get("chunk_id", "n/a")
        score = result.get("_distance", result.get("score", "n/a"))
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
