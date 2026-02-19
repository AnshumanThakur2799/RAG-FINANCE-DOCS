from __future__ import annotations

import argparse
import re
import time

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import build_vector_store
from app.embeddings.client import build_embedding_client
from app.llm.client import build_llm_client
from app.retrieval import RETRIEVAL_MODES, build_retriever


SYSTEM_PROMPT = (
    "You are an expert tender reviewer for government tenders in West Bengal. "
    "The source corpus is from the West Bengal Government e-Procurement portal "
    "(https://tenders.wb.gov.in/nicgep/app) and includes tenders from multiple "
    "government organizations, departments, and authorities. "
    "Answer using only the provided context and do not add facts that are not present. "
    "Focus on decision-useful details such as eligibility, scope of work, dates, fees, "
    "submission requirements, evaluation criteria, and compliance risks. "
    "If the context is insufficient or ambiguous, clearly say you do not have enough information. "
    "Keep the answer concise and structured. "
    "Include citations for factual statements in the form [tender_id/source_name#chunk_id]. "
    "Do not output chain-of-thought, hidden reasoning, or <think> tags."
)

TENDER_ID_PATTERN = re.compile(r"\d{4}_[A-Za-z0-9]+_\d+_\d+")


def _extract_tender_id_from_source_path(source_path: str) -> str | None:
    match = TENDER_ID_PATTERN.search(source_path)
    if not match:
        return None
    return match.group(0)


def _format_citation_source(result: dict) -> str:
    source = result.get("source_name") or result.get("source_path", "unknown")
    tender_id = result.get("tender_id")
    if not tender_id:
        source_path = result.get("source_path", "")
        if isinstance(source_path, str) and source_path:
            tender_id = _extract_tender_id_from_source_path(source_path)
    if not tender_id:
        return str(source)
    source_text = str(source)
    if source_text.startswith(f"{tender_id}/"):
        return source_text
    return f"{tender_id}/{source_text}"


def build_context(results: list[dict]) -> str:
    lines = []
    for result in results:
        source = _format_citation_source(result)
        chunk_id = result.get("chunk_id", "n/a")
        text = (result.get("text") or "").strip()
        lines.append(f"[{source}#{chunk_id}] {text}")
    return "\n\n".join(lines)


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\]]+#\d+\]", text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using local RAG.")
    parser.add_argument("--query", required=True, help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks.")
    parser.add_argument(
        "--table", default="document_chunks", help="Vector collection/table name."
    )
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
    vector_store = build_vector_store(settings, table_name=args.table)
    retrieval_mode = args.retrieval_mode.strip().lower() or settings.retrieval_mode
    retriever = build_retriever(
        mode=retrieval_mode,
        embed_client=embed_client,
        vector_store=vector_store,
        document_store=document_store,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        hybrid_candidate_multiplier=settings.hybrid_candidate_multiplier,
    )
    llm_client = build_llm_client(
        settings.llm_provider,
        deepinfra_api_key=settings.deepinfra_api_key,
        deepinfra_base_url=settings.deepinfra_base_url,
        deepinfra_model=settings.deepinfra_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    search_start = time.perf_counter()
    results = retriever.retrieve(args.query, top_k=max(1, args.top_k))
    search_elapsed = time.perf_counter() - search_start
    print(f"Retrieval time ({retrieval_mode}): {search_elapsed:.4f} seconds")

    if not results:
        print("No results found. Have you indexed PDFs yet?")
        return

    context = build_context(results)
    user_prompt = (
        f"Question:\n{args.query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in plain final form only (no <think> tags). "
        "Every factual bullet/sentence must include at least one citation in the format "
        "[tender_id/source_name#chunk_id]."
    )

    answer = llm_client.chat(SYSTEM_PROMPT, user_prompt)
    if not _has_citation(answer):
        fallback_sources = []
        for result in results[:3]:
            source = _format_citation_source(result)
            chunk_id = result.get("chunk_id", "n/a")
            fallback_sources.append(f"[{source}#{chunk_id}]")
        if fallback_sources:
            answer = (
                f"{answer}\n\nCitations: "
                + ", ".join(dict.fromkeys(fallback_sources))
            )
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
