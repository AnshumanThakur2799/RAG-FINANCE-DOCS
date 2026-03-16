from __future__ import annotations

import re
import time

import gradio as gr

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import build_vector_store
from app.embeddings.client import build_embedding_client
from app.llm.client import build_llm_client
from app.retrieval import (
    RETRIEVAL_MODES,
    build_full_tender_context,
    build_retriever,
)


SYSTEM_PROMPT = """
You are an internal Enterprise AI Assistant for business teams.

Answer strictly from the provided document context.
Do not invent facts and do not use external knowledge.
If the answer is not explicitly supported in context, respond:
"Insufficient information in the provided documents."

Behavior rules:
1. Answer only what is asked.
2. Keep responses concise and decision-ready.
3. Use plain text or markdown; use bullets only when they improve clarity.
4. Highlight dates, amounts, percentages, owners, and deadlines when present.
5. Cite factual statements using: [source#chunk_or_page].
6. If multiple documents are relevant, cite up to the 3 strongest sources.
7. If the question is ambiguous, ask one short clarification question.
8. Do not output chain-of-thought or internal reasoning.

Length control:
- Keep the response concise (target under ~900 tokens).
- If content is long, prioritize the direct answer first.
"""

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


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\]]+#\d+\]", text))


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)


def _build_ask_user_prompt(query: str, context: str) -> str:
    return (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Return only the direct answer to the question.\n"
        "Do not include extra sections or templates.\n"
        "Use bullets only if needed for clarity.\n\n"
        "Rules:\n"
        "- Include citation(s) for factual claims using [source#chunk_or_page].\n"
        "- If info is missing, say 'Insufficient information in the provided documents.'\n"
        "- No chain-of-thought, no XML tags."
    )


class EmbeddingGradioApp:
    def __init__(self) -> None:
        settings = Settings.from_env()
        self.settings = settings
        self.embed_client = build_embedding_client(
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
            local_device=settings.local_embedding_device,\
            deepinfra_base_url=settings.deepinfra_base_url,
        )
        self.document_store = SQLiteDocumentStore(settings.sqlite_db_path)
        self.vector_store = build_vector_store(settings, table_name="tenders_data")
        self.llm_client = build_llm_client(
            settings.llm_provider,
            deepinfra_api_key=settings.deepinfra_api_key,
            deepinfra_base_url=settings.deepinfra_base_url,
            deepinfra_model=settings.deepinfra_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

    def on_embed(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Please enter a query first."

        try:
            embedding = self.embed_client.embed([query], input_type="query")[0]
        except Exception as exc:
            return f"Embedding failed: {exc}"

        preview = ", ".join(f"{value:.6f}" for value in embedding[:12])
        return (
            "### Embedding Result\n\n"
            f"- Length: `{len(embedding)}`\n\n"
            "```text\n"
            f"First 12 values: [{preview}]\n"
            "```"
        )

    def on_search(self, query: str, top_k: float, mode: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Please enter a query first."
        if top_k is None:
            return "Top K must be a number."

        mode = (mode or "").strip().lower() or self.settings.retrieval_mode
        top_k_int = max(1, int(top_k))

        try:
            retriever = build_retriever(
                mode=mode,
                embed_client=self.embed_client,
                vector_store=self.vector_store,
                document_store=self.document_store,
                hybrid_rrf_k=self.settings.hybrid_rrf_k,
                hybrid_candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            )
            start_time = time.perf_counter()
            results = retriever.retrieve(query, top_k=top_k_int)
            search_time = time.perf_counter() - start_time
        except Exception as exc:
            return f"Search failed: {exc}"

        if not results:
            return "No results found. Have you indexed PDFs yet?"

        lines = ["Top matches:"]
        for idx, result in enumerate(results, start=1):
            source = _format_citation_source(result)
            chunk_id = result.get("chunk_id", "n/a")
            score = result.get(
                "_rrf_score",
                result.get(
                    "_distance",
                    result.get("_lexical_score", result.get("score", "n/a")),
                ),
            )
            if isinstance(score, float):
                score_display = f"{score:.4f}"
            else:
                score_display = str(score)
            text = (result.get("text") or "").replace("\n", " ").strip()
            if len(text) > 400:
                text = f"{text[:400]}..."
            lines.append(f"{idx}. {source} (chunk {chunk_id}, score {score_display})")
            lines.append(f"   {text}")

        return (
            "### Retrieval Results\n\n"
            f"- Retrieval ({mode}): `{search_time:.4f}s`\n\n"
            "```text\n"
            + "\n".join(lines)
            + "\n```"
        )

    def on_ask(self, query: str, top_k: float, mode: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Please enter a query first."
        if top_k is None:
            return "Top K must be a number."

        mode = (mode or "").strip().lower() or self.settings.retrieval_mode
        top_k_int = max(1, int(top_k))

        try:
            retriever = build_retriever(
                mode=mode,
                embed_client=self.embed_client,
                vector_store=self.vector_store,
                document_store=self.document_store,
                hybrid_rrf_k=self.settings.hybrid_rrf_k,
                hybrid_candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            )
            results = retriever.retrieve(query, top_k=top_k_int)
        except Exception as exc:
            return f"Search failed: {exc}"

        if not results:
            return "No results found. Have you indexed PDFs yet?"

        context, selected_tender_ids = build_full_tender_context(
            results=results,
            document_store=self.document_store,
            max_unique_tenders=3,
        )

        user_prompt = _build_ask_user_prompt(query, context)

        try:
            answer = self.llm_client.chat(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return f"LLM call failed: {exc}"

        answer = _strip_think_blocks(answer).strip()

        if not _has_citation(answer):
            fallback_sources = []
            if selected_tender_ids:
                fallback_sources = [f"[{tender_id}/full_text#full]" for tender_id in selected_tender_ids]
            else:
                for result in results[:3]:
                    source = _format_citation_source(result)
                    chunk_id = result.get("chunk_id", "n/a")
                    fallback_sources.append(f"[{source}#{chunk_id}]")
            if fallback_sources:
                answer = f"{answer}\n\nCitations: " + ", ".join(dict.fromkeys(fallback_sources))

        return answer

    def on_ask_stream(self, query: str, top_k: float, mode: str):
        query = (query or "").strip()
        if not query:
            yield "Please enter a query first."
            return
        if top_k is None:
            yield "Top K must be a number."
            return

        mode = (mode or "").strip().lower() or self.settings.retrieval_mode
        top_k_int = max(1, int(top_k))

        try:
            retriever = build_retriever(
                mode=mode,
                embed_client=self.embed_client,
                vector_store=self.vector_store,
                document_store=self.document_store,
                hybrid_rrf_k=self.settings.hybrid_rrf_k,
                hybrid_candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            )
            results = retriever.retrieve(query, top_k=top_k_int)
        except Exception as exc:
            yield f"Search failed: {exc}"
            return

        if not results:
            yield "No results found. Have you indexed PDFs yet?"
            return

        context, selected_tender_ids = build_full_tender_context(
            results=results,
            document_store=self.document_store,
            max_unique_tenders=3,
        )

        user_prompt = _build_ask_user_prompt(query, context)
        yield "Generating..."
        partial_answer = ""
        try:
            for token in self.llm_client.stream_chat(SYSTEM_PROMPT, user_prompt):
                partial_answer = _strip_think_blocks(f"{partial_answer}{token}").strip()
                yield partial_answer
            answer = partial_answer
        except Exception:
            try:
                answer = _strip_think_blocks(
                    self.llm_client.chat(SYSTEM_PROMPT, user_prompt)
                ).strip()
            except Exception as exc:
                yield f"LLM call failed: {exc}"
                return

        if not _has_citation(answer):
            fallback_sources = []
            if selected_tender_ids:
                fallback_sources = [f"[{tender_id}/full_text#full]" for tender_id in selected_tender_ids]
            else:
                for result in results[:3]:
                    source = _format_citation_source(result)
                    chunk_id = result.get("chunk_id", "n/a")
                    fallback_sources.append(f"[{source}#{chunk_id}]")
            if fallback_sources:
                answer = f"{answer}\n\nCitations: " + ", ".join(dict.fromkeys(fallback_sources))

        yield answer


def build_demo() -> gr.Blocks:
    app = EmbeddingGradioApp()

    with gr.Blocks(title="Embedding Explorer (Gradio)") as demo:
        gr.Markdown("## Embedding Explorer")
        gr.Markdown(
            "Enter a query to generate its embedding, search indexed PDFs, or ask the LLM."
        )

        query_input = gr.Textbox(
            label="Query",
            lines=8,
            placeholder="Type your enterprise question...",
        )

        with gr.Row():
            top_k = gr.Number(label="Top K", value=20, precision=0, minimum=1)
            mode = gr.Dropdown(
                label="Mode",
                choices=list(RETRIEVAL_MODES),
                value=app.settings.retrieval_mode,
            )

        with gr.Row():
            embed_btn = gr.Button("Generate Embedding", variant="secondary")
            search_btn = gr.Button("Search Vector DB", variant="secondary")
            ask_btn = gr.Button("Ask LLM", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Markdown("### Output")
        output = gr.Markdown()

        gr.Examples(
            examples=[
                ["Summarize key deadlines, owners, and open risks from these documents."],
                ["What decisions are required and what information is still missing?"],
                ["List the top action items with owners and due dates."],
            ],
            inputs=[query_input],
        )

        embed_btn.click(
            fn=app.on_embed,
            inputs=[query_input],
            outputs=[output],
        )
        search_btn.click(
            fn=app.on_search,
            inputs=[query_input, top_k, mode],
            outputs=[output],
        )
        ask_btn.click(
            fn=app.on_ask_stream,
            inputs=[query_input, top_k, mode],
            outputs=[output],
        )
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[query_input, output],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue().launch()


if __name__ == "__main__":
    main()
