from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk
import time
from datetime import datetime

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import build_vector_store
from app.embeddings.client import build_embedding_client
from app.llm.client import build_llm_client
from app.retrieval import RETRIEVAL_MODES, build_full_tender_context, build_retriever


SYSTEM_PROMPT = """You are an internal Enterprise AI Assistant for business teams. Use only the provided documents (uploaded files and parsed chunks), and never invent facts.

Mandatory rules:
- Answer only what the user asked.
- Cite factual claims using [source#page_or_chunk], with up to the 5 most relevant citations.
- If context is insufficient, explicitly say "Insufficient information in the provided documents."
- Keep responses concise, factual, and operational.
- Do not force templates, JSON, or extra sections unless the user asks for them.
- No chain-of-thought, no hidden reasoning, no XML tags.

Token/length control:
- Default budget is 900 tokens unless user requests otherwise.
- If content is too long, prioritize the direct answer first.
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


class EmbeddingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Embedding Explorer")
        self.root.geometry("760x640")

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
            local_device=settings.local_embedding_device,
            deepinfra_base_url=settings.deepinfra_base_url,
        )
        self.document_store = SQLiteDocumentStore(settings.sqlite_db_path)
        self.vector_store = build_vector_store(settings, table_name="document_chunks")
        self.llm_client = build_llm_client(
            settings.llm_provider,
            deepinfra_api_key=settings.deepinfra_api_key,
            deepinfra_base_url=settings.deepinfra_base_url,
            deepinfra_model=settings.deepinfra_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            container,
            text="Enter a query to generate its embedding or search indexed PDFs:",
            font=("Segoe UI", 11),
        )
        header.pack(anchor=tk.W)

        self.input_text = tk.Text(container, height=8, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=False, pady=(8, 8))

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(0, 8))

        embed_button = ttk.Button(button_row, text="Generate Embedding", command=self.on_embed)
        embed_button.pack(side=tk.LEFT)

        search_button = ttk.Button(button_row, text="Search Vector DB", command=self.on_search)
        search_button.pack(side=tk.LEFT, padx=(8, 0))

        ask_button = ttk.Button(button_row, text="Ask LLM", command=self.on_ask)
        ask_button.pack(side=tk.LEFT, padx=(8, 0))

        topk_label = ttk.Label(button_row, text="Top K:")
        topk_label.pack(side=tk.LEFT, padx=(16, 4))

        self.topk_var = tk.StringVar(value="5")
        topk_entry = ttk.Entry(button_row, textvariable=self.topk_var, width=4)
        topk_entry.pack(side=tk.LEFT)

        mode_label = ttk.Label(button_row, text="Mode:")
        mode_label.pack(side=tk.LEFT, padx=(16, 4))

        self.mode_var = tk.StringVar(value=self.settings.retrieval_mode)
        mode_combo = ttk.Combobox(
            button_row,
            textvariable=self.mode_var,
            width=9,
            values=list(RETRIEVAL_MODES),
            state="readonly",
        )
        mode_combo.pack(side=tk.LEFT)

        clear_button = ttk.Button(button_row, text="Clear", command=self.on_clear)
        clear_button.pack(side=tk.LEFT, padx=(8, 0))

        output_label = ttk.Label(container, text="Output:")
        output_label.pack(anchor=tk.W)

        self.output_text = tk.Text(container, height=18, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def on_embed(self) -> None:
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            self._write_output("Please enter a query first.")
            return

        try:
            embedding = self.embed_client.embed([query], input_type="query")[0]
        except Exception as exc:
            self._write_output(f"Embedding failed: {exc}")
            return

        preview = ", ".join(f"{value:.6f}" for value in embedding[:12])
        message = (
            f"Embedding length: {len(embedding)}\n"
            f"First 12 values: [{preview}]\n"
        )
        self._write_output(message)

    def on_search(self) -> None:
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            self._write_output("Please enter a query first.")
            return

        try:
            top_k = max(1, int(self.topk_var.get().strip()))
        except ValueError:
            self._write_output("Top K must be a number.")
            return
        mode = self.mode_var.get().strip().lower() or self.settings.retrieval_mode

        try:
            retriever = build_retriever(
                mode=mode,
                embed_client=self.embed_client,
                vector_store=self.vector_store,
                document_store=self.document_store,
                hybrid_rrf_k=self.settings.hybrid_rrf_k,
                hybrid_candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            )
            start_time = time.time()
            results = retriever.retrieve(query, top_k=top_k)
            end_time = time.time()
            search_time = end_time - start_time
        except Exception as exc:
            self._write_output(f"Search failed: {exc}")
            return

        if not results:
            self._write_output("No results found. Have you indexed PDFs yet?")
            return

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
        
        output = (
            f"Retrieval time ({mode}): {search_time:.4f} seconds\n\n"
            "\n".join(lines)
        )
        self._write_output(output)

    def on_ask(self) -> None:
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            self._write_output("Please enter a query first.")
            return

        try:
            top_k = max(1, int(self.topk_var.get().strip()))
        except ValueError:
            self._write_output("Top K must be a number.")
            return
        mode = self.mode_var.get().strip().lower() or self.settings.retrieval_mode

        try:
            retriever = build_retriever(
                mode=mode,
                embed_client=self.embed_client,
                vector_store=self.vector_store,
                document_store=self.document_store,
                hybrid_rrf_k=self.settings.hybrid_rrf_k,
                hybrid_candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            )
            search_start = time.perf_counter()
            results = retriever.retrieve(query, top_k=top_k)
            search_elapsed = time.perf_counter() - search_start
        except Exception as exc:
            self._write_output(f"Search failed: {exc}")
            return

        if not results:
            self._write_output("No results found. Have you indexed PDFs yet?")
            return

        context, selected_tender_ids = build_full_tender_context(
            results=results,
            document_store=self.document_store,
            max_unique_tenders=3,
        )

        # parse now date and time 
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer directly in plain final form only (no <think> tags). "
            "Include citations for factual statements in the format "
            "[source_name#chunk_id]."
            f"Current date and time: {now}"
        )

        try:
            llm_start = time.perf_counter()
            answer = self.llm_client.chat(SYSTEM_PROMPT, user_prompt)
            llm_elapsed = time.perf_counter() - llm_start
        except Exception as exc:
            self._write_output(f"LLM call failed: {exc}")
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
                answer = (
                    f"{answer}\n\nCitations: "
                    + ", ".join(dict.fromkeys(fallback_sources))
                )

        output = (
            f"Retrieval time ({mode}): {search_elapsed:.4f} seconds\n\n"
            f"Answer:\n{answer} \n\n"
            f"LLM call time: {llm_elapsed:.4f} seconds\n\n"
        )
        self._write_output(output)

    def on_clear(self) -> None:
        self.input_text.delete("1.0", tk.END)
        self._write_output("")

    def _write_output(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    EmbeddingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
