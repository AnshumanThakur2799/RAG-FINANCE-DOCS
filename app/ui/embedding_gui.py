from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import time

import time

from app.config import Settings
from app.db.vector_store import LanceDBVectorStore
from app.embeddings.client import build_embedding_client
from app.llm.client import build_llm_client


SYSTEM_PROMPT = (
    "You are a finance department assistant. Answer using only the provided context. "
    "If the context is insufficient, say you do not have enough information. "
    "Always include citations in the form [source_name#chunk_id]."
)


class EmbeddingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Embedding Explorer")
        self.root.geometry("760x640")

        settings = Settings.from_env()
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
        )
        self.vector_store = LanceDBVectorStore(settings.lancedb_dir)
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

        search_button = ttk.Button(button_row, text="Search LanceDB", command=self.on_search)
        search_button.pack(side=tk.LEFT, padx=(8, 0))

        ask_button = ttk.Button(button_row, text="Ask LLM", command=self.on_ask)
        ask_button.pack(side=tk.LEFT, padx=(8, 0))

        topk_label = ttk.Label(button_row, text="Top K:")
        topk_label.pack(side=tk.LEFT, padx=(16, 4))

        self.topk_var = tk.StringVar(value="3")
        topk_entry = ttk.Entry(button_row, textvariable=self.topk_var, width=4)
        topk_entry.pack(side=tk.LEFT)

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

        try:
            embedding = self.embed_client.embed([query], input_type="query")[0]
            # get the search time 
            start_time = time.time()
            results = self.vector_store.search(embedding, top_k=top_k)
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
            lines.append(f"{idx}. {source} (chunk {chunk_id}, score {score_display})")
            lines.append(f"   {text}")
        
        output = (
            f"Vector search time: {search_time:.4f} seconds\n\n"
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

        try:
            embedding = self.embed_client.embed([query], input_type="query")[0]
            search_start = time.perf_counter()
            results = self.vector_store.search(embedding, top_k=top_k)
            search_elapsed = time.perf_counter() - search_start
        except Exception as exc:
            self._write_output(f"Search failed: {exc}")
            return

        if not results:
            self._write_output("No results found. Have you indexed PDFs yet?")
            return

        context_lines = []
        for result in results:
            source = result.get("source_name") or result.get("source_path", "unknown")
            chunk_id = result.get("chunk_id", "n/a")
            text = (result.get("text") or "").strip()
            context_lines.append(f"[{source}#{chunk_id}] {text}")
        context = "\n\n".join(context_lines)

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer with citations."
        )

        try:
            llm_start = time.perf_counter()
            answer = self.llm_client.chat(SYSTEM_PROMPT, user_prompt)
            llm_elapsed = time.perf_counter() - llm_start
        except Exception as exc:
            self._write_output(f"LLM call failed: {exc}")
            return

        output = (
            f"Vector search time: {search_elapsed:.4f} seconds\n\n"
            f"Answer:\n{answer} \n\n"
            f"LLM call time: {llm_elapsed:.4f} seconds"
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
