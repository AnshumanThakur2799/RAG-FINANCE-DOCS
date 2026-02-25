from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


@dataclass(frozen=True)
class TokenChunker:
    chunk_size: int = 500
    overlap: int = 100
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
    length_kind: str = "token"
    encoding_name: str = "cl100k_base"

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        # Default to token-aware splitting for consistent embedding payload sizes.
        if self.length_kind == "token":
            encoding = tiktoken.get_encoding(self.encoding_name)
            splitter = RecursiveCharacterTextSplitter(
                length_function=lambda chunk_text: len(encoding.encode(chunk_text)),
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=list(self.separators),
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=list(self.separators),
            )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]