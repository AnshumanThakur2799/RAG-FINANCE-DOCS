from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class TokenChunker:
    chunk_size: int = 500
    overlap: int = 50
    encoding_name: str = "cl100k_base"

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        chunks: list[str] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(encoding.decode(chunk_tokens))
            if end == len(tokens):
                break
            start = max(end - self.overlap, 0)
        return chunks
