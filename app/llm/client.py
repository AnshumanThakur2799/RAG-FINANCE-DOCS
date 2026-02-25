from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def stream_chat(self, system_prompt: str, user_prompt: str):
        raise NotImplementedError

    def chat_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        pdf_path: str | Path,
    ) -> str:
        raise NotImplementedError

    def chat_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
    ) -> str:
        raise NotImplementedError


@dataclass
class DeepInfraChatClient:
    api_key: str
    base_url: str
    model: str
    max_tokens: int | None = None
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _sanitize_response(raw: str) -> str:
        # Some reasoning models return internal reasoning in <think> blocks.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
        return cleaned.strip()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens or None,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return self._sanitize_response(content)

    def stream_chat(self, system_prompt: str, user_prompt: str):
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens or None,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield str(delta)

    def chat_with_pdf(
        self,
        system_prompt: str,
        user_prompt: str,
        pdf_path: str | Path,
    ) -> str:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")

        encoded_pdf = base64.b64encode(path.read_bytes()).decode("ascii")
        pdf_data_url = f"data:application/pdf;base64,{encoded_pdf}"

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {
                                "type": "input_file",
                                "filename": path.name,
                                "file_data": pdf_data_url,
                            },
                        ],
                    },
                ],
                max_output_tokens=self.max_tokens or None,
                temperature=self.temperature,
            )
            output_text = (getattr(response, "output_text", None) or "").strip()
            if output_text:
                return self._sanitize_response(output_text)
            return self._sanitize_response(str(response))
        except Exception as exc:
            raise RuntimeError(
                "Direct PDF input is not accepted by the configured endpoint/model."
            ) from exc

    def chat_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
    ) -> str:
        if not image_data_urls:
            raise ValueError("image_data_urls cannot be empty.")

        user_content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
        for data_url in image_data_urls:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self.max_tokens or None,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return self._sanitize_response(content)


def build_llm_client(
    provider: str,
    *,
    deepinfra_api_key: str | None,
    deepinfra_base_url: str,
    deepinfra_model: str,
    max_tokens: int | None = None,
    temperature: float,
) -> LLMClient:
    provider_normalized = provider.strip().lower()
    if provider_normalized in {"deepinfra", "openai-compatible"}:
        if not deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is required for DeepInfra.")
        return DeepInfraChatClient(
            api_key=deepinfra_api_key,
            base_url=deepinfra_base_url,
            model=deepinfra_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
