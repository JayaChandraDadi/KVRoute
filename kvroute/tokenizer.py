"""Tokenizers the router uses to turn a request into token ids.

* ``HFTokenizer`` -- the served model's Hugging Face tokenizer, including its
  chat template, so router-side block hashes line up with vLLM's blocks.
* ``WordTokenizer`` -- one token per whitespace-separated word. Used by the
  mock replica and in tests so everything runs without downloading a model;
  router and mock replica must use the same one.
"""

from __future__ import annotations

import zlib
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode_prompt(self, prompt: str | list[int]) -> list[int]: ...

    def encode_chat(self, messages: list[dict[str, Any]]) -> list[int]: ...


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class WordTokenizer:
    vocab_size = 50_000

    def _word(self, w: str) -> int:
        return zlib.crc32(w.encode()) % self.vocab_size

    def encode_prompt(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, list):
            return [int(t) for t in prompt]
        return [self._word(w) for w in prompt.split()]

    def encode_chat(self, messages: list[dict[str, Any]]) -> list[int]:
        tokens: list[int] = []
        for m in messages:
            tokens.append(self._word(f"<|{m.get('role', 'user')}|>"))
            tokens.extend(self._word(w) for w in _message_text(m.get("content")).split())
        tokens.append(self._word("<|assistant|>"))
        return tokens


class HFTokenizer:
    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model)

    def encode_prompt(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, list):
            return [int(t) for t in prompt]
        return self._tok.encode(prompt)

    def encode_chat(self, messages: list[dict[str, Any]]) -> list[int]:
        msgs = [{"role": m.get("role", "user"), "content": _message_text(m.get("content"))} for m in messages]
        return list(self._tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))


def load_tokenizer(spec: str) -> Tokenizer:
    """``word`` or ``hf:<model id or path>``."""
    if spec == "word":
        return WordTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec {spec!r} (use 'word' or 'hf:<model>')")
