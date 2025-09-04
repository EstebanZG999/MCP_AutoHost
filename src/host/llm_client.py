# MCP_AutoHost - LLM client wrapper (Anthropic and OpenAI)
from __future__ import annotations
import os
from .memory import Memory
from .logging_utils import log_event

class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        # Auto-detect provider
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
        ant = os.getenv("ANTHROPIC_API_KEY")
        oai = os.getenv("OPENAI_API_KEY")
        if not self.provider or self.provider == "auto":
            if ant: self.provider = "anthropic"
            elif oai: self.provider = "openai"
            else: self.provider = "anthropic"  # default preference

        # Default model
        self.model = model or os.getenv("LLM_MODEL") or (
            "claude-3-7-sonnet-latest" if self.provider == "anthropic" else "gpt-4o-mini"
        )
        self._anthropic = None
        self._openai = None

    def _ensure_client(self):
        if self.provider == "anthropic" and self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "openai" and self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, mem: Memory, max_tokens: int = 500, temperature: float = 0.2) -> str:
        """Send the entire history in memory. DO NOT add the last user again here."""
        self._ensure_client()
        if self.provider == "anthropic":
            return self._chat_anthropic(mem, max_tokens, temperature)
        elif self.provider == "openai":
            return self._chat_openai(mem, max_tokens, temperature)
        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

    # --- Anthropic ---
    def _chat_anthropic(self, mem: Memory, max_tokens: int, temperature: float) -> str:
        client = self._anthropic
        system = mem.messages[0].content if mem.messages and mem.messages[0].role == "system" else None
        messages = mem.export_for_anthropic()  # already includes the last user
        log_event("llm_request", provider="anthropic", model=self.model, messages=len(messages))
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        # Extract text from blocks
        text = "".join([blk.text for blk in getattr(resp, "content", []) if getattr(blk, "type", "") == "text"]) or str(resp)
        log_event("llm_response", provider="anthropic", model=self.model, chars=len(text))
        return text

    # --- OpenAI ---
    def _chat_openai(self, mem: Memory, max_tokens: int, temperature: float) -> str:
        client = self._openai
        messages = mem.export_for_openai()  # already includes system + last user
        log_event("llm_request", provider="openai", model=self.model, messages=len(messages))
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        log_event("llm_response", provider="openai", model=self.model, chars=len(text))
        return text
