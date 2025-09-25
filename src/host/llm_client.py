# src/host/llm_client.py
import os
import json
import asyncio
from typing import Any, Dict, List, Optional

class LLMClient:
    """
    Cliente LLM con interfaz async y soporte para:
      - Anthropic (mensajes.create)
      - OpenAI (chat.completions.create)
    Selección por env: LLM_PROVIDER = 'anthropic' | 'openai' (default: 'openai' si hay OPENAI_API_KEY; si no, anthropic).
    Modelo:
      - OPENAI_MODEL (default: gpt-4o-mini)
      - ANTHROPIC_MODEL (default: claude-3-7-sonnet-20250219)
    """

    def __init__(self):
        provider_env = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        self.provider = provider_env or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai")

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except Exception as e:
                raise RuntimeError("anthropic SDK no instalado. pip install anthropic") from e
            self._sdk = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        else:
            # openai por defecto
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("openai SDK no instalado. pip install openai") from e
            self._sdk = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # -------------------- Helpers internos --------------------

    def _anthropic_messages_create(self, system: Optional[str], messages: List[Dict[str, str]], temperature: float) -> str:
        """Llamada síncrona a Anthropic; devuelve el texto plano."""
        # Anthropic espera system y una lista de mensajes con roles user/assistant
        resp = self._sdk.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=temperature,
            system=system or None,
            messages=messages,
        )
        # Unir todos los bloques de texto
        parts = []
        for blk in resp.content:
            # blk.type suele ser "text"
            txt = getattr(blk, "text", None)
            if txt:
                parts.append(txt)
        return "".join(parts).strip()

    def _openai_chat_create(self, system: Optional[str], messages: List[Dict[str, str]], temperature: float) -> str:
        """Llamada síncrona a OpenAI; devuelve el texto plano."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        resp = self._sdk.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    async def _run_sync(self, fn, *args, **kwargs):
        """Ejecuta una función bloqueante en un hilo para exponer interfaz async."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # -------------------- API pública usada por el orquestador --------------------

    async def complete_json(self, system: str, user: str, temperature: float, json_fallback: Dict[str, Any]):
        """
        Pide al modelo que devuelva SOLO JSON. Si falla el parseo, devuelve json_fallback.
        """
        if self.provider == "anthropic":
            text = await self._run_sync(
                self._anthropic_messages_create,
                system,
                self._normalize_messages([{"role": "user", "content": user}]),
                temperature,
            )
        else:
            text = await self._run_sync(
                self._openai_chat_create,
                system,
                self._normalize_messages([{"role": "user", "content": user}]),
                temperature,
            )
        try:
            return json.loads(text)
        except Exception:
            return json_fallback

    async def complete_text(self, system: str, user: str, temperature: float) -> str:
        if self.provider == "anthropic":
            text = await self._run_sync(
                self._anthropic_messages_create,
                system,
                self._normalize_messages([{"role": "user", "content": user}]),
                temperature,
            )
        else:
            text = await self._run_sync(
                self._openai_chat_create,
                system,
                self._normalize_messages([{"role": "user", "content": user}]),
                temperature,
            )
        return text  # <-- Faltaba esto

    async def chat_turn(self, system: str, history: List[Dict[str, str]], user: str, temperature: float) -> str:
        """
        Chat con historial. history: [{"role":"user"|"assistant","content": "..."}]
        """
        msgs = self._normalize_messages(history) + [{"role": "user", "content": user}]
        if self.provider == "anthropic":
            return await self._run_sync(self._anthropic_messages_create, system, msgs, temperature)
        else:
            return await self._run_sync(self._openai_chat_create, system, msgs, temperature)

    def _coerce_msg(self, m) -> Dict[str, str]:
        """
        Convierte un item de historial a {"role": "...", "content": "..."}.
        Acepta dicts, objetos con atributos .role/.content, o strings.
        """
        if isinstance(m, dict):
            role = str(m.get("role", "user"))
            content = m.get("content")
        else:
            # objeto con atributos?
            role = getattr(m, "role", "user")
            content = getattr(m, "content", None)
            if content is None:
                # como último recurso, stringify
                content = str(m)

        role = str(role).lower()
        if role not in ("user", "assistant"):
            role = "user"
        # CONTENT debe ser string
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        return {"role": role, "content": content}

    def _normalize_messages(self, messages_raw) -> list[dict]:
        """
        Asegura una lista de dicts role/content, sin objetos custom.
        """
        return [self._coerce_msg(m) for m in messages_raw or []]