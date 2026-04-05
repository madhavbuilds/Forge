"""LiteLLM wrapper with fast/smart switching and streaming."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import acompletion

from forge.config import ForgeConfig, ollama_api_base
from forge.models.cache import PromptCache


def get_model(intent: str, config: ForgeConfig) -> str:
    if intent == "edit_code":
        return config.smart_model
    return config.fast_model


def _should_retry_with_fast(intent: str, exc: Exception) -> bool:
    if intent != "edit_code":
        return False
    text = str(exc).lower()
    hints = (
        "model runner has unexpectedly stopped",
        "resource limitations",
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "internal error",
        "500 internal server error",
    )
    return any(hint in text for hint in hints)


def _streaming_chunk_text(part: Any) -> str:
    """Best-effort text from one LiteLLM stream chunk (Ollama sometimes omits delta.content)."""
    try:
        choices = getattr(part, "choices", None) or []
        if not choices:
            return ""
        c0 = choices[0]
        delta = getattr(c0, "delta", None)
        if delta is not None:
            t = getattr(delta, "content", None) or ""
            if t:
                return str(t)
        msg = getattr(c0, "message", None)
        if msg is not None:
            t = getattr(msg, "content", None) or ""
            if t:
                return str(t)
    except (AttributeError, IndexError, TypeError):
        pass
    if isinstance(part, dict):
        try:
            ch0 = (part.get("choices") or [{}])[0]
            d = ch0.get("delta") or {}
            if isinstance(d, dict) and d.get("content"):
                return str(d["content"])
            m = ch0.get("message") or {}
            if isinstance(m, dict) and m.get("content"):
                return str(m["content"])
        except (KeyError, IndexError, TypeError):
            pass
    return ""


class ModelClient:
    def __init__(
        self,
        cfg: ForgeConfig,
        *,
        on_active_model: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.api_base = ollama_api_base(cfg.ollama_host)
        self._active: str = "fast"
        self._on_active_model = on_active_model
        self.prompt_cache = PromptCache(
            cfg.cache_db_path, cfg.cache_max_entries, cfg.cache_enabled
        )
        litellm.drop_params = True
        litellm.suppress_debug_info = True

    def set_active(self, name: str) -> None:
        self._active = name
        if self._on_active_model is not None:
            self._on_active_model(name)

    def active_label(self) -> str:
        return self._active

    def _model_id(self, intent: str) -> str:
        return get_model(intent, self.cfg)

    def _kwargs(self, intent: str, json_mode: bool) -> dict[str, Any]:
        mid = self._model_id(intent)
        return self._kwargs_for_model(mid, json_mode=json_mode)

    def _kwargs_for_model(self, model_id: str, *, json_mode: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": model_id,
            "api_base": self.api_base,
            "extra_body": {"keep_alive": self.cfg.keep_alive},
        }
        if json_mode and model_id.startswith("ollama/"):
            kw["response_format"] = {"type": "json_object"}
        return kw

    async def _complete(self, messages: list[dict[str, str]], kw: dict[str, Any], stream: bool):
        if stream:
            return await acompletion(messages=messages, stream=True, **kw)
        return await acompletion(messages=messages, stream=False, **kw)

    async def _complete_json_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        intent: str,
        stream_flag: bool,
    ):
        kw = self._kwargs(intent, json_mode=True)
        self.set_active("smart" if intent == "edit_code" else "fast")
        try:
            return await self._complete(messages, kw, stream_flag)
        except Exception:
            kw_no_format = dict(kw)
            kw_no_format.pop("response_format", None)
            try:
                return await self._complete(messages, kw_no_format, stream_flag)
            except Exception as exc:
                if not _should_retry_with_fast(intent, exc):
                    raise
        fast_kw = self._kwargs_for_model(self.cfg.fast_model, json_mode=True)
        self.set_active("fast")
        try:
            return await self._complete(messages, fast_kw, stream_flag)
        except Exception:
            fast_kw_no_format = dict(fast_kw)
            fast_kw_no_format.pop("response_format", None)
            return await self._complete(messages, fast_kw_no_format, stream_flag)

    async def _complete_text_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        intent: str,
        stream: bool,
    ):
        kw = self._kwargs(intent, json_mode=False)
        self.set_active("smart" if intent == "edit_code" else "fast")
        try:
            return await self._complete(messages, kw, stream)
        except Exception as exc:
            if not _should_retry_with_fast(intent, exc):
                raise
        fast_kw = self._kwargs_for_model(self.cfg.fast_model, json_mode=False)
        self.set_active("fast")
        return await self._complete(messages, fast_kw, stream)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        intent: str,
        on_chunk: Callable[[str], None] | None = None,
        on_stream_end: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Return parsed JSON from model (streaming optional)."""
        cache_ctx = json.dumps({"intent": intent, "messages": messages[-3:]}, ensure_ascii=False) if messages else ""
        prompt = messages[-1]["content"] if messages else ""

        pc = self.prompt_cache
        cached = pc.get(prompt, cache_ctx)
        if cached:
            data = json.loads(cached)
            if on_stream_end is not None:
                on_stream_end()
            return data

        stream_flag = bool(self.cfg.streaming and on_chunk)
        resp = await self._complete_json_with_fallback(
            messages,
            intent=intent,
            stream_flag=stream_flag,
        )
        if stream_flag:
            text = ""
            async for part in resp:
                chunk = _streaming_chunk_text(part)
                if chunk:
                    text += chunk
                    on_chunk(chunk)
            data = self._parse_json_block(text)
        else:
            text = resp.choices[0].message.content or ""
            data = self._parse_json_block(text)

        if on_stream_end is not None:
            on_stream_end()

        pc.set(prompt, cache_ctx, json.dumps(data, ensure_ascii=False))
        return data

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        intent: str,
    ) -> str:
        resp = await self._complete_text_with_fallback(messages, intent=intent, stream=False)
        return (resp.choices[0].message.content or "").strip()

    async def complete_text_stream(
        self,
        messages: list[dict[str, str]],
        *,
        intent: str,
        on_chunk: Callable[[str], None],
        on_stream_end: Callable[[], None] | None = None,
    ) -> str:
        text = ""
        stream = await self._complete_text_with_fallback(messages, intent=intent, stream=True)
        async for part in stream:
            chunk = _streaming_chunk_text(part)
            if chunk:
                text += chunk
                on_chunk(chunk)
        if on_stream_end is not None:
            on_stream_end()
        return text

    def _parse_json_block(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {"error": "empty", "tool_calls": [], "done": True}
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"error": "invalid_json", "raw": text[:2000], "tool_calls": [], "done": True}
