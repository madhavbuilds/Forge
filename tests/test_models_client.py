"""Model routing rules."""

from pathlib import Path

from forge.config import ForgeConfig
from forge.models.client import _should_retry_with_fast, get_model


def _config() -> ForgeConfig:
    return ForgeConfig(
        fast_model="fast-model",
        smart_model="smart-model",
        ollama_host="http://localhost:11434",
        keep_alive="1h",
        max_tool_calls=3,
        auto_checkpoint=True,
        max_fix_retries=2,
        confirm_writes=True,
        confirm_commands=True,
        streaming=True,
        show_tool_calls=True,
        theme="dark",
        compact_mode=False,
        cache_enabled=True,
        cache_db_path=Path("/tmp/forge-cache.db"),
        cache_max_entries=100,
        raw={},
    )


def test_get_model_uses_smart_only_for_edit_code() -> None:
    cfg = _config()
    assert get_model("edit_code", cfg) == "smart-model"
    assert get_model("search", cfg) == "fast-model"
    assert get_model("explain", cfg) == "fast-model"


def test_resource_error_retries_with_fast_for_edit_code() -> None:
    exc = Exception("OllamaException - model runner has unexpectedly stopped due to resource limitations")
    assert _should_retry_with_fast("edit_code", exc) is True


def test_non_edit_requests_do_not_retry_with_fast() -> None:
    exc = Exception("500 Internal Server Error")
    assert _should_retry_with_fast("search", exc) is False
