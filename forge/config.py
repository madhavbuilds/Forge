"""Load ~/.forge/config.toml with defaults."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

FORGE_DIR = Path.home() / ".forge"
CONFIG_PATH = FORGE_DIR / "config.toml"

DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "fast": "ollama/qwen2.5-coder:1.5b",
        "smart": "ollama/qwen2.5-coder:3b",
        "ollama_host": "http://localhost:11434",
        "keep_alive": "1h",
    },
    "agent": {
        "max_tool_calls": 3,
        "auto_checkpoint": True,
        "max_fix_retries": 2,
        "confirm_writes": True,
        "confirm_commands": True,
    },
    "ui": {
        "streaming": True,
        "show_tool_calls": False,
        "theme": "dark",
        "compact_mode": False,
    },
    "cache": {
        "enabled": True,
        "db_path": "~/.forge/cache.db",
        "max_entries": 10000,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand(p: str) -> str:
    return str(Path(p).expanduser().resolve())


@dataclass
class ForgeConfig:
    fast_model: str
    smart_model: str
    ollama_host: str
    keep_alive: str
    max_tool_calls: int
    auto_checkpoint: bool
    max_fix_retries: int
    confirm_writes: bool
    confirm_commands: bool
    streaming: bool
    show_tool_calls: bool
    theme: str
    compact_mode: bool
    cache_enabled: bool
    cache_db_path: Path
    cache_max_entries: int
    raw: dict[str, Any] = field(default_factory=dict)

    def apply_env(self) -> None:
        # Keeps local Ollama models loaded longer between requests (client also sends keep_alive).
        os.environ["OLLAMA_KEEP_ALIVE"] = self.keep_alive


def ensure_config_file() -> None:
    FORGE_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        lines = [
            '[models]',
            'fast = "ollama/qwen2.5-coder:1.5b"',
            'smart = "ollama/qwen2.5-coder:3b"',
            'ollama_host = "http://localhost:11434"',
            'keep_alive = "1h"',
            "",
            "[agent]",
            "max_tool_calls = 3",
            "auto_checkpoint = true",
            "max_fix_retries = 2",
            "confirm_writes = true",
            "confirm_commands = true",
            "",
            "[ui]",
            "streaming = true",
            "show_tool_calls = false",
            'theme = "dark"',
            "compact_mode = false",
            "",
            "[cache]",
            "enabled = true",
            'db_path = "~/.forge/cache.db"',
            "max_entries = 10000",
            "",
        ]
        CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")


def warn_if_fast_model_below_floor(fast_model: str) -> None:
    """
    Forge expects at least ~1.5B coder-class models for reliable JSON tool use.
    Warn (stderr) when the configured fast model likely falls below that floor.
    """
    s = fast_model.lower()
    # Explicit sub-1.5B tags (Ollama-style :Xb)
    m = re.search(r":(\d+(?:\.\d+)?)\s*b\b", s)
    if m:
        try:
            billions = float(m.group(1))
            if billions < 1.5:
                print(
                    "[forge:warn] `fast` model tag suggests <1.5B parameters — below Forge's recommended floor "
                    "(use at least ollama/qwen2.5-coder:1.5b). Smaller models often mis-read tool results.",
                    file=sys.stderr,
                    flush=True,
                )
                return
        except ValueError:
            pass
    if "qwen" in s and "instruct" in s and "coder" not in s and "1.5" not in s and "3b" not in s and "7b" not in s:
        print(
            "[forge:warn] `fast` is a small instruct model without `coder` / 1.5b+ in the name — "
            "JSON tool calling may be unreliable. Prefer ollama/qwen2.5-coder:1.5b.",
            file=sys.stderr,
            flush=True,
        )


def load_config(path: Path | None = None) -> ForgeConfig:
    ensure_config_file()
    cfg_path = path or CONFIG_PATH
    data = dict(DEFAULT_CONFIG)
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            loaded = tomllib.load(f)
        data = _deep_merge(data, loaded)

    m = data["models"]
    a = data["agent"]
    u = data["ui"]
    c = data["cache"]
    db = Path(c["db_path"]).expanduser()
    warn_if_fast_model_below_floor(str(m["fast"]))
    return ForgeConfig(
        fast_model=m["fast"],
        smart_model=m["smart"],
        ollama_host=m["ollama_host"],
        keep_alive=m.get("keep_alive", "1h"),
        max_tool_calls=int(a.get("max_tool_calls", a.get("max_tool_loops", 3))),
        auto_checkpoint=bool(a["auto_checkpoint"]),
        max_fix_retries=int(a.get("max_fix_retries", 2)),
        confirm_writes=bool(a["confirm_writes"]),
        confirm_commands=bool(a["confirm_commands"]),
        streaming=bool(u["streaming"]),
        show_tool_calls=bool(u["show_tool_calls"]),
        theme=str(u["theme"]),
        compact_mode=bool(u["compact_mode"]),
        cache_enabled=bool(c["enabled"]),
        cache_db_path=db,
        cache_max_entries=int(c["max_entries"]),
        raw=data,
    )


def ollama_api_base(host: str) -> str:
    h = host.rstrip("/")
    if not h.startswith("http"):
        h = "http://" + h
    return h
