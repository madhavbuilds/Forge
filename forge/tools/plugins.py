"""Load custom tools from ~/.forge/tools/*.py"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from forge.config import FORGE_DIR

ToolFn = Callable[..., Any]


def load_plugin_tools() -> dict[str, ToolFn]:
    root = FORGE_DIR / "tools"
    root.mkdir(parents=True, exist_ok=True)
    registry: dict[str, ToolFn] = {}
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = _load_module(path)
        if mod is None:
            continue
        tools = getattr(mod, "FORGE_TOOLS", None) or getattr(mod, "tools", None)
        if isinstance(tools, dict):
            for name, fn in tools.items():
                if callable(fn):
                    registry[str(name)] = fn
    return registry


def _load_module(path: Path) -> ModuleType | None:
    name = f"forge_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        return None
    return mod
