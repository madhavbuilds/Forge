"""search_codebase (rg + FAISS), find_definition."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from forge.utils import ast_parser


def _run_rg(repo: Path, query: str, limit: int = 80) -> list[dict]:
    try:
        proc = subprocess.run(
            [
                "rg",
                "--json",
                "-n",
                "--max-count",
                str(limit),
                query,
                str(repo),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    hits: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line.startswith('{"type":"match"'):
            continue
        try:
            obj = json.loads(line)
            d = obj.get("data", {})
            path = d.get("path", {}).get("text", "")
            lines = d.get("lines", {}).get("text", "")
            ln = d.get("line_number")
            if path:
                hits.append({"path": path, "line": ln, "text": lines})
        except json.JSONDecodeError:
            continue
    return hits


def search_codebase(
    query: str,
    *,
    repo_root: Path,
    faiss_hits: list[dict] | None = None,
) -> dict:
    rg = _run_rg(repo_root, query)
    semantic = faiss_hits or []
    return {
        "ok": True,
        "ripgrep": rg[:50],
        "semantic": semantic[:20],
    }


def find_definition(symbol: str, *, repo_root: Path) -> dict:
    results: list[dict] = []
    for p in repo_root.rglob("*"):
        if p.is_dir() or "node_modules" in p.parts or ".git" in p.parts:
            continue
        if p.suffix.lower() not in (
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".java",
        ):
            continue
        try:
            defs = ast_parser.find_symbol_definition(p, symbol)
        except OSError:
            continue
        for d in defs:
            try:
                rel = str(p.relative_to(repo_root))
            except ValueError:
                rel = str(p)
            results.append({"path": rel, "definition": d[:4000]})
    return {"ok": True, "symbol": symbol, "matches": results[:30]}
