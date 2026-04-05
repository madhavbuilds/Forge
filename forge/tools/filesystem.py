"""read_file, write_file, list_directory."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from forge.utils import ast_parser


@dataclass
class FilesystemState:
    read_paths: set[str] = field(default_factory=set)
    listed_dirs: set[str] = field(default_factory=set)

    def mark_read(self, path: str) -> None:
        self.read_paths.add(os.path.normpath(path))

    def mark_listed(self, path: str) -> None:
        self.listed_dirs.add(os.path.normpath(path))


def read_file(path: str, *, repo_root: Path, state: FilesystemState) -> dict:
    p = (repo_root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {p}"}
    if not p.is_file():
        return {"ok": False, "error": "not a file"}
    try:
        rel = str(p.relative_to(repo_root.resolve()))
    except ValueError:
        rel = str(p)
    state.mark_read(rel)
    snippet = ast_parser.extract_definitions_snippet(p)
    return {
        "ok": True,
        "path": rel,
        "content": snippet,
        "truncated": len(snippet) < p.stat().st_size,
    }


def write_file(
    path: str,
    content: str,
    *,
    repo_root: Path,
    state: FilesystemState,
    dry_run: bool = False,
) -> dict:
    rel = path
    p = (repo_root / path).resolve()
    norm = os.path.normpath(rel)
    parent = os.path.normpath(str(Path(rel).parent))
    allowed = (
        norm in state.read_paths
        or str(p) in state.read_paths
        or parent in state.listed_dirs
        or norm in state.listed_dirs
    )
    if not allowed:
        return {
            "ok": False,
            "error": "read-before-write: read_file that path or list_directory its parent first",
            "path": rel,
        }
    if dry_run:
        return {"ok": True, "dry_run": True, "path": rel, "bytes": len(content.encode("utf-8"))}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": rel, "bytes": len(content.encode("utf-8"))}


def list_directory(
    path: str, depth: int, *, repo_root: Path, state: FilesystemState | None = None
) -> dict:
    root = (repo_root / path).resolve() if path else repo_root.resolve()
    if not root.exists():
        return {"ok": False, "error": "path does not exist"}
    lines: list[str] = []

    def walk(cur: Path, d: int, prefix: str) -> None:
        if d > depth:
            return
        try:
            entries = sorted(cur.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        for e in entries:
            if e.name.startswith(".git") or e.name in (".forge", "__pycache__", ".venv", "node_modules"):
                continue
            try:
                st = e.stat()
                mode = "d" if e.is_dir() else "-"
                size = st.st_size if e.is_file() else 0
                mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
                lines.append(f"{prefix}{mode} {e.name}\t{size}\t{mtime}")
            except OSError:
                lines.append(f"{prefix}? {e.name}")
            if e.is_dir() and d < depth:
                walk(e, d + 1, prefix + "  ")

    walk(root, 0, "")
    try:
        shown = str(root.relative_to(repo_root.resolve()))
    except ValueError:
        shown = str(root)
    if state is not None:
        state.mark_listed(shown)
    return {"ok": True, "path": shown, "tree": "\n".join(lines)}
