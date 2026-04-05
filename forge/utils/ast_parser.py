"""tree-sitter helpers: extract defs, trim to relevant nodes."""

from __future__ import annotations

import ctypes
import warnings
from functools import lru_cache
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from tree_sitter import Language, Parser
    from tree_sitter_languages import core as tree_sitter_core

LANGUAGE_LIBRARY = Path(tree_sitter_core.__file__).with_name("languages.so")
pythonquery_str = "(function_definition name: (identifier) @name)"


def _lang_for_path(path: Path) -> str | None:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".hpp": "cpp",
        ".rb": "ruby",
    }.get(ext)


@lru_cache(maxsize=1)
def _language_library() -> ctypes.CDLL:
    return ctypes.CDLL(str(LANGUAGE_LIBRARY))


@lru_cache(maxsize=None)
def _language(name: str) -> Language:
    factory = getattr(_language_library(), f"tree_sitter_{name}")
    factory.restype = ctypes.c_void_p
    ptr = factory()
    return Language(ptr, name)


@lru_cache(maxsize=None)
def _parser(name: str) -> Parser:
    parser = Parser()
    parser.set_language(_language(name))
    return parser


def parse_file(path: Path, source: bytes | None = None) -> tuple[object, object] | tuple[None, None]:
    lang = _lang_for_path(path)
    if not lang:
        return None, None
    try:
        parser = _parser(lang)
        language = _language(lang)
    except Exception:
        return None, None
    if source is None:
        source = path.read_bytes()
    tree = parser.parse(source)
    return tree, language


def extract_definitions_snippet(path: Path, max_chars: int = 12000) -> str:
    """Prefer Python function slices for model context."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if _lang_for_path(path) != "python":
        return text[:max_chars]

    tree, language = parse_file(path, text.encode("utf-8"))
    if tree is None or language is None:
        return text[:max_chars]

    query = language.query(pythonquery_str)
    captures = query.captures(tree.root_node)
    chunks: list[str] = []
    seen: set[tuple[int, int]] = set()
    for node, cap in captures:
        if cap != "name" or node.parent is None:
            continue
        parent = node.parent
        key = (parent.start_byte, parent.end_byte)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(text[parent.start_byte : parent.end_byte])
    body = "\n\n".join(chunks) if chunks else text
    return body[:max_chars]


def find_symbol_definition(path: Path, symbol: str) -> list[str]:
    if _lang_for_path(path) != "python":
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    tree, language = parse_file(path, text.encode("utf-8"))
    out: list[str] = []
    if tree is None or language is None:
        return out

    query = language.query(pythonquery_str)
    for node, cap in query.captures(tree.root_node):
        if cap != "name":
            continue
        name = text[node.start_byte : node.end_byte]
        if name == symbol and node.parent is not None:
            parent = node.parent
            out.append(text[parent.start_byte : parent.end_byte])
    return out
