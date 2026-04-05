"""Session notes + FAISS semantic index over the repo."""

from __future__ import annotations

import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path

from forge.config import FORGE_DIR

INDEX_DIR = FORGE_DIR / "faiss"
META_FILE = INDEX_DIR / "meta.json"
INDEX_FILE = INDEX_DIR / "index.faiss"
VECS_FILE = INDEX_DIR / "vectors.pkl"


@lru_cache(maxsize=1)
def _faiss():
    try:
        import faiss
    except ImportError:
        return None
    return faiss


@lru_cache(maxsize=1)
def _np():
    try:
        import numpy as np
    except ImportError:
        return None
    return np


@lru_cache(maxsize=1)
def _sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer


def _hash_repo(repo: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(repo.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "node_modules" not in p.parts:
            try:
                rel = str(p.relative_to(repo))
                st = p.stat()
                h.update(rel.encode())
                h.update(str(st.st_mtime_ns).encode())
                h.update(str(st.st_size).encode())
            except OSError:
                continue
    return h.hexdigest()[:16]


def _chunk_text(path: Path, text: str, max_lines: int = 40) -> list[tuple[str, str]]:
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    for i in range(0, len(lines), max_lines):
        block = "\n".join(lines[i : i + max_lines])
        if block.strip():
            chunks.append((str(path), f"L{i+1}-{i+len(block.splitlines())}: {block}"))
    return chunks


class CodeMemory:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.session_notes: list[str] = []
        self._model: object | None = None
        self._index: object | None = None
        self._docs: list[dict] = []
        self._dim = 384

    def remember(self, line: str) -> None:
        self.session_notes.append(line)

    def session_count(self) -> int:
        return len(self.session_notes)

    def _encoder(self) -> object:
        SentenceTransformer = _sentence_transformer()
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        if self._model is None:
            # Small model for 8GB-class machines
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._model

    def ensure_index(self) -> None:
        faiss = _faiss()
        if faiss is None or _np() is None or _sentence_transformer() is None:
            self._index = None
            self._docs = []
            return
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        current = _hash_repo(self.repo_root)
        if META_FILE.exists() and INDEX_FILE.exists():
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == current:
                self._load_index()
                return
        self._build_index(current)

    def _load_index(self) -> None:
        faiss = _faiss()
        if faiss is None:
            self._index = None
            self._docs = []
            return
        self._index = faiss.read_index(str(INDEX_FILE))
        if VECS_FILE.exists():
            with VECS_FILE.open("rb") as f:
                self._docs = pickle.load(f)

    def _save_meta(self, fp: str) -> None:
        META_FILE.write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")

    def _build_index(self, fingerprint: str) -> None:
        faiss = _faiss()
        np = _np()
        if faiss is None or np is None:
            self._index = None
            self._docs = []
            return
        enc = self._encoder()
        docs: list[dict] = []
        texts: list[str] = []
        exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".md"}
        for p in self.repo_root.rglob("*"):
            if not p.is_file() or ".git" in p.parts or "node_modules" in p.parts:
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(self.repo_root))
            for path_s, chunk in _chunk_text(rel, content):
                docs.append({"path": path_s, "chunk": chunk})
                texts.append(chunk[:2000])
        if not texts:
            self._index = faiss.IndexFlatL2(self._dim)
            self._docs = []
            faiss.write_index(self._index, str(INDEX_FILE))
            with VECS_FILE.open("wb") as f:
                pickle.dump(self._docs, f)
            self._save_meta(fingerprint)
            return
        emb = enc.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self._index = index
        self._docs = docs
        faiss.write_index(index, str(INDEX_FILE))
        with VECS_FILE.open("wb") as f:
            pickle.dump(self._docs, f)
        self._save_meta(fingerprint)

    def search(self, query: str, k: int = 8) -> list[dict]:
        faiss = _faiss()
        np = _np()
        if faiss is None or np is None or _sentence_transformer() is None:
            return []
        self.ensure_index()
        if self._index is None or not self._docs:
            return []
        enc = self._encoder()
        q = enc.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(k, len(self._docs)))
        out: list[dict] = []
        for score, i in zip(scores[0], idxs[0], strict=False):
            if 0 <= i < len(self._docs):
                row = dict(self._docs[i])
                row["score"] = float(score)
                out.append(row)
        return out
