"""Simple intent router for Forge."""

from __future__ import annotations

READ_KEYWORDS = [
    "what",
    "list",
    "show",
    "find",
    "where",
    "how many",
    "explain",
    "describe",
    "tell me",
    "which",
    "who",
    "does",
    "is there",
    "what does",
]

WRITE_KEYWORDS = [
    "fix",
    "create",
    "write",
    "add",
    "build",
    "design",
    "generate",
    "implement",
    "change",
    "update",
    "refactor",
    "delete",
    "rename",
    "move",
    "edit",
    "make",
]


def classify(task: str) -> str:
    task_lower = task.lower()
    if any(keyword in task_lower for keyword in WRITE_KEYWORDS):
        return "edit_code"
    return "search"
