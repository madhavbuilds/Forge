"""run_command with read-only vs destructive classification."""

from __future__ import annotations

import asyncio


READONLY_PREFIXES = (
    "ls ",
    "dir ",
    "pwd",
    "git status",
    "git diff",
    "git log",
    "git show",
    "cat ",
    "head ",
    "tail ",
    "wc ",
    "find ",
    "rg ",
    "grep ",
    "which ",
    "pytest --collect-only",
    "python -m pytest --collect-only",
)


def is_readonly_command(cmd: str) -> bool:
    c = cmd.strip().lower()
    return any(c.startswith(p.strip().lower()) for p in READONLY_PREFIXES)


async def run_command(cmd: str, timeout: float) -> dict:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        text = out.decode("utf-8", errors="replace")
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "output": text[-200000:]}
    except asyncio.TimeoutError:
        proc.kill()
        return {"ok": False, "error": "timeout", "output": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "output": ""}


def preview_command(cmd: str) -> str:
    return f"$ {cmd}"
