"""Git checkpoint and log."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_checkpoint(repo: Path, message: str | None = None) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    msg = message or f"forge-checkpoint-{ts}"
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "stash", "push", "-m", msg],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "ok": r.returncode == 0,
            "message": msg,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e)}


def get_git_log(n: int, repo: Path) -> dict:
    try:
        r = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "log",
                f"-{n}",
                "--pretty=format:%H %s",
                "-p",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {"ok": r.returncode == 0, "log": r.stdout[:50000]}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e)}
