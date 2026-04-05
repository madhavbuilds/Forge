"""Confirmation gates: green safe, yellow caution, red destructive."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.text import Text

from forge.ui.logo import TEAL, VIOLET

SAFE = "#5DCAA5"
WARN = "#E5C07B"
DANGER = "#E06C75"


def ps1_line(cwd: Path) -> str:
    return f"[bold {VIOLET}]you[/] [dim]❯[/] "


def chat_ps1_line(cwd: Path) -> str:
    return f"[bold {VIOLET}]you[/] [dim]❯[/] "


def confirm_write_gate(console: Console, path: str, content: str, *, lexer: str = "python") -> bool:
    """Yellow / caution — file mutation."""
    preview = content[:1200] + ("…" if len(content) > 1200 else "")
    body = Group(
        Text.from_markup(f"[bold {WARN}]Write file[/] [bold white]{path}[/]"),
        Text(""),
        Syntax(preview, lexer, theme="monokai", line_numbers=False, word_wrap=True),
    )
    console.print()
    console.print(
        Panel(
            body,
            title="[bold]confirmation[/]",
            subtitle=f"[{WARN}]caution · disk write[/]",
            border_style=WARN,
            box=box.HEAVY,
            padding=(1, 2),
        )
    )
    return Confirm.ask(
        f"[{SAFE}]Apply this write?[/]",
        default=True,
        show_default=True,
    )


def confirm_command_gate(console: Console, cmd: str) -> bool:
    """Red — arbitrary shell (non read-only)."""
    body = Group(
        Text.from_markup(f"[bold {DANGER}]Shell command[/] [dim](requires approval)[/]"),
        Text(""),
        Syntax(cmd, "bash", theme="monokai", word_wrap=True),
    )
    console.print()
    console.print(
        Panel(
            body,
            title="[bold]confirmation[/]",
            subtitle=f"[{DANGER}]destructive / non–read-only[/]",
            border_style=DANGER,
            box=box.HEAVY,
            padding=(1, 2),
        )
    )
    return Confirm.ask(
        f"[{WARN}]Execute on this machine?[/]",
        default=False,
        show_default=True,
    )


def confirm_safe_gate(console: Console, message: str) -> bool:
    """Green — low-risk action."""
    console.print()
    console.print(
        Panel(
            Text(message),
            title="[bold]confirm[/]",
            subtitle=f"[{SAFE}]safe[/]",
            border_style=SAFE,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    return Confirm.ask(f"[{SAFE}]Continue?[/]", default=True)
