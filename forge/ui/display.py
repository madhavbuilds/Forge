"""Rich terminal UI: streaming, tool panels, diffs, gates, summary."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.json import JSON
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from forge import __version__
from forge.ui.logo import TAGLINE, TEAL, VIOLET

SAFE = "#5DCAA5"
WARN = "#E5C07B"
DANGER = "#E06C75"
DIM = "dim"


def lexer_for_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".toml": "toml",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".sh": "bash",
        ".html": "html",
        ".css": "css",
    }.get(ext, "text")


def _args_summary(args: dict[str, Any], max_len: int = 72) -> str:
    parts: list[str] = []
    for k, v in list(args.items())[:5]:
        s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else repr(v)
        if len(s) > 48:
            s = s[:45] + "…"
        parts.append(f"{k}={s}")
    out = " ".join(parts)
    return out if len(out) <= max_len else out[: max_len - 1] + "…"


class StreamController:
    """Token stream with a clear header/footer (typewriter on stdout)."""

    def __init__(self, console: Console, *, enabled: bool) -> None:
        self.console = console
        self.enabled = enabled
        self._open = False

    def feed(self, chunk: str) -> None:
        if not self.enabled or not chunk:
            return
        if not self._open:
            self.console.print()
            self.console.print(
                Rule(f"[bold {VIOLET}]assistant[/] [dim]· streaming[/]", style=f"dim {VIOLET}")
            )
            self._open = True
        self.console.print(chunk, end="", highlight=False, soft_wrap=True)

    def end(self) -> None:
        if self._open:
            self.console.print()
            self.console.print(Rule(style="dim"))
            self._open = False


def tool_call_panel(
    name: str,
    args: dict[str, Any],
    *,
    compact: bool,
) -> Panel:
    """Tool invocation: compact one-liner or full JSON body (mockup-style tool rail)."""
    destructive = name in ("write_file", "run_command")
    icon = "▾"
    chip = "⚡" if destructive else "◇"
    title = f"[bold {VIOLET}]{icon}[/] {chip} [bold]{name}[/]"

    if compact:
        body: str | Text = Text.from_markup(
            f"[{DIM}]{_args_summary(args)}[/]" if args else f"[{DIM}](no args)[/]"
        )
        border = WARN if name == "write_file" else (DANGER if name == "run_command" else TEAL)
        return Panel(
            body,
            title=title,
            title_align="left",
            border_style=border,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    try:
        body_json: RenderableType = JSON.from_data(args, indent=2, default=str)
    except TypeError:
        body_json = Syntax(json.dumps(args, default=str, indent=2), "json", theme="monokai")

    border = WARN if name == "write_file" else (DANGER if name == "run_command" else TEAL)
    return Panel(
        body_json,
        title=title,
        subtitle="[dim]full args · forge ❯[/]",
        title_align="left",
        border_style=border,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def file_diff_montage(console: Console, path: str, before: str, after: str) -> None:
    """Side-by-side syntax panels + unified diff (Rich + difflib)."""
    lexer = lexer_for_path(path)
    theme = "monokai"
    left = Panel(
        Syntax(before or " ", lexer, theme=theme, line_numbers=True, word_wrap=True),
        title=f"[red]− before[/] [dim]{path}[/]",
        border_style="red",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    right = Panel(
        Syntax(after or " ", lexer, theme=theme, line_numbers=True, word_wrap=True),
        title=f"[green]+ after[/] [dim]{path}[/]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(
        Panel(
            Columns([left, right], equal=True, expand=True),
            title=f"[bold {VIOLET}]diff preview[/]",
            border_style=VIOLET,
            box=box.DOUBLE,
        )
    )

    u_before = before.splitlines(keepends=True)
    u_after = after.splitlines(keepends=True)
    unified = "".join(difflib.unified_diff(u_before, u_after, fromfile="before", tofile="after", lineterm=""))
    if unified.strip():
        console.print(
            Panel(
                Syntax(unified, "diff", theme=theme, line_numbers=True, word_wrap=True),
                title="[dim]unified[/]",
                border_style=TEAL,
                box=box.ROUNDED,
            )
        )


def error_panel(console: Console, exc: BaseException, hint: str | None = None) -> None:
    import traceback

    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_text = "".join(tb_lines)
    body = Group(
        Syntax(tb_text, "pytb", theme="monokai", word_wrap=True),
        Text(""),
        Text.from_markup(f"[bold {DANGER}]suggested[/] [dim]{hint or 'See traceback above.'}[/]"),
    )
    console.print(
        Panel(
            body,
            title="[bold red]forge error[/]",
            subtitle="[dim]full traceback[/]",
            border_style="red",
            box=box.HEAVY,
        )
    )


def model_switch_rail(console: Console, label: str, *, model_id: str | None = None) -> None:
    """Live model indicator: fast vs smart."""
    is_smart = label == "smart"
    color = TEAL if is_smart else VIOLET
    tag = "SMART" if is_smart else "FAST"
    extra = f" [dim]{model_id}[/]" if model_id else ""
    console.print(
        Rule(
            f"[bold {color}]model · {tag}[/]{extra}",
            style=f"dim {color}",
            characters="─",
        )
    )


def completion_summary(
    console: Console,
    *,
    files: list[str],
    commands: list[str],
    tests_ok: bool | None,
    model_active: str,
    session_memory: str,
) -> None:
    t = Table.grid(padding=(0, 2), expand=True)
    t.add_column(style=f"bold {TEAL}", justify="right")
    t.add_column(ratio=1)
    t.add_row("model", f"[bold {VIOLET if model_active == 'fast' else TEAL}]{model_active}[/]")
    t.add_row("files", "\n".join(f"· [bold]{f}[/]" for f in files) or "[dim]—[/]")
    t.add_row("commands", "\n".join(f"· [dim]{c}[/]" for c in commands) or "[dim]—[/]")
    if tests_ok is None:
        t.add_row("tests", "[dim]not run / skipped[/]")
    else:
        t.add_row("tests", "[green]passed[/]" if tests_ok else "[red]failed[/]")
    inner = Panel(
        t,
        title=f"[bold {VIOLET}]session complete[/]",
        subtitle=f"[dim]{session_memory}[/]",
        border_style=VIOLET,
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print()
    console.print(
        Panel(
            Group(
                inner,
                Text.from_markup(f"[dim]{TAGLINE} · v{__version__}[/]", justify="center"),
            ),
            box=box.SIMPLE,
        )
    )


def model_badge(active: str) -> str:
    color = VIOLET if active == "fast" else TEAL
    return f"[bold {color}]active model: {active}[/]"
