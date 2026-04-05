"""ASCII diamond logo and startup banner."""

from __future__ import annotations

import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from forge import __version__

VIOLET = "#7F77DD"
TEAL = "#5DCAA5"
TAGLINE = "Code at the speed of thought. No cloud. No cost. No limits."

LOGO_FRAMES = [
    r"""
     /\
    /  \
   /    \
  /      \
 /        \
 \        /
  \      /
   \    /
    \  /
     \/
""".strip(),
    r"""
     /\
    /◆ \
   /    \
  /      \
 /   ◆    \
 \        /
  \  ◆   /
   \    /
    \  /
     \/
""".strip(),
]


def diamond_logo() -> Text:
    t = Text(LOGO_FRAMES[0], style=f"bold {VIOLET}")
    return t


def animate_logo(console: Console, frames: int = 3) -> None:
    for i in range(frames):
        txt = Text(LOGO_FRAMES[i % len(LOGO_FRAMES)], style=f"bold {VIOLET}")
        console.print(
            Panel(
                txt,
                border_style=VIOLET,
                title="[bold]forge[/]",
                subtitle=f"[dim]{TAGLINE}[/]",
                box=box.DOUBLE,
            )
        )
        time.sleep(0.1)


def startup_banner(console: Console, repo: Path | None, models: tuple[str, str]) -> None:
    console.print()
    animate_logo(console, frames=1)
    fast, smart = models
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {TEAL}", justify="right")
    grid.add_column(ratio=1)
    grid.add_row("version", f"[bold]{__version__}[/]")
    grid.add_row("fast", fast)
    grid.add_row("smart", smart)
    if repo:
        grid.add_row("repo", f"[dim]{repo}[/]")
    console.print(
        Panel(
            grid,
            title=f"[bold {VIOLET}]Forge[/] [dim]· local agent[/]",
            border_style=VIOLET,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print(Rule(style=f"dim {VIOLET}"))
