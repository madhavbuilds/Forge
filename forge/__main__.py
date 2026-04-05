"""Typer entry: `forge` CLI."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status

from forge import __version__
from forge.agent.loop import AgentLoop, summarize_session
from forge.agent.memory import CodeMemory
from forge.config import ForgeConfig, load_config
from forge.gui.server import serve_studio
from forge.models.client import ModelClient
from forge.ui import display, logo, prompts

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console(highlight=True, soft_wrap=True)


class _PreviewServer:
    def __init__(self) -> None:
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._root: Path | None = None
        self._url: str | None = None

    def ensure(self, root: Path) -> str:
        root = root.resolve()
        if self._httpd is not None and self._root == root and self._url is not None:
            return self._url
        self.close()
        handler = partial(SimpleHTTPRequestHandler, directory=str(root))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = int(self._httpd.server_address[1])
        self._root = root
        self._url = f"http://127.0.0.1:{port}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self._url

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._httpd = None
        self._thread = None
        self._root = None
        self._url = None


def main() -> None:
    app()


def _read_user_input(prompt_text: str) -> str:
    console.print(prompt_text, end="")
    return (sys.stdin.readline() if not sys.stdin.isatty() else input()).strip()


def _show_header(cwd: Path, cfg: ForgeConfig, *, debug: bool) -> None:
    if debug:
        logo.startup_banner(console, cwd, (cfg.fast_model, cfg.smart_model))
        return
    console.print()
    console.print(f"[bold {logo.VIOLET}]Forge[/] [dim]offline AI for chat and HTML/CSS websites[/]")


def _task_with_history(task: str, history: list[tuple[str, str]], active_project: Path | None = None) -> str:
    lines: list[str] = []
    if active_project is not None:
        lines.append(f"Active website project: {active_project.name}")
    if not history:
        lines.append(f"Current user request: {task}")
        return "\n".join(lines)
    recent = history[-4:]
    lines.append("Conversation context:")
    for user_text, assistant_text in recent:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    lines.append("")
    lines.append(f"Current user request: {task}")
    return "\n".join(lines)


def _run_task_once(
    *,
    cfg: ForgeConfig,
    cwd: Path,
    user_task: str,
    memory: CodeMemory,
    debug: bool,
) -> tuple[str, AgentLoop]:
    streams = display.StreamController(console, enabled=cfg.streaming)

    def on_active_model(label: str) -> None:
        if not debug:
            return
        mid = cfg.smart_model if label == "smart" else cfg.fast_model
        display.model_switch_rail(console, label, model_id=mid)

    client = ModelClient(cfg, on_active_model=on_active_model)

    def stream_cb(s: str) -> None:
        streams.feed(s)

    def stream_end_cb() -> None:
        streams.end()

    status_holder: list[str] = [""]
    output_chunks: list[str] = []
    shown_statuses: set[str] = set()

    def status_cb(msg: str) -> None:
        status_holder[0] = msg
        if debug:
            return
        public = {
            "PLANNING": "Planning",
            "THINKING": "Thinking",
            "CODING": "Coding",
            "CHECKING": "Checking",
            "RESPONDING": "Answering",
        }
        phase = msg.split("·", 1)[0].strip().upper()
        label = public.get(phase)
        if label and label not in shown_statuses:
            shown_statuses.add(label)
            console.print(f"[dim]{label}...[/]")

    def capture_chunk(s: str) -> None:
        output_chunks.append(s)
        stream_cb(s)

    def confirm_write(path: str, content: str) -> bool:
        if not cfg.confirm_writes:
            return True
        target = cwd / path
        before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        if before:
            display.file_diff_montage(console, path, before, content)
        lexer = display.lexer_for_path(path)
        return prompts.confirm_write_gate(console, path, content, lexer=lexer)

    def confirm_cmd(cmd: str) -> bool:
        if not cfg.confirm_commands:
            return True
        return prompts.confirm_command_gate(console, cmd)

    def tool_echo(name: str, args: dict) -> None:
        if debug and cfg.show_tool_calls:
            console.print(display.tool_call_panel(name, args, compact=cfg.compact_mode))

    agent = AgentLoop(
        cfg,
        cwd,
        client,
        memory,
        stream=capture_chunk,
        stream_end=stream_end_cb,
        status=status_cb,
        confirm_write=confirm_write,
        confirm_cmd=confirm_cmd,
        tool_echo=tool_echo,
        debug=debug,
    )

    async def _go() -> None:
        if not debug:
            await agent.run(user_task)
            return

        status_console = Console(stderr=True, highlight=False, soft_wrap=True)
        with Status(
            f"[{logo.VIOLET}]…[/{logo.VIOLET}]",
            console=status_console,
            spinner="dots",
            spinner_style=logo.TEAL,
        ) as st:
            prev = ""

            def tick() -> None:
                nonlocal prev
                msg = status_holder[0] or "thinking"
                if msg != prev:
                    st.update(f"[{logo.TEAL}]{msg}[/]")
                    prev = msg

            async def runner() -> None:
                await agent.run(user_task)

            fut = asyncio.create_task(runner())
            while not fut.done():
                tick()
                await asyncio.sleep(0.08)
            await fut

    try:
        asyncio.run(_go())
    finally:
        streams.end()

    return "".join(output_chunks).strip(), agent


def _run_task_once_quiet(
    *,
    cfg: ForgeConfig,
    cwd: Path,
    user_task: str,
    memory: CodeMemory,
    debug: bool,
) -> tuple[str, AgentLoop]:
    output_chunks: list[str] = []
    client = ModelClient(cfg)
    shown_statuses: set[str] = set()

    def capture_chunk(s: str) -> None:
        output_chunks.append(s)
        print(s, end="", flush=True)

    def status_cb(msg: str) -> None:
        if debug:
            return
        public = {
            "PLANNING": "Planning",
            "THINKING": "Thinking",
            "CODING": "Coding",
            "CHECKING": "Checking",
            "RESPONDING": "Answering",
        }
        phase = msg.split("·", 1)[0].strip().upper()
        label = public.get(phase)
        if label and label not in shown_statuses:
            shown_statuses.add(label)
            print(f"\n{label}...", flush=True)

    agent = AgentLoop(
        cfg,
        cwd,
        client,
        memory,
        stream=capture_chunk,
        stream_end=lambda: print(flush=True),
        status=status_cb,
        confirm_write=lambda path, content: prompts.confirm_write_gate(
            console,
            path,
            content,
            lexer=display.lexer_for_path(path),
        ) if cfg.confirm_writes else True,
        confirm_cmd=lambda cmd: prompts.confirm_command_gate(console, cmd) if cfg.confirm_commands else True,
        tool_echo=None,
        debug=debug,
    )
    asyncio.run(agent.run(user_task))
    return "".join(output_chunks).strip(), agent


def _website_project_from_files(repo_root: Path, files_touched: list[str]) -> Path | None:
    if not files_touched:
        return None
    if not all(path.endswith((".html", ".css")) for path in files_touched):
        return None
    first = Path(files_touched[0])
    if len(first.parts) < 2:
        return None
    root_name = first.parts[0]
    if any(Path(path).parts[:1] != (root_name,) for path in files_touched):
        return None
    project_root = repo_root / root_name
    if not (project_root / "index.html").is_file():
        return None
    return project_root


@app.command()
def run(
    task_arg: str = typer.Argument("", help="Task description; omit to type interactively"),
    repo: Path | None = typer.Option(None, "--repo", "-r", help="Repository root"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config TOML path"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Log model JSON and phases to stderr"),
) -> None:
    """Run the Forge agent on a task."""
    cfg = load_config(config)
    cfg.apply_env()
    cwd = (repo or Path.cwd()).resolve()
    os.chdir(cwd)
    _show_header(cwd, cfg, debug=debug)

    user_task = task_arg.strip()
    if not user_task:
        user_task = _read_user_input(prompts.ps1_line(cwd))
    if not user_task:
        raise typer.Exit(code=1)

    memory = CodeMemory(cwd)
    try:
        _, agent = _run_task_once(
            cfg=cfg,
            cwd=cwd,
            user_task=user_task,
            memory=memory,
            debug=debug,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted[/]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        display.error_panel(
            console,
            e,
            hint="Check Ollama is running (`ollama serve`) and models are pulled.",
        )
        raise typer.Exit(code=1) from e

    if debug or agent.files_touched or agent.commands_run or agent.tests_ok is not None:
        display.completion_summary(
            console,
            files=agent.files_touched,
            commands=agent.commands_run,
            tests_ok=agent.tests_ok,
            model_active="session",
            session_memory=summarize_session(memory),
        )


@app.command()
def version() -> None:
    """Print version."""
    typer.echo(__version__)


@app.command()
def chat(
    repo: Path | None = typer.Option(None, "--repo", "-r", help="Repository root"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config TOML path"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Log model JSON and phases to stderr"),
) -> None:
    """Keep Forge running as a local chat session."""
    cfg = load_config(config)
    cfg.apply_env()
    cwd = (repo or Path.cwd()).resolve()
    os.chdir(cwd)

    console.print()
    console.print("[bold]Forge[/] [dim]chat + HTML/CSS website builder[/]")
    console.print("[dim]Ask anything, or describe a website to make. Type /exit to quit.[/]")

    memory = CodeMemory(cwd)
    history: list[tuple[str, str]] = []
    active_project: Path | None = None
    preview = _PreviewServer()

    try:
        while True:
            user_task = _read_user_input(prompts.chat_ps1_line(cwd))
            if not user_task:
                continue
            lowered = user_task.strip().lower()
            if lowered in {"/exit", "exit", "quit"}:
                break
            if lowered == "/clear":
                history.clear()
                memory = CodeMemory(cwd)
                active_project = None
                console.print("[dim]Chat memory cleared.[/]")
                continue

            task = _task_with_history(user_task, history, active_project)
            try:
                response, agent = _run_task_once_quiet(
                    cfg=cfg,
                    cwd=cwd,
                    user_task=task,
                    memory=memory,
                    debug=debug,
                )
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted[/]")
                raise typer.Exit(code=130) from None
            except Exception as e:
                display.error_panel(
                    console,
                    e,
                    hint="Check Ollama is running (`ollama serve`) and models are pulled.",
                )
                continue

            history.append((user_task, response))
            history = history[-8:]
            if agent.files_touched:
                console.print(f"[dim]Updated: {', '.join(agent.files_touched)}[/]")
                project_root = _website_project_from_files(cwd, agent.files_touched)
                if project_root is not None:
                    active_project = project_root
                    url = preview.ensure(project_root)
                    console.print(f"[dim]Preview: {url}[/]")
                    webbrowser.open(url)
            elif agent.commands_run:
                console.print(f"[dim]Ran: {', '.join(agent.commands_run)}[/]")
    finally:
        preview.close()


@app.command()
def ui(
    repo: Path | None = typer.Option(None, "--repo", "-r", help="Repository root"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config TOML path"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8765, "--port", help="Bind port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
) -> None:
    """Launch the local browser studio."""
    cfg = load_config(config)
    cfg.apply_env()
    cwd = (repo or Path.cwd()).resolve()
    typer.echo(f"Forge Studio running at http://{host}:{port}")
    try:
        serve_studio(
            repo=cwd,
            config_path=config,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None


if __name__ == "__main__":
    main()
