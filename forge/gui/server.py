"""Local browser-based studio for Forge."""

from __future__ import annotations

import asyncio
import difflib
import json
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from forge.agent.loop import AgentLoop, summarize_session
from forge.agent.memory import CodeMemory
from forge.agent.router import classify
from forge.config import load_config
from forge.models.client import ModelClient

MAX_EVENTS = 160
MAX_OUTPUT_CHARS = 80_000
MAX_DIFF_CHARS = 24_000


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\n"


def _build_diff(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        lineterm="",
    )
    return _trim_text("".join(diff), MAX_DIFF_CHARS)


@dataclass(slots=True)
class PendingWrite:
    path: str
    diff: str
    before_preview: str
    after_preview: str


@dataclass(slots=True)
class StudioSession:
    id: str
    task: str
    repo: str
    auto_apply: bool
    intent: str
    status: str = "Queued"
    active_model: str = "fast"
    output: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    tests_ok: bool | None = None
    session_memory: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    finished: bool = False
    error: str | None = None
    waiting_for_write: bool = False
    pending_write: PendingWrite | None = None
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _write_decision: bool | None = field(default=None, repr=False)

    def add_event(self, kind: str, title: str, detail: str = "") -> None:
        with self._condition:
            self.events.append(
                {
                    "kind": kind,
                    "title": title,
                    "detail": detail,
                    "time": time.strftime("%H:%M:%S"),
                }
            )
            self.events = self.events[-MAX_EVENTS:]
            self.updated_at = time.time()

    def set_status(self, status: str) -> None:
        with self._condition:
            self.status = status
            self.updated_at = time.time()
        self.add_event("status", "Status", status)

    def set_model(self, label: str, model_id: str) -> None:
        with self._condition:
            self.active_model = label
            self.updated_at = time.time()
        self.add_event("model", f"{label.upper()} model", model_id)

    def append_output(self, chunk: str) -> None:
        with self._condition:
            self.output = _trim_text(self.output + chunk, MAX_OUTPUT_CHARS)
            self.updated_at = time.time()

    def begin_write_confirmation(self, pending: PendingWrite) -> None:
        with self._condition:
            self.waiting_for_write = True
            self.pending_write = pending
            self._write_decision = None
            self.updated_at = time.time()
        self.add_event("confirm", "Awaiting write confirmation", pending.path)

    def resolve_write_decision(self, approve: bool) -> None:
        with self._condition:
            self._write_decision = approve
            self.waiting_for_write = False
            self.pending_write = None
            self.updated_at = time.time()
            self._condition.notify_all()
        action = "Approved write" if approve else "Skipped write"
        self.add_event("confirm", action)

    def wait_for_write_decision(self) -> bool:
        with self._condition:
            while self._write_decision is None:
                self._condition.wait(timeout=0.25)
            decision = bool(self._write_decision)
            self._write_decision = None
            return decision

    def mark_finished(
        self,
        *,
        files_touched: list[str],
        commands_run: list[str],
        tests_ok: bool | None,
        session_memory: str,
    ) -> None:
        with self._condition:
            self.files_touched = files_touched
            self.commands_run = commands_run
            self.tests_ok = tests_ok
            self.session_memory = session_memory
            self.status = "Done"
            self.finished = True
            self.finished_at = time.time()
            self.updated_at = time.time()
        self.add_event("done", "Run complete")

    def mark_failed(self, error: str) -> None:
        with self._condition:
            self.status = "Error"
            self.error = error
            self.finished = True
            self.finished_at = time.time()
            self.updated_at = time.time()
        self.add_event("error", "Run failed", error)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            pending_write = None
            if self.pending_write is not None:
                pending_write = {
                    "path": self.pending_write.path,
                    "diff": self.pending_write.diff,
                    "before_preview": self.pending_write.before_preview,
                    "after_preview": self.pending_write.after_preview,
                }
            return {
                "id": self.id,
                "task": self.task,
                "repo": self.repo,
                "intent": self.intent,
                "status": self.status,
                "active_model": self.active_model,
                "output": self.output,
                "events": list(self.events),
                "files_touched": list(self.files_touched),
                "commands_run": list(self.commands_run),
                "tests_ok": self.tests_ok,
                "session_memory": self.session_memory,
                "finished": self.finished,
                "error": self.error,
                "waiting_for_write": self.waiting_for_write,
                "pending_write": pending_write,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class ForgeStudio:
    def __init__(self, default_repo: Path, config_path: Path | None = None) -> None:
        self.default_repo = default_repo.resolve()
        self.config_path = config_path
        self._sessions: dict[str, StudioSession] = {}
        self._lock = threading.Lock()

    def create_session(self, task: str, repo: str | None, auto_apply: bool) -> StudioSession:
        task = task.strip()
        if not task:
            raise ValueError("Task is required.")
        repo_path = Path(repo or self.default_repo).expanduser().resolve()
        if not repo_path.exists():
            raise ValueError(f"Repo does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise ValueError(f"Repo must be a directory: {repo_path}")

        with self._lock:
            if any(not session.finished for session in self._sessions.values()):
                raise RuntimeError("Forge is already processing a task. Finish the current run to keep the UI responsive.")
            session = StudioSession(
                id=uuid.uuid4().hex[:10],
                task=task,
                repo=str(repo_path),
                auto_apply=auto_apply,
                intent=classify(task),
            )
            session.add_event("classify", "Intent", session.intent)
            self._sessions[session.id] = session
            worker = threading.Thread(target=self._run_session, args=(session.id,), daemon=True)
            worker.start()
            return session

    def session(self, session_id: str) -> StudioSession | None:
        return self._sessions.get(session_id)

    def snapshot(self, session_id: str) -> dict[str, Any] | None:
        session = self.session(session_id)
        if session is None:
            return None
        return session.snapshot()

    def resolve_write(self, session_id: str, approve: bool) -> dict[str, Any] | None:
        session = self.session(session_id)
        if session is None:
            return None
        session.resolve_write_decision(approve)
        return session.snapshot()

    def info(self) -> dict[str, Any]:
        active = any(not session.finished for session in self._sessions.values())
        latest = None
        if self._sessions:
            latest = max(self._sessions.values(), key=lambda item: item.started_at).id
        return {
            "default_repo": str(self.default_repo),
            "active": active,
            "latest_session_id": latest,
        }

    def _run_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        repo_root = Path(session.repo)
        cfg = load_config(self.config_path)
        cfg.apply_env()

        memory = CodeMemory(repo_root)

        def on_active_model(label: str) -> None:
            model_id = cfg.smart_model if label == "smart" else cfg.fast_model
            session.set_model(label, model_id)

        client = ModelClient(cfg, on_active_model=on_active_model)

        def stream_cb(chunk: str) -> None:
            session.append_output(chunk)

        def status_cb(message: str) -> None:
            session.set_status(message)

        def tool_echo(name: str, args: dict[str, Any]) -> None:
            session.add_event("tool", name, json.dumps(args, ensure_ascii=False))

        def confirm_write(path: str, content: str) -> bool:
            target = (repo_root / path).resolve()
            before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
            pending = PendingWrite(
                path=path,
                diff=_build_diff(path, before, content),
                before_preview=_trim_text(before, 8_000),
                after_preview=_trim_text(content, 8_000),
            )
            session.begin_write_confirmation(pending)
            if session.auto_apply:
                session.resolve_write_decision(True)
            return session.wait_for_write_decision()

        agent = AgentLoop(
            cfg,
            repo_root,
            client,
            memory,
            stream=stream_cb,
            stream_end=lambda: None,
            status=status_cb,
            confirm_write=confirm_write,
            tool_echo=tool_echo,
            debug=False,
        )

        try:
            asyncio.run(agent.run(session.task))
        except Exception:
            session.mark_failed(traceback.format_exc())
            return

        session.mark_finished(
            files_touched=list(agent.files_touched),
            commands_run=list(agent.commands_run),
            tests_ok=agent.tests_ok,
            session_memory=summarize_session(memory),
        )


def _page(default_repo: str) -> str:
    default_repo_json = json.dumps(default_repo)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forge Studio</title>
  <style>
    :root {{
      --bg: #f6f1e9;
      --bg-accent: #fdf8f1;
      --card: rgba(255, 252, 247, 0.9);
      --line: rgba(30, 41, 59, 0.12);
      --text: #1d2939;
      --muted: #667085;
      --teal: #0f766e;
      --teal-soft: #d8f3ef;
      --coral: #d9485f;
      --sand: #f3dfc5;
      --gold: #c7882a;
      --shadow: 0 24px 60px rgba(84, 63, 39, 0.12);
      --radius: 24px;
      --mono: "SFMono-Regular", "SF Mono", "JetBrains Mono", "Menlo", monospace;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(217, 72, 95, 0.12), transparent 28%),
        linear-gradient(180deg, #fff9f1 0%, var(--bg) 55%, #efe4d5 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1380px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.2fr 0.8fr;
      align-items: stretch;
      margin-bottom: 22px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .brand {{
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}
    .brand::after {{
      content: "";
      position: absolute;
      inset: auto -20px -30px auto;
      width: 180px;
      height: 180px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(15, 118, 110, 0.16), transparent 70%);
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--teal-soft);
      color: var(--teal);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 18px 0 10px;
      font-size: clamp(2.2rem, 4vw, 3.6rem);
      line-height: 0.94;
      letter-spacing: -0.04em;
      max-width: 10ch;
    }}
    .lead {{
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.7;
      max-width: 64ch;
      margin: 0;
    }}
    .hero-meta {{
      padding: 28px;
      display: grid;
      gap: 18px;
      align-content: space-between;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .metric {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .metric strong {{
      font-size: 1rem;
      line-height: 1.4;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
      gap: 22px;
    }}
    .stack {{
      display: grid;
      gap: 22px;
    }}
    .composer {{
      padding: 24px;
    }}
    .composer-grid {{
      display: grid;
      gap: 14px;
    }}
    .field-label {{
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
      font-size: 0.9rem;
    }}
    input[type="text"], textarea {{
      width: 100%;
      border: 1px solid rgba(15, 23, 42, 0.1);
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      color: inherit;
      background: rgba(255,255,255,0.82);
      transition: border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
    }}
    input[type="text"]:focus, textarea:focus {{
      outline: none;
      border-color: rgba(15, 118, 110, 0.55);
      box-shadow: 0 0 0 5px rgba(15, 118, 110, 0.12);
      transform: translateY(-1px);
    }}
    textarea {{
      min-height: 140px;
      resize: vertical;
      line-height: 1.55;
    }}
    .composer-actions {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-top: 6px;
    }}
    .toggle {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.94rem;
    }}
    .toggle input {{
      width: 18px;
      height: 18px;
      accent-color: var(--teal);
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease, box-shadow 140ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    button:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
    }}
    .primary {{
      background: linear-gradient(135deg, #0f766e, #1c9a86);
      color: white;
      box-shadow: 0 12px 24px rgba(15, 118, 110, 0.24);
    }}
    .ghost {{
      background: rgba(255,255,255,0.76);
      color: var(--text);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }}
    .danger {{
      background: linear-gradient(135deg, #d9485f, #ea6277);
      color: white;
    }}
    .section {{
      padding: 24px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 1.05rem;
      letter-spacing: -0.02em;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--teal);
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .badge.warn {{
      background: rgba(199, 136, 42, 0.14);
      color: #8b5e17;
    }}
    .badge.error {{
      background: rgba(217, 72, 95, 0.12);
      color: var(--coral);
    }}
    .output {{
      min-height: 220px;
      max-height: 420px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: var(--sans);
      line-height: 1.7;
      color: #243145;
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }}
    .timeline {{
      display: grid;
      gap: 10px;
      max-height: 340px;
      overflow: auto;
    }}
    .event {{
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.76);
    }}
    .event-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
      font-size: 0.92rem;
      font-weight: 700;
    }}
    .event small {{
      color: var(--muted);
      font-size: 0.78rem;
    }}
    .event-detail {{
      color: var(--muted);
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.84rem;
      line-height: 1.55;
    }}
    .stats {{
      display: grid;
      gap: 12px;
    }}
    .stat-list {{
      display: grid;
      gap: 10px;
    }}
    .stat {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.76);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }}
    .stat strong {{
      display: block;
      font-size: 0.84rem;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .collection {{
      margin: 0;
      padding-left: 18px;
      color: var(--text);
    }}
    .collection li + li {{
      margin-top: 6px;
    }}
    .diff {{
      max-height: 340px;
      overflow: auto;
      margin: 0;
      border-radius: 18px;
      padding: 16px;
      font-family: var(--mono);
      font-size: 0.84rem;
      line-height: 1.55;
      background: #fff;
      border: 1px solid rgba(15, 23, 42, 0.08);
      white-space: pre-wrap;
    }}
    .hint {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }}
    .footer-note {{
      padding: 18px 24px 26px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 1120px) {{
      .hero, .layout {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .shell {{
        width: min(100vw - 18px, 100%);
        margin: 10px auto 20px;
      }}
      .brand, .hero-meta, .composer, .section {{
        padding: 18px;
      }}
      .composer-actions {{
        flex-direction: column;
        align-items: stretch;
      }}
      button {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel brand">
        <div class="eyebrow">Offline vibe coding studio</div>
        <h1>Forge builds locally and stays light.</h1>
        <p class="lead">Describe what you want, review the diff, approve the write, and keep moving. Everything runs on your machine with local models and a lightweight browser UI.</p>
      </div>
      <div class="panel hero-meta">
        <div class="meta-grid">
          <div class="metric">
            <span>Default repo</span>
            <strong id="defaultRepo"></strong>
          </div>
          <div class="metric">
            <span>Concurrency</span>
            <strong>1 active run to protect laptop responsiveness</strong>
          </div>
          <div class="metric">
            <span>Model split</span>
            <strong>FAST responds, SMART only edits code</strong>
          </div>
          <div class="metric">
            <span>Write safety</span>
            <strong>Diff preview with explicit approval</strong>
          </div>
        </div>
        <p class="hint">Tip: keep Ollama running in the background with your fast and smart models loaded for the smoothest experience.</p>
      </div>
    </section>

    <div class="layout">
      <div class="stack">
        <section class="panel composer">
          <div class="composer-grid">
            <div>
              <label class="field-label" for="repoInput">Repository</label>
              <input id="repoInput" type="text" />
            </div>
            <div>
              <label class="field-label" for="taskInput">What should Forge do?</label>
              <textarea id="taskInput" placeholder="Example: make the landing page feel warmer, reduce clutter, and keep everything offline."></textarea>
            </div>
            <div class="composer-actions">
              <label class="toggle">
                <input id="autoApply" type="checkbox" />
                Auto-approve writes for this run
              </label>
              <button id="runButton" class="primary">Start run</button>
            </div>
          </div>
        </section>

        <section class="panel section">
          <div class="section-head">
            <h2>Assistant response</h2>
            <div id="statusBadge" class="badge">Idle</div>
          </div>
          <div id="output" class="output">Forge is ready.</div>
        </section>

        <section class="panel section">
          <div class="section-head">
            <h2>Activity</h2>
            <div id="modelBadge" class="badge warn">FAST</div>
          </div>
          <div id="timeline" class="timeline"></div>
        </section>
      </div>

      <div class="stack">
        <section class="panel section stats">
          <div class="section-head">
            <h2>Run details</h2>
            <div id="intentBadge" class="badge warn">No run yet</div>
          </div>
          <div class="stat-list">
            <div class="stat">
              <strong>Task</strong>
              <div id="taskValue" class="hint">Waiting for a run.</div>
            </div>
            <div class="stat">
              <strong>Files touched</strong>
              <ul id="filesList" class="collection"></ul>
            </div>
            <div class="stat">
              <strong>Commands run</strong>
              <ul id="commandsList" class="collection"></ul>
            </div>
            <div class="stat">
              <strong>Tests</strong>
              <div id="testsValue" class="hint">Not run.</div>
            </div>
            <div class="stat">
              <strong>Session memory</strong>
              <div id="memoryValue" class="hint">No session yet.</div>
            </div>
          </div>
        </section>

        <section class="panel section" id="confirmCard" style="display:none">
          <div class="section-head">
            <h2>Approve write</h2>
            <div class="badge error">Attention needed</div>
          </div>
          <p id="confirmPath" class="hint"></p>
          <pre id="diffView" class="diff"></pre>
          <div class="composer-actions" style="margin-top:14px">
            <button id="approveButton" class="primary">Apply change</button>
            <button id="rejectButton" class="danger">Skip write</button>
          </div>
        </section>

        <section class="panel section">
          <h2>Performance notes</h2>
          <p class="hint">Forge keeps the UI smooth by running one task at a time, doing background work in a worker thread, using the small model for reads and responses, and only waking the larger model when it actually needs to generate code.</p>
        </section>
      </div>
    </div>

    <div class="footer-note">Forge Studio is local-first. Keep this tab open while a run is active so you can approve writes when needed.</div>
  </div>

  <script>
    const defaultRepo = {default_repo_json};
    const repoInput = document.getElementById("repoInput");
    const taskInput = document.getElementById("taskInput");
    const autoApply = document.getElementById("autoApply");
    const runButton = document.getElementById("runButton");
    const statusBadge = document.getElementById("statusBadge");
    const modelBadge = document.getElementById("modelBadge");
    const intentBadge = document.getElementById("intentBadge");
    const output = document.getElementById("output");
    const timeline = document.getElementById("timeline");
    const taskValue = document.getElementById("taskValue");
    const filesList = document.getElementById("filesList");
    const commandsList = document.getElementById("commandsList");
    const testsValue = document.getElementById("testsValue");
    const memoryValue = document.getElementById("memoryValue");
    const confirmCard = document.getElementById("confirmCard");
    const confirmPath = document.getElementById("confirmPath");
    const diffView = document.getElementById("diffView");
    const approveButton = document.getElementById("approveButton");
    const rejectButton = document.getElementById("rejectButton");
    const defaultRepoLabel = document.getElementById("defaultRepo");

    let currentSessionId = null;
    let pollHandle = null;

    repoInput.value = defaultRepo;
    defaultRepoLabel.textContent = defaultRepo;

    async function request(path, options = {{}}) {{
      const response = await fetch(path, {{
        headers: {{ "Content-Type": "application/json" }},
        ...options,
      }});
      const data = await response.json();
      if (!response.ok) {{
        throw new Error(data.error || "Request failed");
      }}
      return data;
    }}

    function renderList(target, items, emptyLabel) {{
      target.innerHTML = "";
      if (!items || !items.length) {{
        const li = document.createElement("li");
        li.textContent = emptyLabel;
        target.appendChild(li);
        return;
      }}
      items.forEach((item) => {{
        const li = document.createElement("li");
        li.textContent = item;
        target.appendChild(li);
      }});
    }}

    function renderEvents(events) {{
      timeline.innerHTML = "";
      if (!events || !events.length) {{
        const empty = document.createElement("div");
        empty.className = "event";
        empty.innerHTML = '<div class="event-head"><span>No activity yet</span></div><div class="event-detail">Start a run to see classification, tool calls, model switches, and completion events.</div>';
        timeline.appendChild(empty);
        return;
      }}
      events.slice().reverse().forEach((event) => {{
        const node = document.createElement("div");
        node.className = "event";
        node.innerHTML = `
          <div class="event-head">
            <span>${{event.title}}</span>
            <small>${{event.time}}</small>
          </div>
          <div class="event-detail">${{event.detail || event.kind}}</div>
        `;
        timeline.appendChild(node);
      }});
    }}

    function renderSnapshot(snapshot) {{
      currentSessionId = snapshot.id;
      statusBadge.textContent = snapshot.status || "Running";
      modelBadge.textContent = (snapshot.active_model || "fast").toUpperCase();
      intentBadge.textContent = snapshot.intent || "search";
      output.textContent = snapshot.output || "Forge is thinking…";
      output.scrollTop = output.scrollHeight;
      taskValue.textContent = snapshot.task || "Waiting for a run.";
      testsValue.textContent =
        snapshot.tests_ok === null ? "Not run." : (snapshot.tests_ok ? "Passing." : "Failed.");
      memoryValue.textContent = snapshot.session_memory || "Working…";
      renderList(filesList, snapshot.files_touched, "No files changed.");
      renderList(commandsList, snapshot.commands_run, "No commands run.");
      renderEvents(snapshot.events || []);

      if (snapshot.waiting_for_write && snapshot.pending_write) {{
        confirmCard.style.display = "block";
        confirmPath.textContent = `Review change for ${{snapshot.pending_write.path}}`;
        diffView.textContent = snapshot.pending_write.diff || "No diff available.";
      }} else {{
        confirmCard.style.display = "none";
      }}

      if (snapshot.error) {{
        statusBadge.className = "badge error";
      }} else if (snapshot.waiting_for_write) {{
        statusBadge.className = "badge warn";
      }} else {{
        statusBadge.className = "badge";
      }}

      runButton.disabled = !snapshot.finished;
      taskInput.disabled = !snapshot.finished;
      repoInput.disabled = !snapshot.finished;
      autoApply.disabled = !snapshot.finished;

      if (snapshot.finished && pollHandle) {{
        clearInterval(pollHandle);
        pollHandle = null;
      }}
    }}

    async function pollSnapshot() {{
      if (!currentSessionId) return;
      try {{
        const snapshot = await request(`/api/session/${{currentSessionId}}`);
        renderSnapshot(snapshot);
      }} catch (error) {{
        statusBadge.textContent = error.message;
        statusBadge.className = "badge error";
      }}
    }}

    async function startRun() {{
      const task = taskInput.value.trim();
      if (!task) {{
        statusBadge.textContent = "Enter a task first";
        statusBadge.className = "badge error";
        return;
      }}

      runButton.disabled = true;
      statusBadge.textContent = "Starting…";
      statusBadge.className = "badge warn";

      try {{
        const snapshot = await request("/api/session", {{
          method: "POST",
          body: JSON.stringify({{
            task,
            repo: repoInput.value.trim(),
            auto_apply: autoApply.checked,
          }}),
        }});
        renderSnapshot(snapshot);
        if (pollHandle) clearInterval(pollHandle);
        pollHandle = setInterval(pollSnapshot, 900);
      }} catch (error) {{
        runButton.disabled = false;
        statusBadge.textContent = error.message;
        statusBadge.className = "badge error";
      }}
    }}

    async function resolveWrite(approve) {{
      if (!currentSessionId) return;
      approveButton.disabled = true;
      rejectButton.disabled = true;
      try {{
        const snapshot = await request(`/api/session/${{currentSessionId}}/confirm_write`, {{
          method: "POST",
          body: JSON.stringify({{ approve }}),
        }});
        renderSnapshot(snapshot);
      }} catch (error) {{
        statusBadge.textContent = error.message;
        statusBadge.className = "badge error";
      }} finally {{
        approveButton.disabled = false;
        rejectButton.disabled = false;
      }}
    }}

    runButton.addEventListener("click", startRun);
    approveButton.addEventListener("click", () => resolveWrite(true));
    rejectButton.addEventListener("click", () => resolveWrite(false));
    taskInput.addEventListener("keydown", (event) => {{
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {{
        startRun();
      }}
    }});

    async function bootstrap() {{
      try {{
        const info = await request("/api/info");
        repoInput.value = info.default_repo || defaultRepo;
        defaultRepoLabel.textContent = info.default_repo || defaultRepo;
        if (info.latest_session_id) {{
          currentSessionId = info.latest_session_id;
          await pollSnapshot();
          if (pollHandle) clearInterval(pollHandle);
          const latest = await request(`/api/session/${{currentSessionId}}`);
          if (!latest.finished) {{
            pollHandle = setInterval(pollSnapshot, 900);
          }}
        }}
      }} catch (error) {{
        statusBadge.textContent = error.message;
        statusBadge.className = "badge error";
      }}
    }}

    renderEvents([]);
    bootstrap();
  </script>
</body>
</html>"""


def _make_handler(studio: ForgeStudio, default_repo: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ForgeStudio/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self._send_html(_page(str(default_repo)))
            if parsed.path == "/api/info":
                return self._send_json(studio.info())
            if parsed.path.startswith("/api/session/"):
                session_id = parsed.path.rsplit("/", 1)[-1]
                snapshot = studio.snapshot(session_id)
                if snapshot is None:
                    return self._send_json({"error": "Session not found."}, status=HTTPStatus.NOT_FOUND)
                return self._send_json(snapshot)
            return self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/session":
                try:
                    session = studio.create_session(
                        task=str(payload.get("task", "")),
                        repo=payload.get("repo"),
                        auto_apply=bool(payload.get("auto_apply")),
                    )
                except ValueError as exc:
                    return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                except RuntimeError as exc:
                    return self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                return self._send_json(session.snapshot(), status=HTTPStatus.CREATED)

            if parsed.path.startswith("/api/session/") and parsed.path.endswith("/confirm_write"):
                session_id = parsed.path.split("/")[3]
                snapshot = studio.resolve_write(session_id, bool(payload.get("approve")))
                if snapshot is None:
                    return self._send_json({"error": "Session not found."}, status=HTTPStatus.NOT_FOUND)
                return self._send_json(snapshot)

            return self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def serve_studio(
    *,
    repo: Path,
    config_path: Path | None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> str:
    studio = ForgeStudio(repo, config_path=config_path)
    handler = _make_handler(studio, repo.resolve())
    server = ThreadingHTTPServer((host, port), handler)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{server.server_port}"

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    finally:
        server.server_close()

    return url
