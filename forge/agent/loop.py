"""Linear agent flow: classify -> gather -> act -> respond."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from forge.agent import router
from forge.agent.memory import CodeMemory
from forge.config import ForgeConfig
from forge.models.client import ModelClient
from forge.tools import filesystem, git, search, shell

StreamFn = Callable[[str], None]
StreamEndFn = Callable[[], None]
StatusFn = Callable[[str], None]

RESPOND_SYSTEM = (
    "You are Forge, a helpful local AI coding assistant. Answer in natural, polished plain English "
    "using the provided tool results. Synthesize the information instead of echoing raw output. "
    "If the user asked about a project, summarize what it does, the stack, and the most relevant files. "
    "If the user asked for a list, use bullets. If information is incomplete, say what you can infer. "
    "Never output JSON. Never call tools."
)
DIRECT_CHAT_SYSTEM = (
    "You are Forge, a friendly local AI assistant. Reply naturally and helpfully like a normal chat assistant. "
    "Keep the tone warm, clear, and concise. Do not mention tools or internal steps unless the user asks. "
    "When the user wants a website, Forge builds lightweight static HTML and CSS projects."
)

FILE_PATTERN = re.compile(r"`([^`]+)`|([A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)")
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
SYMBOL_PATTERNS = (
    re.compile(rf"`({IDENTIFIER})`"),
    re.compile(rf"\b({IDENTIFIER})\(\)"),
    re.compile(rf"\b(?:function|class|method|def)\s+({IDENTIFIER})\b"),
)
SKIP_PARTS = {".git", ".forge", ".venv", "__pycache__", "node_modules"}
OVERVIEW_KEYWORDS = (
    "what does this project do",
    "what does the project do",
    "what is this project",
    "what does this repo do",
    "what is this repo",
    "explain this project",
    "explain this repo",
    "describe this project",
    "describe this repo",
)
WEBSITE_KEYWORDS = (
    "website",
    "landing page",
    "homepage",
    "portfolio",
    "marketing site",
    "web page",
    "web app",
    "dashboard",
    "html",
    "css",
)
SITE_RELEVANT_FILES = (
    "index.html",
    "styles.css",
    "README.md",
)
MAX_WEBSITE_FILES = 2
CASUAL_CHAT_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "yo",
    "how are you",
    "thanks",
    "thank you",
    "who are you",
    "what can you do",
    "help",
)
ACTIVE_PROJECT_MARKER = "Active website project:"


@dataclass(slots=True)
class GatherContext:
    file_path: str | None
    symbol: str | None
    results: list[dict[str, Any]]


@dataclass(slots=True)
class GeneratedFile:
    path: str
    content: str


class AgentLoop:
    def __init__(
        self,
        cfg: ForgeConfig,
        repo: Path,
        client: ModelClient,
        memory: CodeMemory,
        *,
        stream: StreamFn | None = None,
        stream_end: StreamEndFn | None = None,
        status: StatusFn | None = None,
        confirm_write: Callable[[str, str], bool] | None = None,
        confirm_cmd: Callable[[str], bool] | None = None,
        tool_echo: Callable[[str, dict[str, Any]], None] | None = None,
        debug: bool = False,
    ) -> None:
        self.cfg = cfg
        self.repo = repo.resolve()
        self.client = client
        self.memory = memory
        self.stream = stream or (lambda s: None)
        self.stream_end = stream_end or (lambda: None)
        self.status = status or (lambda s: None)
        self.confirm_write = confirm_write
        self.confirm_cmd = confirm_cmd
        self.tool_echo = tool_echo
        self.debug = debug
        self.fs_state = filesystem.FilesystemState()
        self.intent = "search"
        self.files_touched: list[str] = []
        self.commands_run: list[str] = []
        self.tests_ok: bool | None = None
        self._user_task = ""
        self._writes_done = 0

    def _dbg(self, message: str, data: Any | None = None) -> None:
        if not self.debug:
            return
        payload = f"[forge:debug] {message}"
        if data is not None:
            payload += f"\n{json.dumps(data, indent=2, ensure_ascii=False, default=str)}"
        print(payload)

    async def run(self, task: str) -> None:
        self._user_task = task
        latest_task = self._latest_user_request(task)
        self.intent = router.classify(latest_task)
        self.memory.remember(f"task: {latest_task}")
        self.status(f"PLANNING · {self.intent}")

        if self._is_casual_chat(latest_task):
            await self._respond_direct(task, latest_task=latest_task)
            self.client.prompt_cache.remember_session(self.memory.session_notes)
            return

        gathered = await self._gather(latest_task)
        action_results: list[dict[str, Any]] = []
        if self.intent == "edit_code":
            action_results = await self._act(task, gathered)

        await self._respond(task, gathered.results + action_results)
        self.client.prompt_cache.remember_session(self.memory.session_notes)

    async def _gather(self, task: str) -> GatherContext:
        self.status("THINKING · reading project context…")
        file_path = self._extract_file_mention(task)
        symbol = self._extract_symbol_mention(task, file_path=file_path)
        max_calls = min(self.cfg.max_tool_calls, 3)

        calls: list[tuple[str, dict[str, Any]]] = [("list_directory", {"path": ".", "depth": 2})]
        if file_path and len(calls) < max_calls:
            calls.append(("read_file", {"path": file_path}))
        if symbol and len(calls) < max_calls:
            calls.append(("search_codebase", {"query": symbol}))
        if not file_path and len(calls) < max_calls:
            for extra_path in self._extra_context_files(task):
                if len(calls) >= max_calls:
                    break
                if any(existing_args.get("path") == extra_path for name, existing_args in calls if name == "read_file"):
                    continue
                calls.append(("read_file", {"path": extra_path}))

        results = list(await asyncio.gather(*(self._run_gather_tool(name, args) for name, args in calls)))
        return GatherContext(file_path=file_path, symbol=symbol, results=results)

    async def _run_gather_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.tool_echo:
            self.tool_echo(name, args)
        self.memory.remember(f"tool {name} {args}")

        try:
            if name == "list_directory":
                result = await asyncio.to_thread(
                    filesystem.list_directory,
                    str(args["path"]),
                    int(args["depth"]),
                    repo_root=self.repo,
                    state=self.fs_state,
                )
                return {"tool": name, "result": result}
            if name == "read_file":
                path = str(args["path"])
                full = (self.repo / path).resolve()
                try:
                    full.relative_to(self.repo)
                except ValueError:
                    return {"tool": name, "error": "path outside repo"}
                result = await asyncio.to_thread(
                    filesystem.read_file,
                    path,
                    repo_root=self.repo,
                    state=self.fs_state,
                )
                return {"tool": name, "result": result}
            if name == "search_codebase":
                result = await asyncio.to_thread(
                    search.search_codebase,
                    str(args["query"]),
                    repo_root=self.repo,
                    faiss_hits=None,
                )
                return {"tool": name, "result": result}
        except Exception as exc:
            return {"tool": name, "error": str(exc)}

        return {"tool": name, "error": "unknown tool"}

    async def _act(self, task: str, gathered: GatherContext) -> list[dict[str, Any]]:
        if self._active_website_project(task) or self._is_website_task(task):
            return await self._act_website(task, gathered)

        self.status("CODING · preparing edit…")
        target_path = gathered.file_path or self._select_target_file(gathered.results)
        if not target_path:
            return [
                {
                    "tool": "write_file",
                    "result": {"ok": False, "error": "no target file could be determined"},
                }
            ]

        if self._writes_done >= 1:
            return [
                {
                    "tool": "write_file",
                    "result": {"ok": False, "error": "maximum writes reached for this session"},
                }
            ]

        target = (self.repo / target_path).resolve()
        if not target.is_file():
            return [
                {
                    "tool": "write_file",
                    "result": {"ok": False, "error": f"target is not a file: {target_path}"},
                }
            ]

        original = target.read_text(encoding="utf-8", errors="replace")
        self.fs_state.mark_read(target_path)

        self.status("CODING · writing file update…")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Forge SMART. Return only the full updated file contents as plain code. "
                    "Do not explain the change. Do not output markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"Target file: {target_path}\n\n"
                    "Current file contents:\n"
                    f"{original}\n\n"
                    "Gathered context:\n"
                    f"{self._results_as_plain_text(gathered.results)}"
                ),
            },
        ]
        updated = self._strip_code_fences(
            await self.client.complete_text(messages, intent="edit_code")
        )

        if not updated.strip():
            return [{"tool": "write_file", "result": {"ok": False, "error": "empty model output"}}]

        if self.cfg.confirm_writes and self.confirm_write:
            if not self.confirm_write(target_path, updated):
                return [{"tool": "write_file", "result": {"ok": False, "cancelled": True}}]

        if self.cfg.auto_checkpoint and (self.repo / ".git").exists():
            git.git_checkpoint(self.repo)

        write_result = filesystem.write_file(
            target_path,
            updated,
            repo_root=self.repo,
            state=self.fs_state,
        )
        results: list[dict[str, Any]] = [{"tool": "write_file", "result": write_result}]
        if write_result.get("ok"):
            self._writes_done += 1
            self.files_touched.append(target_path)
            test_result = await self._maybe_run_tests()
            if test_result is not None:
                results.append({"tool": "run_command", "result": test_result})
        return results

    async def _act_website(self, task: str, gathered: GatherContext) -> list[dict[str, Any]]:
        self.status("PLANNING · shaping the website…")
        active_project = self._active_website_project(task)
        if active_project:
            project_dir = self.repo / active_project
            project_name = project_dir.name
        else:
            project_name = self._derive_project_name(task)
            project_dir = self._ensure_unique_project_dir(project_name)
        context = self._website_context_text(gathered.results)
        fallback_files = self._default_website_files(task, project_dir.name)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Forge SMART, a strong local website builder. Return one JSON object only with this schema: "
                    '{"summary": string, "files": [{"path": string, "content": string}]}. '
                    "Create a lightweight static website using only plain HTML and plain CSS. "
                    "Do not use JavaScript, frameworks, build tools, package managers, React, Next.js, or Tailwind CSS. "
                    f"Return at most {MAX_WEBSITE_FILES} files. "
                    'The files must be "index.html" and "styles.css". '
                    "Use modern responsive layouts, accessible markup, and clean CSS. "
                    "Make the site feel polished but keep it lightweight. "
                    "The layout must include a strong hero, trust strip, feature cards, workflow section, testimonial or proof section, and final CTA. "
                    "Avoid generic boilerplate, Arial, flat dark headers, and basic one-column document styling."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"Project folder name:\n{project_dir.name}\n\n"
                    f"Existing project:\n{'yes' if active_project else 'no'}\n\n"
                    "Existing repo context:\n"
                    f"{context}\n\n"
                    "Return the complete contents for every file that should be written."
                ),
            },
        ]
        data = await self.client.complete_json(messages, intent="edit_code")
        raw_files = data.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            files = fallback_files
            summary = f"Created a polished landing page in {project_dir.name}."
            return await self._apply_generated_files(files, summary=summary)

        files: list[GeneratedFile] = []
        for item in raw_files[:MAX_WEBSITE_FILES]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            if not path or not content.strip():
                continue
            if not self._is_allowed_website_path(path):
                continue
            files.append(GeneratedFile(path=f"{project_dir.name}/{path}", content=self._strip_code_fences(content)))

        if not files:
            files = fallback_files

        if not self._is_high_quality_website(files):
            files = fallback_files

        summary = str(data.get("summary", "")).strip() or f"Created a polished landing page in {project_dir.name}"
        return await self._apply_generated_files(files, summary=summary)

    async def _respond(self, task: str, results: list[dict[str, Any]]) -> None:
        self.status("RESPONDING · writing the final answer…")
        website_summary = self._website_result_summary(results)
        if website_summary:
            self.stream(website_summary)
            self.stream_end()
            return

        messages = [
            {"role": "system", "content": RESPOND_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    "Tool results:\n"
                    f"{self._results_as_plain_text(results)}"
                ),
            },
        ]

        if self.cfg.streaming:
            text = await self.client.complete_text_stream(
                messages,
                intent="search",
                on_chunk=self.stream,
                on_stream_end=self.stream_end,
            )
            if text.strip():
                return

        text = await self.client.complete_text(messages, intent="search")
        if not text.strip():
            text = self._fallback_response(task, results)
        self.stream(text)
        self.stream_end()

    async def _respond_direct(self, task: str, *, latest_task: str) -> None:
        self.status("RESPONDING · chatting…")
        messages = [
            {"role": "system", "content": DIRECT_CHAT_SYSTEM},
            {"role": "user", "content": task},
        ]
        if self.cfg.streaming:
            text = await self.client.complete_text_stream(
                messages,
                intent="search",
                on_chunk=self.stream,
                on_stream_end=self.stream_end,
            )
            if text.strip():
                return
        text = await self.client.complete_text(messages, intent="search")
        if not text.strip():
            text = self._fallback_chat_reply(latest_task)
        self.stream(text)
        self.stream_end()

    async def _apply_generated_files(
        self,
        files: list[GeneratedFile],
        *,
        summary: str = "",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        checkpoint_done = False
        for generated in files:
            path = generated.path
            target = (self.repo / path).resolve()
            try:
                target.relative_to(self.repo)
            except ValueError:
                results.append(
                    {"tool": "write_file", "result": {"ok": False, "error": f"path outside repo: {path}"}}
                )
                continue

            if target.exists():
                self.fs_state.mark_read(path)
            else:
                self.fs_state.mark_listed(str(Path(path).parent))

            if self.cfg.confirm_writes and self.confirm_write and not self.confirm_write(path, generated.content):
                results.append({"tool": "write_file", "result": {"ok": False, "path": path, "cancelled": True}})
                continue

            if self.cfg.auto_checkpoint and not checkpoint_done and (self.repo / ".git").exists():
                git.git_checkpoint(self.repo)
                checkpoint_done = True

            write_result = filesystem.write_file(
                path,
                generated.content,
                repo_root=self.repo,
                state=self.fs_state,
            )
            results.append({"tool": "write_file", "result": write_result})
            if write_result.get("ok"):
                self.files_touched.append(path)
                self._writes_done += 1

        if summary:
            results.insert(0, {"tool": "act", "result": {"ok": True, "summary": summary}})

        if any(item.get("tool") == "write_file" and item.get("result", {}).get("ok") for item in results):
            test_result = await self._maybe_run_tests()
            if test_result is not None:
                results.append({"tool": "run_command", "result": test_result})
        return results

    async def _maybe_run_tests(self) -> dict[str, Any] | None:
        if self._should_skip_tests():
            self.tests_ok = None
            return None

        has_tests = (self.repo / "tests").is_dir() or (self.repo / "test").is_dir()
        pytest_ini = self.repo / "pytest.ini"
        pyproject = self.repo / "pyproject.toml"
        if not has_tests and not pytest_ini.exists():
            if not pyproject.exists():
                return None
            text = pyproject.read_text(encoding="utf-8", errors="replace")
            if "[tool.pytest" not in text and "pytest" not in text:
                return None

        self.status("CHECKING · running tests…")
        cmd = "python -m pytest -q"
        self.commands_run.append(cmd)
        result = await shell.run_command(cmd, timeout=300)
        self.tests_ok = bool(result.get("ok"))
        return result

    def _extract_file_mention(self, task: str) -> str | None:
        for match in FILE_PATTERN.finditer(task):
            candidate = (match.group(1) or match.group(2) or "").strip().strip(".,:;()[]{}\"'")
            if not candidate:
                continue
            resolved = self._resolve_file_candidate(candidate)
            if resolved:
                return resolved
        return None

    def _resolve_file_candidate(self, candidate: str) -> str | None:
        direct: Path | None = (self.repo / candidate).resolve()
        try:
            direct.relative_to(self.repo)
        except ValueError:
            direct = None
        if direct is not None and direct.is_file():
            return str(direct.relative_to(self.repo))

        matches = [
            path
            for path in self.repo.rglob(candidate)
            if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
        ]
        if not matches:
            return None
        matches.sort(key=lambda path: (len(path.parts), str(path)))
        return str(matches[0].relative_to(self.repo))

    def _extract_symbol_mention(self, task: str, *, file_path: str | None) -> str | None:
        for pattern in SYMBOL_PATTERNS:
            match = pattern.search(task)
            if not match:
                continue
            symbol = match.group(1)
            if "." not in symbol:
                return symbol

        skip = {Path(file_path).stem} if file_path else set()
        keywords = {keyword.replace(" ", "") for keyword in (*router.READ_KEYWORDS, *router.WRITE_KEYWORDS)}
        for raw in re.findall(rf"\b{IDENTIFIER}\b", task):
            if raw in skip:
                continue
            if raw.lower() in keywords:
                continue
            if raw[0].isupper() or "_" in raw:
                return raw
        return None

    def _extra_context_files(self, task: str) -> list[str]:
        task_lower = task.lower()
        extra: list[str] = []
        if any(keyword in task_lower for keyword in OVERVIEW_KEYWORDS):
            for candidate in ("README.md", "package.json", "pyproject.toml"):
                if (self.repo / candidate).is_file():
                    extra.append(candidate)
            return extra

        if self._is_website_task(task):
            active_project = self._active_website_project(task)
            if active_project:
                for candidate in (f"{active_project}/index.html", f"{active_project}/styles.css"):
                    if (self.repo / candidate).is_file():
                        extra.append(candidate)
                return extra
            for candidate in ("index.html", "styles.css", "README.md"):
                if (self.repo / candidate).is_file():
                    extra.append(candidate)
        return extra

    def _select_target_file(self, results: list[dict[str, Any]]) -> str | None:
        for item in results:
            result = item.get("result")
            if item.get("tool") == "read_file" and isinstance(result, dict) and result.get("ok"):
                path = result.get("path")
                if isinstance(path, str) and path:
                    return path

        for item in results:
            result = item.get("result")
            if item.get("tool") != "search_codebase" or not isinstance(result, dict):
                continue
            hits = result.get("ripgrep")
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                path = hit.get("path")
                if isinstance(path, str) and path:
                    return path
        return None

    def _results_as_plain_text(self, results: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for item in results:
            tool = str(item.get("tool", "tool"))
            if "error" in item:
                blocks.append(f"{tool}\nerror: {item['error']}")
                continue
            result = item.get("result", {})
            blocks.append(f"{tool}\n{self._result_block(tool, result)}")
        return "\n\n".join(blocks) if blocks else "No tool results."

    def _result_block(self, tool: str, result: Any) -> str:
        if tool == "list_directory" and isinstance(result, dict):
            tree = str(result.get("tree", "")).strip()
            if tree:
                return self._summarize_tree(tree)
        if tool == "read_file" and isinstance(result, dict):
            path = str(result.get("path", ""))
            content = str(result.get("content", "")).strip()
            if path and content:
                return f"path: {path}\ncontent:\n{_trim_block(content, 6000)}"
        return self._value_as_plain_text(result, indent=0)

    def _summarize_tree(self, tree: str) -> str:
        lines = [line.rstrip() for line in tree.splitlines() if line.strip()]
        cleaned: list[str] = []
        for line in lines[:120]:
            parts = line.split("\t", 1)[0]
            parts = parts.lstrip("-d ").rstrip()
            cleaned.append(parts)
        return "\n".join(cleaned)

    def _website_context_text(self, gathered_results: list[dict[str, Any]]) -> str:
        parts = [self._results_as_plain_text(gathered_results)]
        for candidate in SITE_RELEVANT_FILES:
            path = self.repo / candidate
            if path.is_dir():
                continue
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                parts.append(f"{candidate}:\n{_trim_block(text, 4000)}")
        return "\n\n".join(part for part in parts if part.strip())

    def _is_website_task(self, task: str) -> bool:
        task_lower = task.lower()
        if self._active_website_project(task):
            return True
        if any(keyword in task_lower for keyword in WEBSITE_KEYWORDS):
            return True
        if any(keyword in task_lower for keyword in ("build a site", "build a page", "make a site", "make a page")):
            return True
        if any(keyword in task_lower for keyword in ("redesign", "hero section", "single page", "static site")):
            return True
        return False

    def _active_website_project(self, task: str) -> str | None:
        match = re.search(rf"{re.escape(ACTIVE_PROJECT_MARKER)}\s*([A-Za-z0-9][A-Za-z0-9_-]*)", task)
        if not match:
            return None
        candidate = match.group(1).strip()
        project_dir = self.repo / candidate
        if not project_dir.is_dir():
            return None
        return candidate

    def _is_casual_chat(self, task: str) -> bool:
        task_lower = task.strip().lower()
        if not task_lower:
            return False
        if task_lower in CASUAL_CHAT_PATTERNS:
            return True
        short = len(task_lower.split()) <= 6
        return short and any(task_lower.startswith(pattern) for pattern in CASUAL_CHAT_PATTERNS)

    def _latest_user_request(self, task: str) -> str:
        marker = "Current user request:"
        if marker not in task:
            return task.strip()
        latest = task.rsplit(marker, 1)[-1].strip()
        return latest or task.strip()

    def _fallback_chat_reply(self, task: str) -> str:
        task_lower = task.strip().lower()
        if any(task_lower.startswith(word) for word in ("hi", "hello", "hey", "yo")):
            return (
                "Hi! I can chat normally, explain this project, or build lightweight websites with HTML and CSS."
            )
        if "what can you do" in task_lower or task_lower == "help":
            return (
                "I can answer questions about this repo, explain code, and build simple websites using HTML and CSS."
            )
        if "thanks" in task_lower or "thank you" in task_lower:
            return "Any time."
        return "I’m here. Ask a question or describe the website you want to make."

    def _fallback_response(self, task: str, results: list[dict[str, Any]]) -> str:
        latest_task = self._latest_user_request(task).lower()
        read_files = self._read_file_map(results)

        if any(keyword in latest_task for keyword in OVERVIEW_KEYWORDS):
            return self._fallback_project_overview(read_files)

        if "file" in latest_task or "files" in latest_task or latest_task.startswith(("list", "show", "what")):
            directory_tree = self._directory_tree(results)
            if directory_tree:
                lines = [line for line in directory_tree.splitlines() if line.strip()][:20]
                bullets = "\n".join(f"- {line}" for line in lines)
                return f"Here are the main files and folders I found:\n{bullets}"

        act_summary = self._action_summary(results)
        if act_summary:
            return act_summary

        return "I looked through the project and found relevant context, but I couldn't turn it into a clean answer yet."

    def _fallback_project_overview(self, read_files: dict[str, str]) -> str:
        readme = read_files.get("README.md", "").strip()
        pyproject = read_files.get("pyproject.toml", "")
        package_json = read_files.get("package.json", "")

        summary_lines: list[str] = []
        if readme:
            meaningful = [
                line.strip("# ").strip()
                for line in readme.splitlines()
                if line.strip() and not line.strip().startswith(("```", "- ["))
            ]
            if meaningful:
                summary_lines.append(meaningful[0])
            if len(meaningful) > 1:
                summary_lines.append(meaningful[1])

        stack: list[str] = []
        if "typer" in pyproject.lower():
            stack.append("Typer CLI")
        if "rich" in pyproject.lower():
            stack.append("Rich terminal UI")
        if "litellm" in pyproject.lower():
            stack.append("LiteLLM model routing")
        if '"next"' in package_json.lower():
            stack.append("Next.js")
        if "tailwind" in package_json.lower():
            stack.append("Tailwind CSS")

        lines: list[str] = []
        if summary_lines:
            lines.append("This project is:")
            lines.extend(f"- {line}" for line in summary_lines[:2])
        if stack:
            lines.append("")
            lines.append("Main stack:")
            lines.extend(f"- {item}" for item in stack)
        if not lines:
            return "This looks like a local coding assistant project, but I’d need a bit more project context to summarize it well."
        return "\n".join(lines)

    def _read_file_map(self, results: list[dict[str, Any]]) -> dict[str, str]:
        read_files: dict[str, str] = {}
        for item in results:
            result = item.get("result")
            if item.get("tool") != "read_file" or not isinstance(result, dict) or not result.get("ok"):
                continue
            path = result.get("path")
            content = result.get("content")
            if isinstance(path, str) and isinstance(content, str):
                read_files[path] = content
        return read_files

    def _directory_tree(self, results: list[dict[str, Any]]) -> str:
        for item in results:
            result = item.get("result")
            if item.get("tool") == "list_directory" and isinstance(result, dict):
                tree = result.get("tree")
                if isinstance(tree, str):
                    return tree
        return ""

    def _action_summary(self, results: list[dict[str, Any]]) -> str:
        touched = []
        for item in results:
            result = item.get("result")
            if item.get("tool") == "write_file" and isinstance(result, dict) and result.get("ok"):
                path = result.get("path")
                if isinstance(path, str):
                    touched.append(path)
        if touched:
            bullets = "\n".join(f"- {path}" for path in touched[:8])
            return f"I updated these files:\n{bullets}"
        return ""

    def _website_result_summary(self, results: list[dict[str, Any]]) -> str:
        touched = []
        summary = ""
        for item in results:
            if item.get("tool") == "act":
                result = item.get("result", {})
                if isinstance(result, dict):
                    summary = str(result.get("summary", "")).strip()
            if item.get("tool") == "write_file":
                result = item.get("result", {})
                if isinstance(result, dict) and result.get("ok"):
                    path = result.get("path")
                    if isinstance(path, str):
                        touched.append(path)

        if not touched or not all(path.endswith((".html", ".css")) for path in touched):
            return ""

        project_root = Path(touched[0]).parts[0] if Path(touched[0]).parts else ""
        lines = []
        if summary:
            lines.append(summary)
        else:
            lines.append(f"Created a new website in `{project_root}`.")
        if project_root:
            lines.append(f"Project folder: `{project_root}`")
        lines.append("Files created:")
        lines.extend(f"- `{path}`" for path in touched)
        lines.append("Open `index.html` in your browser to preview it.")
        return "\n".join(lines)

    def _should_skip_tests(self) -> bool:
        if not self.files_touched:
            return False
        return all(path.endswith((".html", ".css")) for path in self.files_touched)

    def _is_allowed_website_path(self, path: str) -> bool:
        return path in {"index.html", "styles.css"}

    def _is_high_quality_website(self, files: list[GeneratedFile]) -> bool:
        html = next((item.content for item in files if item.path.endswith("index.html")), "")
        css = next((item.content for item in files if item.path.endswith("styles.css")), "")
        if not html or not css:
            return False
        html_lower = html.lower()
        css_lower = css.lower()
        required_html = ("hero", "feature", "testimonial", "cta")
        if not all(token in html_lower for token in required_html):
            return False
        if len(css) < 1800:
            return False
        low_quality_markers = ("font-family: arial", "background-color: #333", "welcome to")
        if any(marker in css_lower or marker in html_lower for marker in low_quality_markers):
            return False
        return True

    def _default_website_files(self, task: str, project_name: str) -> list[GeneratedFile]:
        brand = self._brand_name_from_project(project_name)
        audience = self._website_audience(task)
        headline = self._website_headline(task, brand)
        subhead = (
            f"Build, refine, and ship polished product ideas faster with a local-first coding flow designed for {audience}."
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{brand}</title>
  <meta name="description" content="{brand} helps teams turn ideas into polished websites with a local-first AI workflow.">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="page-shell">
    <header class="site-header">
      <div class="brand-lockup">
        <span class="brand-mark">F</span>
        <span class="brand-name">{brand}</span>
      </div>
      <nav class="site-nav" aria-label="Primary">
        <a href="#features">Features</a>
        <a href="#workflow">Workflow</a>
        <a href="#proof">Proof</a>
        <a href="#contact">Contact</a>
      </nav>
    </header>

    <main>
      <section class="hero" id="top">
        <div class="hero-copy">
          <p class="eyebrow">Offline-first product builder</p>
          <h1>{headline}</h1>
          <p class="hero-text">{subhead}</p>
          <div class="hero-actions">
            <a class="button button-primary" href="#contact">Start building</a>
            <a class="button button-secondary" href="#features">See the flow</a>
          </div>
          <ul class="hero-points" aria-label="Highlights">
            <li>Fast local workflow</li>
            <li>Clean visible planning and coding phases</li>
            <li>Simple static output that stays lightweight</li>
          </ul>
        </div>
        <aside class="hero-card" aria-label="Product snapshot">
          <div class="hero-card-top">
            <span class="pulse-dot"></span>
            <span>Live build session</span>
          </div>
          <div class="phase-list">
            <div><strong>Planning</strong><span>Shape the page and structure the sections.</span></div>
            <div><strong>Thinking</strong><span>Choose messaging, hierarchy, and layout direction.</span></div>
            <div><strong>Coding</strong><span>Generate clean HTML and CSS for a polished page.</span></div>
          </div>
          <div class="metric-row">
            <div><strong>0 cloud</strong><span>Local-first workflow</span></div>
            <div><strong>2 files</strong><span>Simple static delivery</span></div>
          </div>
        </aside>
      </section>

      <section class="trust-strip" aria-label="Trusted qualities">
        <p>Lightweight by design</p>
        <p>Built for laptops</p>
        <p>Fast local iteration</p>
        <p>Clear user flow</p>
      </section>

      <section class="feature-section" id="features">
        <div class="section-heading">
          <p class="eyebrow">Why it feels better</p>
          <h2>Designed for flow instead of friction.</h2>
          <p>A focused interface, stronger visual structure, and a tighter build path help the experience feel deliberate instead of noisy.</p>
        </div>
        <div class="feature-grid">
          <article class="feature-card">
            <h3>Chat that stays natural</h3>
            <p>Talk to the tool like a collaborator, not a command parser.</p>
          </article>
          <article class="feature-card">
            <h3>Visible progress</h3>
            <p>See planning, thinking, and coding without getting buried in technical internals.</p>
          </article>
          <article class="feature-card">
            <h3>Local-first speed</h3>
            <p>Keep the stack simple so your laptop stays responsive while you build.</p>
          </article>
        </div>
      </section>

      <section class="workflow-section" id="workflow">
        <div class="section-heading">
          <p class="eyebrow">Workflow</p>
          <h2>From idea to live page in one straight path.</h2>
        </div>
        <div class="workflow-grid">
          <article class="workflow-step">
            <span>01</span>
            <h3>Describe the idea</h3>
            <p>Start with a plain-English prompt describing the page you want.</p>
          </article>
          <article class="workflow-step">
            <span>02</span>
            <h3>Generate the project</h3>
            <p>The tool creates a clean project folder and writes the site files.</p>
          </article>
          <article class="workflow-step">
            <span>03</span>
            <h3>Refine the vibe</h3>
            <p>Keep iterating with simple feedback until the design feels right.</p>
          </article>
        </div>
      </section>

      <section class="proof-section" id="proof">
        <div class="quote-card testimonial-card">
          <p class="quote">“This feels calm, fast, and practical. It helps the work move without making the laptop feel heavy.”</p>
          <p class="quote-author">Product designer, local-first workflow team</p>
        </div>
        <div class="proof-panel">
          <h2>Built for makers who want clean momentum.</h2>
          <p>Use a narrow stack, a focused UI, and a stable local workflow to turn prompts into production-ready landing pages faster.</p>
          <ul>
            <li>Readable hierarchy and spacing</li>
            <li>Responsive layout without framework overhead</li>
            <li>Strong CTA flow and polished visual rhythm</li>
          </ul>
        </div>
      </section>

      <section class="cta-section" id="contact">
        <div class="cta-card">
          <p class="eyebrow">Ready to build</p>
          <h2>Make the next page feel premium from the first draft.</h2>
          <p>Start with a simple idea, generate the site, and keep refining until the design feels sharp.</p>
          <a class="button button-primary" href="mailto:hello@example.com">hello@example.com</a>
        </div>
      </section>
    </main>

    <footer class="site-footer">
      <p>{brand}</p>
      <p>Local-first website building for teams who care about speed, clarity, and polish.</p>
    </footer>
  </div>
</body>
</html>
"""
        css = """\
:root {
  --bg: #f4efe7;
  --bg-soft: #fffaf4;
  --surface: rgba(255, 252, 247, 0.82);
  --surface-strong: #fffdf9;
  --text: #1f2937;
  --muted: #5f6c7b;
  --line: rgba(31, 41, 55, 0.08);
  --brand: #0f766e;
  --brand-deep: #0b4f4a;
  --accent: #d76b4d;
  --accent-soft: #f8d8c7;
  --shadow: 0 24px 80px rgba(49, 37, 26, 0.12);
  --radius-xl: 32px;
  --radius-lg: 24px;
  --radius-md: 18px;
  --max: 1180px;
  --sans: "Sora", "Avenir Next", "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: var(--sans);
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30%),
    radial-gradient(circle at top right, rgba(215, 107, 77, 0.15), transparent 28%),
    linear-gradient(180deg, #fff8f0 0%, var(--bg) 52%, #ece2d3 100%);
}

a {
  color: inherit;
  text-decoration: none;
}

.page-shell {
  width: min(calc(100vw - 32px), var(--max));
  margin: 0 auto;
  padding: 24px 0 48px;
}

.site-header,
.hero,
.feature-section,
.workflow-section,
.proof-section,
.cta-card,
.site-footer {
  border: 1px solid var(--line);
  background: var(--surface);
  backdrop-filter: blur(16px);
  box-shadow: var(--shadow);
}

.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  border-radius: 999px;
  padding: 14px 20px;
  margin-bottom: 22px;
}

.brand-lockup {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  font-weight: 700;
}

.brand-mark {
  width: 38px;
  height: 38px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: white;
  background: linear-gradient(135deg, var(--brand), var(--accent));
}

.site-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  color: var(--muted);
  font-size: 0.95rem;
}

.hero {
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 28px;
  border-radius: var(--radius-xl);
  padding: 34px;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(15, 118, 110, 0.1);
  color: var(--brand);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero h1,
.section-heading h2,
.proof-panel h2,
.cta-card h2 {
  margin: 16px 0 0;
  line-height: 0.96;
  letter-spacing: -0.04em;
}

.hero h1 {
  font-size: clamp(2.8rem, 6vw, 5rem);
  max-width: 10ch;
}

.hero-text,
.section-heading p,
.proof-panel p,
.cta-card p,
.feature-card p,
.workflow-step p,
.quote-author,
.site-footer p {
  color: var(--muted);
  line-height: 1.7;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 28px 0 22px;
}

.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 48px;
  padding: 0 18px;
  border-radius: 999px;
  font-weight: 700;
  transition: transform 160ms ease, box-shadow 160ms ease;
}

.button:hover {
  transform: translateY(-1px);
}

.button-primary {
  color: white;
  background: linear-gradient(135deg, var(--brand), #169487);
  box-shadow: 0 14px 28px rgba(15, 118, 110, 0.24);
}

.button-secondary {
  color: var(--text);
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid var(--line);
}

.hero-points {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 10px;
}

.hero-points li::before,
.proof-panel li::before {
  content: "•";
  color: var(--accent);
  margin-right: 10px;
}

.hero-card,
.feature-card,
.workflow-step,
.quote-card,
.proof-panel,
.cta-card {
  border-radius: var(--radius-lg);
}

.hero-card {
  padding: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(244, 240, 232, 0.95));
  border: 1px solid rgba(31, 41, 55, 0.08);
}

.hero-card-top {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  margin-bottom: 20px;
}

.pulse-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 0 8px rgba(215, 107, 77, 0.15);
}

.phase-list,
.metric-row,
.feature-grid,
.workflow-grid,
.proof-section {
  display: grid;
  gap: 16px;
}

.phase-list div,
.metric-row div,
.feature-card,
.workflow-step,
.quote-card,
.proof-panel {
  padding: 18px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(31, 41, 55, 0.08);
}

.phase-list strong,
.metric-row strong,
.workflow-step span {
  display: block;
  margin-bottom: 6px;
}

.metric-row {
  grid-template-columns: repeat(2, 1fr);
  margin-top: 16px;
}

.trust-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin: 18px 0 22px;
}

.trust-strip p {
  margin: 0;
  padding: 16px;
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.6);
  border: 1px solid var(--line);
  text-align: center;
  color: var(--muted);
  font-weight: 600;
}

.feature-section,
.workflow-section,
.cta-card,
.site-footer {
  padding: 30px;
}

.feature-section,
.workflow-section,
.proof-section,
.cta-section {
  margin-top: 22px;
  border-radius: var(--radius-xl);
}

.section-heading {
  max-width: 720px;
  margin-bottom: 22px;
}

.section-heading h2,
.proof-panel h2,
.cta-card h2 {
  font-size: clamp(2rem, 4vw, 3.4rem);
}

.feature-grid {
  grid-template-columns: repeat(3, 1fr);
}

.feature-card h3,
.workflow-step h3 {
  margin: 0 0 10px;
  font-size: 1.15rem;
}

.workflow-grid {
  grid-template-columns: repeat(3, 1fr);
}

.workflow-step span {
  color: var(--brand);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.proof-section {
  grid-template-columns: 0.9fr 1.1fr;
  align-items: stretch;
}

.quote {
  margin: 0 0 18px;
  font-size: 1.4rem;
  line-height: 1.5;
  letter-spacing: -0.02em;
}

.proof-panel ul {
  list-style: none;
  padding: 0;
  margin: 20px 0 0;
  display: grid;
  gap: 10px;
}

.cta-card {
  text-align: center;
  background:
    radial-gradient(circle at top center, rgba(15, 118, 110, 0.18), transparent 38%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(246, 239, 231, 0.94));
}

.site-footer {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  margin-top: 22px;
  border-radius: var(--radius-xl);
}

@media (max-width: 960px) {
  .hero,
  .proof-section,
  .feature-grid,
  .workflow-grid,
  .trust-strip {
    grid-template-columns: 1fr;
  }

  .site-header,
  .site-footer {
    border-radius: 28px;
    flex-direction: column;
    align-items: flex-start;
  }
}

@media (max-width: 640px) {
  .page-shell {
    width: min(calc(100vw - 20px), var(--max));
    padding-top: 10px;
  }

  .hero,
  .feature-section,
  .workflow-section,
  .cta-card,
  .site-footer {
    padding: 22px;
  }

  .hero h1 {
    font-size: 2.5rem;
  }

  .metric-row {
    grid-template-columns: 1fr;
  }
}
"""
        return [
            GeneratedFile(path=f"{project_name}/index.html", content=html),
            GeneratedFile(path=f"{project_name}/styles.css", content=css),
        ]

    def _brand_name_from_project(self, project_name: str) -> str:
        parts = [part for part in project_name.replace("-", " ").split() if part]
        if not parts:
            return "Forge Studio"
        return " ".join(word.capitalize() if word.lower() != "ai" else "AI" for word in parts)

    def _website_audience(self, task: str) -> str:
        task_lower = task.lower()
        if "designer" in task_lower:
            return "design teams"
        if "startup" in task_lower or "saas" in task_lower:
            return "startup teams"
        if "developer" in task_lower or "coding" in task_lower:
            return "developers"
        return "small product teams"

    def _website_headline(self, task: str, brand: str) -> str:
        task_lower = task.lower()
        if "premium" in task_lower:
            return f"Premium local AI coding flow, without the cloud drag."
        if "portfolio" in task_lower:
            return f"A portfolio experience that feels sharp, focused, and modern."
        return f"{brand} turns rough ideas into polished website drafts fast."

    def _derive_project_name(self, task: str) -> str:
        quoted = re.search(r'["\']([^"\']{3,60})["\']', task)
        if quoted:
            return self._slugify_name(quoted.group(1))

        named = re.search(r"\b(?:called|named)\s+([A-Za-z0-9][A-Za-z0-9 _-]{2,50})", task, re.IGNORECASE)
        if named:
            return self._slugify_name(named.group(1))

        target = re.search(r"\bfor\s+(?:an?|the)\s+([A-Za-z0-9][A-Za-z0-9 _-]{2,50})", task, re.IGNORECASE)
        if target:
            phrase = re.split(r"\b(?:in|using|with)\b", target.group(1), maxsplit=1, flags=re.IGNORECASE)[0].strip()
            candidate = self._slugify_name(phrase)
            if candidate and candidate != "a":
                return candidate

        words = [
            word
            for word in re.findall(r"[A-Za-z0-9]+", task.lower())
            if word not in set(router.READ_KEYWORDS + router.WRITE_KEYWORDS)
            and word not in {
                "website",
                "page",
                "site",
                "html",
                "css",
                "landing",
                "build",
                "make",
                "create",
                "premium",
                "modern",
                "simple",
                "clean",
                "tool",
            }
        ]
        if words:
            return self._slugify_name(" ".join(words[:3]))
        return "forge-site"

    def _slugify_name(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:40] or "forge-site"

    def _ensure_unique_project_dir(self, base_name: str) -> Path:
        candidate = self.repo / base_name
        if not candidate.exists():
            return candidate
        for idx in range(2, 50):
            numbered = self.repo / f"{base_name}-{idx}"
            if not numbered.exists():
                return numbered
        return self.repo / f"{base_name}-{abs(hash(base_name)) % 10000}"

    def _value_as_plain_text(self, value: Any, *, indent: int) -> str:
        prefix = "  " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._value_as_plain_text(item, indent=indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {item}")
            return "\n".join(lines)
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(self._value_as_plain_text(item, indent=indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
            return "\n".join(lines)
        return f"{prefix}{value}"

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip("\n")


def summarize_session(memory: CodeMemory) -> str:
    return f"remembering {memory.session_count()} things from this session"


def _trim_block(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\n"
