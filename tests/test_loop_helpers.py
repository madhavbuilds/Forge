"""Loop helper behavior."""

from pathlib import Path

from forge.agent.loop import AgentLoop
from forge.agent.memory import CodeMemory
from forge.config import ForgeConfig


class _DummyClient:
    def prompt_cache(self):  # pragma: no cover
        return None


def _cfg() -> ForgeConfig:
    return ForgeConfig(
        fast_model="fast",
        smart_model="smart",
        ollama_host="http://localhost:11434",
        keep_alive="1h",
        max_tool_calls=3,
        auto_checkpoint=True,
        max_fix_retries=2,
        confirm_writes=True,
        confirm_commands=True,
        streaming=True,
        show_tool_calls=True,
        theme="dark",
        compact_mode=False,
        cache_enabled=True,
        cache_db_path=Path("/tmp/forge-cache.db"),
        cache_max_entries=100,
        raw={},
    )


def _loop(repo: Path) -> AgentLoop:
    return AgentLoop(_cfg(), repo, _DummyClient(), CodeMemory(repo))


def test_extra_context_files_prefers_readme_for_project_overview(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Forge\n\nOffline coding assistant\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name":"forge"}', encoding="utf-8")
    loop = _loop(tmp_path)
    assert loop._extra_context_files("what does this project do?") == ["README.md", "package.json"]


def test_website_task_detection_for_html_css_request(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    assert loop._is_website_task("make a whole website in html and css") is True


def test_allowed_website_paths_are_constrained(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    assert loop._is_allowed_website_path("index.html") is True
    assert loop._is_allowed_website_path("styles.css") is True
    assert loop._is_allowed_website_path("app/page.tsx") is False


def test_casual_chat_detection_handles_greetings(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    assert loop._is_casual_chat("hi") is True
    assert loop._is_casual_chat("hello there") is True
    assert loop._is_casual_chat("what files are in this project?") is False


def test_latest_user_request_extracts_current_turn(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    wrapped = (
        "Conversation context:\n"
        "User: hi\n"
        "Assistant: hello\n\n"
        "Current user request: make a landing page in html and css"
    )
    assert loop._latest_user_request(wrapped) == "make a landing page in html and css"


def test_fallback_project_overview_uses_readme_and_stack(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    results = [
        {
            "tool": "read_file",
            "result": {
                "ok": True,
                "path": "README.md",
                "content": "# Forge\n\nOffline local coding assistant",
            },
        },
        {
            "tool": "read_file",
            "result": {
                "ok": True,
                "path": "pyproject.toml",
                "content": 'dependencies = ["typer", "rich", "litellm"]',
            },
        },
    ]
    text = loop._fallback_response("what does this project do?", results)
    assert "Offline local coding assistant" in text
    assert "Typer CLI" in text


def test_derive_project_name_uses_named_phrase(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    assert loop._derive_project_name("create a website called Pixel Studio") == "pixel-studio"


def test_ensure_unique_project_dir_increments_name(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    (tmp_path / "forge-site").mkdir()
    candidate = loop._ensure_unique_project_dir("forge-site")
    assert candidate.name == "forge-site-2"


def test_derive_project_name_prefers_target_after_for_phrase(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    name = loop._derive_project_name("make a premium landing page for a local AI coding tool in html and css")
    assert name == "local-ai-coding-tool"


def test_skip_tests_for_static_site_files(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop.files_touched = ["local-ai-coding-tool/index.html", "local-ai-coding-tool/styles.css"]
    assert loop._should_skip_tests() is True


def test_website_result_summary_is_clean(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    results = [
        {"tool": "act", "result": {"ok": True, "summary": "Created a polished landing page."}},
        {"tool": "write_file", "result": {"ok": True, "path": "local-ai-coding-tool/index.html"}},
        {"tool": "write_file", "result": {"ok": True, "path": "local-ai-coding-tool/styles.css"}},
    ]
    text = loop._website_result_summary(results)
    assert "Created a polished landing page." in text
    assert "Project folder: `local-ai-coding-tool`" in text
    assert "- `local-ai-coding-tool/index.html`" in text


def test_default_website_files_are_premium_and_constrained(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    files = loop._default_website_files(
        "make a premium landing page for a local AI coding tool in html and css",
        "local-ai-coding-tool",
    )
    assert len(files) == 2
    html = next(item.content for item in files if item.path.endswith("index.html"))
    css = next(item.content for item in files if item.path.endswith("styles.css"))
    assert "feature-section" in html
    assert "workflow-section" in html
    assert "cta-section" in html
    assert '"Sora"' in css
    assert "font-family: Arial" not in css


def test_high_quality_website_check_rejects_plain_template(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    files = [
        type("F", (), {"path": "demo/index.html", "content": "<html><body><h1>Welcome to</h1></body></html>"})(),
        type("F", (), {"path": "demo/styles.css", "content": "body { font-family: Arial; background-color: #333; }"})(),
    ]
    assert loop._is_high_quality_website(files) is False
