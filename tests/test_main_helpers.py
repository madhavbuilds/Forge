"""CLI helper behavior."""

from pathlib import Path

from forge.__main__ import _task_with_history, _website_project_from_files


def test_task_with_history_includes_active_project_marker() -> None:
    task = _task_with_history(
        "make the hero section cleaner",
        [("make a landing page", "created the site")],
        Path("/tmp/local-ai-coding-tool"),
    )
    assert "Active website project: local-ai-coding-tool" in task
    assert "Current user request: make the hero section cleaner" in task


def test_website_project_from_files_detects_project_root(tmp_path: Path) -> None:
    project = tmp_path / "local-ai-coding-tool"
    project.mkdir()
    (project / "index.html").write_text("<!doctype html>", encoding="utf-8")
    root = _website_project_from_files(
        tmp_path,
        ["local-ai-coding-tool/index.html", "local-ai-coding-tool/styles.css"],
    )
    assert root == project
