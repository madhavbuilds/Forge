"""GUI studio helpers."""

from forge.gui.server import PendingWrite, StudioSession, _build_diff


def test_build_diff_contains_headers_and_changed_lines() -> None:
    diff = _build_diff("app.py", "value = 1\n", "value = 2\n")
    assert "app.py (before)" in diff
    assert "app.py (after)" in diff
    assert "-value = 1" in diff
    assert "+value = 2" in diff


def test_studio_session_snapshot_includes_pending_write() -> None:
    session = StudioSession(
        id="abc123",
        task="fix app.py",
        repo="/tmp/project",
        auto_apply=False,
        intent="edit_code",
    )
    session.begin_write_confirmation(
        PendingWrite(
            path="app.py",
            diff="--- before\n+++ after",
            before_preview="value = 1",
            after_preview="value = 2",
        )
    )
    snapshot = session.snapshot()
    assert snapshot["waiting_for_write"] is True
    assert snapshot["pending_write"]["path"] == "app.py"
    assert snapshot["pending_write"]["diff"] == "--- before\n+++ after"
