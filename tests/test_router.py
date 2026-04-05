"""Router classification rules."""

from forge.agent.router import classify


def test_classify_defaults_to_search() -> None:
    assert classify("what files are in this project?") == "search"


def test_classify_returns_edit_code_for_write_task() -> None:
    assert classify("fix the FutureWarning in ast_parser.py") == "edit_code"


def test_classify_checks_write_keywords_first() -> None:
    assert classify("what should I change in this file?") == "edit_code"


def test_classify_treats_website_build_requests_as_edit_code() -> None:
    assert classify("build me a landing page in html and css") == "edit_code"
    assert classify("design a homepage for this app") == "edit_code"
