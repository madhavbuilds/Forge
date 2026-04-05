"""Optional DuckDuckGo lite search (needs network when used)."""

from __future__ import annotations

import re

import httpx


def web_search(query: str, *, max_results: int = 5) -> dict:
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": "Forge/0.1 (local assistant)"},
            )
        text = r.text
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}

    links = re.findall(r'<a rel="nofollow" class="result-link" href="([^"]+)">([^<]+)</a>', text)
    out = [{"url": u, "title": t.strip()} for u, t in links[:max_results]]
    return {"ok": True, "query": query, "results": out}
