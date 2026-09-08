"""Restricted native Tavily parameters and per-call admission."""
import json

import httpx

from .budget import ResearchBudgetError


def prepare_tavily_request(budget, request, data):
    endpoint = request.url.path.removeprefix("/")
    try:
        body = json.loads(data)
        if not isinstance(body, dict) or len(data) > 262_144:
            raise ValueError()
        common = {"api_key", "include_usage"}
        if endpoint == "search":
            allowed = common | {"query", "search_depth", "topic", "days", "include_answer", "include_raw_content",
                                "max_results", "include_domains", "exclude_domains", "include_images", "use_cache", "auto_parameters"}
            depth = body.get("search_depth", "basic")
            if (not isinstance(body.get("query"), str) or not body["query"]
                    or body.get("auto_parameters", False) is not False):
                raise ValueError()
            units = 1
        else:
            allowed = common | {"urls", "extract_depth", "include_images", "include_favicon", "format", "timeout"}
            depth = body.get("extract_depth", "basic")
            urls = body.get("urls")
            urls = [urls] if isinstance(urls, str) else urls
            if not isinstance(urls, list) or not 1 <= len(urls) <= 20 or any(not isinstance(url, str) or not url for url in urls):
                raise ValueError()
            units = len(urls)
        if set(body) - allowed or depth not in {"basic", "advanced"}:
            raise ValueError()
    except Exception:
        if budget.mode == "enforce":
            raise ResearchBudgetError("budget_invalid_transition") from None
        return request, None
    # Build the controlled native request BEFORE taking ownership of a reserve.
    body["include_usage"] = True
    body["search_depth" if endpoint == "search" else "extract_depth"] = depth
    if endpoint == "search":
        body["auto_parameters"] = False
    headers = request.headers.copy()
    headers.pop("content-length", None)
    prepared = httpx.Request(request.method, request.url, headers=headers, json=body, extensions=request.extensions)
    return prepared, budget.reserve_tavily(endpoint, depth, units)
