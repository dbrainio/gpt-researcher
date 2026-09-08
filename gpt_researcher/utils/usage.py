"""Provider usage evidence. Missing/invalid cost is not evidence of free work."""

from __future__ import annotations

import math
from uuid import uuid4
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(*values: Any) -> float | None:
    # Preserve explicit zero; reject an invalid present field instead of falling
    # through to an unrelated alias that happens to look valid.
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            number = float(value)
        except (ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None
    return None


def _tokens(*values: Any) -> int | None:
    number = _number(*values)
    return int(number) if number is not None and number.is_integer() and number <= 2**53 - 1 else None


def extract_usage_report(message: Any, fallback_model: str | None = None) -> dict[str, Any] | None:
    usage_metadata = _mapping(getattr(message, "usage_metadata", None))
    response_metadata = _mapping(getattr(message, "response_metadata", None))
    token_usage = (
        _mapping(response_metadata.get("token_usage"))
        or _mapping(response_metadata.get("usage"))
        or _mapping(response_metadata.get("usage_metadata"))
    )
    usage = {**token_usage, **usage_metadata}
    prompt_tokens = _tokens(usage.get("input_tokens"), usage.get("prompt_tokens"), usage.get("tokens_in"))
    completion_tokens = _tokens(usage.get("output_tokens"), usage.get("completion_tokens"), usage.get("tokens_out"))
    total_tokens = _tokens(usage.get("total_tokens"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    cost = _number(usage.get("cost"), usage.get("total_cost"), response_metadata.get("cost"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None and cost is None:
        return None
    model = response_metadata.get("model_name") or response_metadata.get("model") or usage.get("model") or fallback_model
    return {
        "model": model,
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or 0,
        "cost": cost,
        "cost_is_complete": cost is not None,
        "cost_source": "provider" if cost is not None else "unknown",
    }


def empty_research_usage() -> dict[str, Any]:
    return {
        "model_id": None, "models": [], "tokens_in": 0, "tokens_out": 0,
        "total_tokens": 0, "cost": 0.0, "cost_is_complete": True,
        "cost_source": "none",
        "scope_id": uuid4().hex, "scope_revision": 0, "included_scopes": {},
    }


def accumulate_research_usage(current: dict[str, Any], incoming: float | dict) -> dict[str, Any]:
    """Aggregate disjoint call/child results, keeping known subtotal and quality.

    Numeric legacy callbacks are estimates, even when they happen to be zero.
    Unknown native cost does not become a confirmed zero as totals are rolled up.
    Callers must not feed the same cumulative child snapshot twice.
    """
    result = {**current, "models": list(current.get("models", [])), "included_scopes": dict(current.get("included_scopes", {}))}
    result["scope_revision"] = current.get("scope_revision", 0) + 1
    if isinstance(incoming, dict):
        cost = _number(incoming.get("cost"))
        source = incoming.get("cost_source", "unknown")
        if not isinstance(source, str) or source not in {"provider", "estimated", "mixed", "unknown", "none"}:
            source = "unknown"
        complete = incoming.get("cost_is_complete") is True and cost is not None and source in {"provider", "none"}
        if cost is None:
            source = "unknown"
        prompt = _tokens(incoming.get("prompt_tokens"), incoming.get("tokens_in")) or 0
        completion = _tokens(incoming.get("completion_tokens"), incoming.get("tokens_out")) or 0
        total = _tokens(incoming.get("total_tokens"))
        result["tokens_in"] += prompt
        result["tokens_out"] += completion
        result["total_tokens"] += total if total is not None else prompt + completion
        model = incoming.get("model") or incoming.get("model_id")
        models = incoming.get("models", [])
        if not isinstance(models, list):
            models = []
        for item in [*models, model]:
            if isinstance(item, str) and item and item not in result["models"]:
                result["models"].append(item)
        if isinstance(model, str) and model:
            result["model_id"] = model
        scope = incoming.get("scope_id")
        revision = _tokens(incoming.get("scope_revision"))
        if isinstance(scope, str) and scope and revision is not None:
            result["included_scopes"][scope] = revision
            included = incoming.get("included_scopes")
            if isinstance(included, dict):
                for child_scope, child_revision in included.items():
                    if isinstance(child_scope, str) and _tokens(child_revision) is not None:
                        result["included_scopes"][child_scope] = child_revision
    else:
        cost = _number(incoming)
        if not isinstance(incoming, (int, float)) or isinstance(incoming, bool) or cost is None:
            raise ValueError("Estimated cost must be a finite nonnegative number")
        source, complete = "estimated", False
    subtotal = result["cost"] + (cost if cost is not None else 0.0)
    if not math.isfinite(subtotal):
        raise ValueError("Research cost subtotal overflow")
    result["cost"] = subtotal
    result["cost_is_complete"] = current.get("cost_is_complete") is True and complete
    previous = current.get("cost_source", "unknown")
    if previous == "unknown" or source == "unknown":
        result["cost_source"] = "unknown"
    elif previous == "none":
        result["cost_source"] = source
    elif source == "none" or source == previous:
        result["cost_source"] = previous
    else:
        result["cost_source"] = "mixed"
    return result
