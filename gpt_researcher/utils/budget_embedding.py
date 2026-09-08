"""Native embedding input bounds; no text, token arrays or vectors in callbacks."""
import json

from .budget import ResearchBudgetError


def prepare_embedding_request(budget, request, data):
    try:
        body = json.loads(data)
        if (len(data) > 4_000_000 or not isinstance(body, dict)
                or set(body) - {"model", "input", "dimensions", "encoding_format", "user"}
                or body.get("model") != "text-embedding-3-small"
                or body.get("encoding_format", "float") not in {"float", "base64"}):
            raise ValueError()
        dimensions = body.get("dimensions", 1536)
        if type(dimensions) is not int or not 1 <= dimensions <= 1536:
            raise ValueError()
        value = body.get("input")
        if isinstance(value, str):
            inputs = [value]
        elif isinstance(value, list) and value and all(type(token) is int for token in value):
            inputs = [value]
        else:
            inputs = value
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 16:
            raise ValueError()
        tokens = 0
        for item in inputs:
            if isinstance(item, str) and item:
                # UTF-8 bytes conservatively bound tokens for string inputs.
                # LangChain's normal length-safe path sends exact token arrays.
                tokens += len(item.encode("utf-8"))
            elif (isinstance(item, list) and 1 <= len(item) <= 8192
                  and all(type(token) is int and token >= 0 for token in item)):
                tokens += len(item)
            else:
                raise ValueError()
        if tokens > 300_000:
            raise ValueError()
    except Exception:
        if budget.mode == "enforce":
            raise ResearchBudgetError("budget_invalid_transition") from None
        return request, None
    return request, budget.reserve_embedding(body["model"], tokens)
