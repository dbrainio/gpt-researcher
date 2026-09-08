"""Private per-run budget client. Credentials never enter provider parameters.

This module does not retry paid operations or infer a zero cost from failure.
The application owns entitlement, prices, funding and receipt verification.
"""
from __future__ import annotations

from contextvars import ContextVar
import asyncio
import json
import logging
import os
import re
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ResearchBudgetError(RuntimeError):
    def __init__(self, code="budget_internal_error"):
        self.code = code if isinstance(code, str) and code in {
            "budget_exceeded", "budget_internal_error", "budget_invalid_transition",
            "budget_idempotency_conflict", "budget_reservation_not_found",
        } else "budget_internal_error"
        self.retryable = False
        super().__init__("Research budget exhausted" if self.code == "budget_exceeded" else "Research budget operation unavailable; do not repeat paid work")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer capability to another origin/path.
        return None


class BudgetCallback:
    def __init__(self, url: str):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/api/internal/budget/operations"):
            raise ResearchBudgetError()
        # HTTP is permitted only for explicitly configured local/cluster service
        # endpoints. Public callback traffic must use TLS.
        if parsed.scheme == "http" and not (
            parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            or parsed.hostname.endswith(".svc.cluster.local")
        ):
            raise ResearchBudgetError()
        self._url = url
        self._opener = build_opener(_NoRedirect())

    def __call__(self, credential: str, payload: dict) -> dict:
        request = Request(self._url, method="POST", data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": f"Bearer {credential}", "Content-Type": "application/json",
        })
        try:
            with self._opener.open(request, timeout=10) as response:
                data = response.read(16_385)
                if len(data) > 16_384:
                    raise ResearchBudgetError()
                result = json.loads(data)
                if not isinstance(result, dict):
                    raise ResearchBudgetError()
                return result
        except HTTPError as error:
            code = "budget_internal_error"
            try:
                data = error.read(16_385)
                result = json.loads(data) if len(data) <= 16_384 else {}
                if isinstance(result, dict):
                    code = result.get("code")
            except Exception:
                pass
            finally:
                error.close()
            raise ResearchBudgetError(code) from None
        except Exception:
            # urllib errors can carry response/request objects. Don't retain
            # their causes or server messages in logs, task errors or prompts.
            raise ResearchBudgetError() from None


def _credential(value):
    return isinstance(value, str) and len(value) <= 2048 and re.fullmatch(r"nbgt1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}", value) is not None


class ResearchBudget:
    def __init__(self, capability: str, mode: str, callback=None):
        if not _credential(capability) or mode not in {"shadow", "enforce"}:
            raise ResearchBudgetError()
        self._capability = capability
        self.mode = mode
        self._callback = callback if callback is not None else BudgetCallback(os.environ.get("NEVEL_BUDGET_URL", ""))
        self._counter = 0
        self._lock = Lock()
        self._clients = None
        self._closed = False
        self._failure = None

    def __repr__(self):
        return f"ResearchBudget(mode={self.mode!r})"

    def reserve_model(self, model: str, input_bytes: int, max_output_tokens: int):
        # One namespace across child researchers and concurrent executor threads.
        with self._lock:
            if self._closed or self._failure:
                raise ResearchBudgetError(self._failure or "budget_invalid_transition")
            step = self._counter
            self._counter += 1
        try:
            if (step > 511 or not isinstance(model, str) or not 1 <= len(model) <= 160
                    or type(input_bytes) is not int or not 0 <= input_bytes <= 4_194_304
                    or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 32_000):
                raise ResearchBudgetError()
            result = self._callback(self._capability, {
                "action": "reserve", "step": step, "provider": "openrouter",
                "modelId": model, "inputBytes": input_bytes, "maxOutputTokens": max_output_tokens,
            })
            if result == {"kind": "bypass"}:
                return None
            if (not isinstance(result, dict) or result.get("kind") != "tracked"
                    or not _credential(result.get("receipt")) or result.get("mode") not in {"shadow", "enforce"}
                    or type(result.get("maxOutputTokens")) is not int
                    or not 1 <= result["maxOutputTokens"] <= max_output_tokens):
                raise ResearchBudgetError()
            canonical = f"openai/{model}" if "/" not in model and re.match(r"^(gpt-|o[134]-)", model) else model
            if result.get("modelId") != canonical:
                raise ResearchBudgetError()
            return ResearchBudgetOperation(self._callback, result)
        except Exception as error:
            if self.mode == "shadow" and not (
                isinstance(error, ResearchBudgetError) and error.code in {"budget_exceeded", "budget_idempotency_conflict", "budget_invalid_transition"}
            ):
                logging.getLogger(__name__).warning("Shadow research budget admission unavailable")
                return None
            self.deny_new_calls(error.code if isinstance(error, ResearchBudgetError) else None)
            raise ResearchBudgetError(error.code if isinstance(error, ResearchBudgetError) else None) from None

    def deny_new_calls(self, code=None):
        with self._lock:
            self._failure = ResearchBudgetError(code).code

    def http_clients(self):
        # Reuse one connection pool pair per run, not one pair per model step.
        with self._lock:
            if self._closed:
                raise ResearchBudgetError("budget_invalid_transition")
            if self._clients is None:
                import httpx
                from .budget_http import ResearchBudgetTransport, ResearchBudgetSyncTransport
                self._clients = {
                    "http_client": httpx.Client(transport=ResearchBudgetSyncTransport(self), trust_env=False),
                    "http_async_client": httpx.AsyncClient(transport=ResearchBudgetTransport(self), trust_env=False),
                }
            return dict(self._clients)

    async def aclose(self):
        with self._lock:
            self._closed = True
            clients = self._clients
            self._clients = None
        if clients is not None:
            try:
                await clients["http_async_client"].aclose()
            finally:
                await asyncio.to_thread(clients["http_client"].close)


class ResearchBudgetOperation:
    def __init__(self, callback, admission):
        self._callback = callback
        self._receipt = admission["receipt"]
        self.mode = admission["mode"]
        self.model_id = admission["modelId"]
        self.max_output_tokens = admission["maxOutputTokens"]
        self._provider_id = None
        self._observed = False
        self._finalized = None

    def __repr__(self):
        return f"ResearchBudgetOperation(mode={self.mode!r}, model_id={self.model_id!r})"

    def _send(self, payload):
        try:
            if self._callback(self._receipt, payload) != {"kind": "acknowledged"}:
                raise ResearchBudgetError()
            return True
        except Exception as error:
            if self.mode == "shadow":
                logging.getLogger(__name__).warning("Shadow research budget settlement unavailable")
                return False
            raise ResearchBudgetError(error.code if isinstance(error, ResearchBudgetError) else None) from None

    def observe(self, provider_id: str):
        if not isinstance(provider_id, str) or not 1 <= len(provider_id) <= 256:
            raise ResearchBudgetError("budget_invalid_transition")
        if self._provider_id is not None and self._provider_id != provider_id:
            raise ResearchBudgetError("budget_idempotency_conflict")
        if self._provider_id == provider_id and self._observed:
            return
        self._provider_id = provider_id
        if self._send({"action": "observe", "providerUsageId": provider_id}):
            self._observed = True

    def finalize(self, cost_usd: str):
        # Decimal string from native JSON. Do not coerce None/NaN/bool to zero or
        # turn a token estimate into native provider cost.
        if not isinstance(cost_usd, str) or len(cost_usd) > 128 or not re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cost_usd):
            raise ResearchBudgetError("budget_invalid_transition")
        if self._finalized is not None:
            if self._finalized != cost_usd:
                raise ResearchBudgetError("budget_idempotency_conflict")
            return
        if self._send({"action": "finalize", "providerUsageId": self._provider_id, "providerCostUsd": cost_usd}):
            self._finalized = cost_usd


current_research_budget: ContextVar[ResearchBudget | None] = ContextVar("research_budget", default=None)
