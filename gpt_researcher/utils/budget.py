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
            if error.code in {401, 403, 409}:
                code = "budget_invalid_transition"
            elif error.code == 429:
                code = "budget_exceeded"
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
        step = self._next_step()
        return self._reserve_model(step, model, input_bytes, max_output_tokens)

    def _next_step(self):
        with self._lock:
            if self._closed or self._failure:
                raise ResearchBudgetError(self._failure or "budget_invalid_transition")
            step = self._counter
            self._counter += 1
            return step

    def reserve_embedding(self, model: str, input_tokens: int):
        step = self._next_step()
        try:
            if (step > 511 or model != "text-embedding-3-small" or type(input_tokens) is not int
                    or not 1 <= input_tokens <= 300_000):
                raise ResearchBudgetError("budget_invalid_transition")
            result = self._callback(self._capability, {
                "action": "reserve_embedding", "step": step, "modelId": model, "inputTokens": input_tokens, "correlationVersion": 1,
            })
            if result == {"kind": "bypass"}:
                return None
            if (not isinstance(result, dict) or result.get("kind") != "tracked_embedding"
                    or not _credential(result.get("receipt")) or result.get("mode") != self.mode
                    or result.get("modelId") != model):
                raise ResearchBudgetError("budget_invalid_transition")
            return ResearchBudgetOperation(self._callback, result)
        except Exception as error:
            return self._admission_error(error)

    def _reserve_model(self, step, model, input_bytes, max_output_tokens):
        try:
            if (step > 511 or not isinstance(model, str) or not 1 <= len(model) <= 160
                    or type(input_bytes) is not int or not 0 <= input_bytes <= 4_194_304
                    or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 32_000):
                raise ResearchBudgetError()
            result = self._callback(self._capability, {
                "action": "reserve", "step": step, "provider": "openrouter", "correlationVersion": 1,
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
            return self._admission_error(error)

    def reserve_tavily(self, endpoint: str, depth: str, units: int):
        step = self._next_step()
        try:
            if (step > 511 or endpoint not in {"search", "extract"} or depth not in {"basic", "advanced"}
                    or type(units) is not int or not 1 <= units <= 20 or (endpoint == "search" and units != 1)):
                raise ResearchBudgetError("budget_invalid_transition")
            result = self._callback(self._capability, {
                "action": "reserve_tavily", "step": step, "endpoint": endpoint, "depth": depth, "units": units, "correlationVersion": 1,
            })
            if result == {"kind": "bypass"}:
                return None
            if (not isinstance(result, dict) or result.get("kind") != "tracked_tavily"
                    or not _credential(result.get("receipt")) or result.get("mode") != self.mode
                    or result.get("endpoint") != endpoint or result.get("depth") != depth):
                raise ResearchBudgetError("budget_invalid_transition")
            return ResearchBudgetOperation(self._callback, result)
        except Exception as error:
            return self._admission_error(error)

    def ensure_active(self):
        with self._lock:
            if self._closed or self._failure:
                raise ResearchBudgetError(self._failure or "budget_invalid_transition")

    def _admission_error(self, error):
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

    def raise_if_denied(self):
        with self._lock:
            if self._failure:
                raise ResearchBudgetError(self._failure)

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
        pair = admission.get("correlation")
        if (not isinstance(pair, dict) or set(pair) != {"budget_operation_id", "budget_reservation_id"}
                or not isinstance(pair.get("budget_operation_id"), str)
                or len(pair["budget_operation_id"]) > 256
                or re.fullmatch(r"deep-research:[A-Za-z0-9_:-]+:[0-9]+", pair["budget_operation_id"]) is None
                or not isinstance(pair.get("budget_reservation_id"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]{16}", pair["budget_reservation_id"]) is None):
            raise ResearchBudgetError("budget_invalid_transition")
        # These are diagnostics only. The bearer receipt stays private below.
        self.correlation = dict(pair)
        self._callback = callback
        self._receipt = admission["receipt"]
        self.mode = admission["mode"]
        self.model_id = admission.get("modelId")
        self.embedding = admission.get("kind") == "tracked_embedding"
        self.tavily = admission.get("kind") == "tracked_tavily"
        self.max_output_tokens = admission.get("maxOutputTokens")
        self._provider_id = None
        self._observed = False
        self._finalized = None
        self._started = False
        self._released = False

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

    def mark_started(self):
        if self._released:
            raise ResearchBudgetError("budget_invalid_transition")
        self._started = True

    def release_unstarted(self):
        # Called only by the transport that still owns admission. An exception
        # after native.send is ambiguous even if no provider ID was received.
        if self._started or self._provider_id is not None or self._finalized is not None:
            raise ResearchBudgetError("budget_invalid_transition")
        if not self._released and self._send({"action": "release", "reason": "provider_not_called"}):
            self._released = True

    def finalize(self, cost_usd: str):
        # Decimal string from native JSON. Do not coerce None/NaN/bool to zero or
        # turn a token estimate into native provider cost.
        if self.embedding or self.tavily or self._released or not isinstance(cost_usd, str) or len(cost_usd) > 128 or not re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cost_usd):
            raise ResearchBudgetError("budget_invalid_transition")
        if self._finalized is not None:
            if self._finalized != cost_usd:
                raise ResearchBudgetError("budget_idempotency_conflict")
            return
        if self._send({"action": "finalize", "providerUsageId": self._provider_id, "providerCostUsd": cost_usd}):
            self._finalized = cost_usd

    def finalize_embedding(self, model: str, input_tokens: int):
        if (not self.embedding or self._released or model != self.model_id
                or type(input_tokens) is not int or not 0 <= input_tokens <= 300_000):
            raise ResearchBudgetError("budget_invalid_transition")
        if self._finalized is not None:
            if self._finalized != input_tokens:
                raise ResearchBudgetError("budget_idempotency_conflict")
            return
        if self._send({"action": "finalize_embedding", "modelId": model,
                       "inputTokens": input_tokens, "providerUsageId": self._provider_id}):
            self._finalized = input_tokens

    def finalize_tavily(self, credits: str):
        if (not self.tavily or self._released or not isinstance(credits, str) or len(credits) > 64
                or not re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", credits)):
            raise ResearchBudgetError("budget_invalid_transition")
        if self._finalized is not None:
            if self._finalized != credits:
                raise ResearchBudgetError("budget_idempotency_conflict")
            return
        if self._send({"action": "finalize_tavily", "credits": credits, "providerUsageId": self._provider_id}):
            self._finalized = credits


current_research_budget: ContextVar[ResearchBudget | None] = ContextVar("research_budget", default=None)


def require_budget_coverage(component: str, selected: str, supported: tuple[str, ...]):
    """Fail closed on configuration drift, without changing off/B2B execution."""
    run = current_research_budget.get()
    if run is None:
        return
    run.ensure_active()
    if selected in supported:
        return
    if run.mode == "shadow":
        # Never log arbitrary selected values, MCP endpoints or credentials.
        logging.getLogger(__name__).warning("Shadow research configuration has an unmetered component")
        return
    run.deny_new_calls("budget_invalid_transition")
    raise ResearchBudgetError("budget_invalid_transition")


def find_budget_error(error):
    seen = set()
    for _ in range(8):
        if isinstance(error, ResearchBudgetError):
            return error
        if not isinstance(error, BaseException) or id(error) in seen:
            break
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return None
