"""Budget at the native OpenRouter HTTP boundary, including LangChain chains."""
from __future__ import annotations

import asyncio
from decimal import Decimal
import json
import logging
import re

import httpx

from .budget import ResearchBudgetError


class NativeEvidence:
    def __init__(self, operation, sse: bool):
        self.operation = operation
        self.sse = sse
        self.buffer = b""
        self.cost = None
        self.done = False

    def _event(self, data):
        if data == b"[DONE]":
            self.done = True
            if self.cost is not None:
                self.operation.finalize(self.cost)
            return
        value = json.loads(data, parse_float=Decimal)
        if not isinstance(value, dict):
            raise ResearchBudgetError()
        if self.operation.tavily is True:
            identity = value.get("request_id")
            if isinstance(identity, str) and len(identity) <= 256 and re.fullmatch(r"[A-Za-z0-9_-]+", identity):
                self.operation.observe(identity)
            usage = value.get("usage")
            credits = usage.get("credits") if isinstance(usage, dict) else None
            if type(credits) is int or isinstance(credits, Decimal):
                credits = str(credits)
            self.operation.finalize_tavily(credits)
            return
        if self.operation.embedding is True:
            usage = value.get("usage")
            if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                    or usage.get("prompt_tokens") != usage.get("total_tokens")
                    or type(usage.get("total_tokens")) is not int):
                raise ResearchBudgetError("budget_invalid_transition")
            self.operation.finalize_embedding(value.get("model"), usage["prompt_tokens"])
            return
        identity = value.get("id")
        if isinstance(identity, str) and re.fullmatch(r"gen-[A-Za-z0-9_-]+", identity) and len(identity) <= 256:
            self.operation.observe(identity)
        usage = value.get("usage")
        if isinstance(usage, dict) and "cost" in usage:
            cost = usage["cost"]
            if type(cost) is int or isinstance(cost, Decimal):
                cost = str(cost)
            if isinstance(cost, str) and re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cost) and len(cost) <= 128:
                self.cost = cost
            else:
                self.cost = None

    def feed(self, chunk: bytes):
        self.buffer += chunk
        if len(self.buffer) > 4_194_304:
            raise ResearchBudgetError()
        if self.sse:
            # Normalize complete lines, including CRLF split across byte chunks.
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    self._event(line[5:].strip())

    def finish(self):
        if not self.sse:
            self._event(self.buffer)
            if self.cost is not None:
                self.operation.finalize(self.cost)
        # A truncated SSE stream has no terminal proof. Its native ID remains
        # attached to the held reservation for the existing generation reconciler.


class _DecodedAsyncResponse(httpx.AsyncByteStream):
    def __init__(self, response):
        self.response = response

    async def __aiter__(self):
        async for chunk in self.response.aiter_bytes():
            yield chunk

    async def aclose(self):
        await self.response.aclose()


class _DecodedSyncResponse(httpx.SyncByteStream):
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        yield from self.response.iter_bytes()

    def close(self):
        self.response.close()


def _decoded_headers(response):
    headers = response.headers.copy()
    headers.pop("content-encoding", None)
    headers.pop("content-length", None)
    return headers


class _ObservedStream(httpx.AsyncByteStream):
    def __init__(self, stream, operation, sse, deny):
        self._stream = stream
        self._evidence = NativeEvidence(operation, sse)
        self._operation = operation
        self._deny = deny
        self._finished = False

    async def _record(self, action, *args):
        try:
            await asyncio.to_thread(action, *args)
        except Exception:
            if self._operation.mode == "enforce":
                self._deny()
                raise ResearchBudgetError() from None

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                await self._record(self._evidence.feed, chunk)
                yield chunk
            await self._record(self._evidence.finish)
            self._finished = True
        except BaseException:
            if self._operation.mode == "enforce":
                self._deny()
            raise
        finally:
            await self._stream.aclose()

    async def aclose(self):
        if not self._finished and not self._evidence.done and self._operation.mode == "enforce":
            self._deny()
        await self._stream.aclose()


def _prepare_request(budget, request, data):
    if (request.method == "POST" and request.url.scheme == "https" and request.url.host == "api.tavily.com"
            and request.url.port in {None, 443} and request.url.path in {"/search", "/extract"} and not request.url.query):
        from .budget_tavily import prepare_tavily_request
        return prepare_tavily_request(budget, request, data)
    if (request.method == "POST" and request.url.scheme == "https" and request.url.host == "api.openai.com"
            and request.url.port in {None, 443} and request.url.path == "/v1/embeddings" and not request.url.query):
        from .budget_embedding import prepare_embedding_request
        return prepare_embedding_request(budget, request, data)
    if (request.method != "POST" or request.url.scheme != "https" or request.url.host != "openrouter.ai"
            or request.url.port not in {None, 443} or request.url.path != "/api/v1/chat/completions"
            or request.url.query):
        raise ResearchBudgetError("budget_invalid_transition")
    try:
        body = json.loads(data)
        if not isinstance(body, dict) or len(data) > 4_000_000:
            raise ValueError()
        # Multimodal input and extra priced plugins need their own admission
        # dimensions. This researcher route currently sends text only.
        if (body.get("n", 1) != 1 or body.get("plugins") or body.get("models")
                or not isinstance(body.get("messages"), list)
                or any(not isinstance(m, dict) or not isinstance(m.get("content"), (str, type(None))) for m in body["messages"])):
            raise ValueError()
        output = body.get("max_completion_tokens") or body.get("max_tokens") or 4000
        operation = budget.reserve_model(body.get("model"), len(data) + 512, output)
    except ResearchBudgetError:
        raise
    except Exception:
        if budget.mode == "enforce":
            raise ResearchBudgetError() from None
        operation = None
    if operation is not None and operation.mode == "enforce":
        body["model"] = operation.model_id
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = operation.max_output_tokens
        body["n"] = 1
        body["usage"] = {"include": True}
        if body.get("stream"):
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
    if operation is not None:
        trace = body.get("trace")
        # Preserve event attribution, replace caller-supplied operation fields.
        # No capability/receipt, callback URL or subject payload is forwarded.
        body["trace"] = {**(trace if isinstance(trace, dict) else {}), **operation.correlation}
        headers = request.headers.copy()
        headers.pop("content-length", None)
        request = httpx.Request(request.method, request.url, headers=headers, json=body, extensions=request.extensions)
    return request, operation


class ResearchBudgetTransport(httpx.AsyncBaseTransport):
    def __init__(self, budget, native_client=None):
        self.budget = budget
        # Inner client retains env proxies; outer client MUST use trust_env=False
        # so proxy mounts cannot bypass the budget transport.
        self.native = native_client if native_client is not None else httpx.AsyncClient(follow_redirects=False)

    async def handle_async_request(self, request):
        data = await request.aread()
        admission = asyncio.create_task(asyncio.to_thread(_prepare_request, self.budget, request, data))
        try:
            request, operation = await asyncio.shield(admission)
        except asyncio.CancelledError:
            # to_thread keeps running after caller cancellation. Retain ownership
            # until the bounded callback returns; never abandon a late receipt.
            await _drain_cancelled_admission(admission)
            raise
        # No refunds or retries for any exception/status after this point.
        try:
            self.budget.ensure_active()
        except Exception:
            await _drain_cancelled_admission(admission)
            raise
        if operation is not None:
            operation.mark_started()
        try:
            response = await self.native.send(request, stream=True, follow_redirects=False)
        except BaseException:
            if operation is not None and operation.mode == "enforce":
                self.budget.deny_new_calls()
            raise
        if operation is not None and operation.mode == "enforce" and response.status_code >= 300:
            self.budget.deny_new_calls()
        if operation is None:
            return response
        if operation.embedding is True:
            try:
                await asyncio.to_thread(_observe_embedding_response, operation, response)
            except BaseException:
                self.budget.deny_new_calls()
                await response.aclose()
                raise
        return httpx.Response(response.status_code, headers=_decoded_headers(response), extensions=response.extensions,
                              stream=_ObservedStream(_DecodedAsyncResponse(response), operation, operation.embedding is not True and operation.tavily is not True and "text/event-stream" in response.headers.get("content-type", ""), self.budget.deny_new_calls))

    async def aclose(self):
        await self.native.aclose()


async def _drain_cancelled_admission(admission):
    async def release():
        try:
            _, operation = await admission
            if operation is not None:
                await asyncio.to_thread(operation.release_unstarted)
        except Exception:
            # Unknown admission or failed release stays held. Preserve the
            # original cancellation and never include credentials/payloads.
            logging.getLogger(__name__).warning("Cancelled research admission cleanup unavailable")

    cleanup = asyncio.create_task(release())
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Repeated cancellation must not orphan the same receipt either.
            continue
    cleanup.result()


def _observe_embedding_response(operation, response):
    identity = response.headers.get("x-request-id")
    if identity and len(identity) <= 256 and re.fullmatch(r"[A-Za-z0-9_-]+", identity):
        operation.observe(identity)


def _start_native(budget, operation):
    try:
        budget.ensure_active()
    except Exception:
        if operation is not None:
            operation.release_unstarted()
        raise
    if operation is not None:
        operation.mark_started()


class _ObservedSyncStream(httpx.SyncByteStream):
    def __init__(self, stream, operation, sse, deny):
        self._stream = stream
        self._evidence = NativeEvidence(operation, sse)
        self._operation = operation
        self._deny = deny
        self._finished = False

    def _record(self, action, *args):
        try:
            action(*args)
        except Exception:
            if self._operation.mode == "enforce":
                self._deny()
                raise ResearchBudgetError() from None

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._record(self._evidence.feed, chunk)
                yield chunk
            self._record(self._evidence.finish)
            self._finished = True
        except BaseException:
            if self._operation.mode == "enforce":
                self._deny()
            raise
        finally:
            self._stream.close()

    def close(self):
        if not self._finished and not self._evidence.done and self._operation.mode == "enforce":
            self._deny()
        self._stream.close()


class ResearchBudgetSyncTransport(httpx.BaseTransport):
    def __init__(self, budget, native_client=None):
        self.budget = budget
        self.native = native_client if native_client is not None else httpx.Client(follow_redirects=False)

    def handle_request(self, request):
        request, operation = _prepare_request(self.budget, request, request.read())
        _start_native(self.budget, operation)
        try:
            response = self.native.send(request, stream=True, follow_redirects=False)
        except Exception:
            if operation is not None and operation.mode == "enforce":
                self.budget.deny_new_calls()
            raise
        if operation is not None and operation.mode == "enforce" and response.status_code >= 300:
            self.budget.deny_new_calls()
        if operation is None:
            return response
        if operation.embedding is True:
            try:
                _observe_embedding_response(operation, response)
            except BaseException:
                self.budget.deny_new_calls()
                response.close()
                raise
        return httpx.Response(response.status_code, headers=_decoded_headers(response), extensions=response.extensions,
                              stream=_ObservedSyncStream(_DecodedSyncResponse(response), operation, operation.embedding is not True and operation.tavily is not True and "text/event-stream" in response.headers.get("content-type", ""), self.budget.deny_new_calls))

    def close(self):
        self.native.close()
