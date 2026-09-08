"""Real HTTPX transport API, entirely in-memory provider/callback responses."""
import importlib.util
import asyncio
import json
import gzip
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from threading import Event
import unittest
from unittest.mock import Mock, patch

import httpx
from test_stream_usage_evidence import module as provider_module

ROOT = Path(__file__).parents[1] / "gpt_researcher/utils"
package = ModuleType("budget_http_fixture")
package.__path__ = [str(ROOT)]
with patch.dict(sys.modules, {"budget_http_fixture": package}):
    spec = importlib.util.spec_from_file_location("budget_http_fixture.budget_http", ROOT / "budget_http.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    budget_module = sys.modules["budget_http_fixture.budget"]

URL = "https://openrouter.ai/api/v1/chat/completions"


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self):
        self.closed = True


class BudgetHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_admission_releases_late_receipt_without_native_call(self):
        entered, resume = Event(), Event()
        native = Mock()
        client, budget, operation = self.setup_client(native)
        def reserve(*args):
            entered.set()
            if not resume.wait(5):
                raise AssertionError("test admission was not resumed")
            return operation
        budget.reserve_model.side_effect = reserve
        pending = asyncio.create_task(client.post(URL, json={"model": "gpt-4o-mini", "messages": []}))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
        finally:
            resume.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        native.assert_not_called()
        operation.mark_started.assert_not_called()
        operation.release_unstarted.assert_called_once()

    async def test_cancellation_after_native_send_never_releases_reservation(self):
        entered = asyncio.Event()
        async def native(request):
            entered.set()
            await asyncio.Future()
        client, budget, operation = self.setup_client(native)
        pending = asyncio.create_task(client.post(URL, json={"model": "gpt-4o-mini", "messages": []}))
        await asyncio.wait_for(entered.wait(), 5)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        operation.mark_started.assert_called_once()
        operation.release_unstarted.assert_not_called()
        budget.deny_new_calls.assert_called_once()

    async def test_cancelled_admission_cleanup_failure_preserves_cancellation(self):
        admission = asyncio.create_task(asyncio.sleep(0, result=(None, Mock())))
        await admission
        operation = admission.result()[1]
        operation.release_unstarted.side_effect = RuntimeError("private receipt")
        with self.assertLogs(module.__name__, level="WARNING") as logs:
            await module._drain_cancelled_admission(admission)
        self.assertNotIn("private receipt", str(logs.output))
        operation.release_unstarted.assert_called_once()

    async def test_compressed_native_evidence_is_decoded_once(self):
        raw = b'{"id":"gen-compressed","usage":{"cost":0.2}}'
        client, _, operation = self.setup_client(lambda _: httpx.Response(200, headers={"content-type": "application/json", "content-encoding": "gzip"}, stream=Chunks([gzip.compress(raw)])))
        response = await client.post(URL, json={"model": "gpt-4o-mini", "messages": []})
        self.assertEqual(response.content, raw)
        operation.observe.assert_called_once_with("gen-compressed")
        operation.finalize.assert_called_once_with("0.2")
    async def test_run_pools_are_reused_closed_and_cannot_admit_after_cancellation(self):
        with patch.dict(sys.modules, {"budget_http_fixture": package, "budget_http_fixture.budget_http": module}):
            run = budget_module.ResearchBudget("nbgt1.fixture." + "s" * 43, "enforce", Mock())
            clients = run.http_clients()
            self.assertIs(clients["http_client"], run.http_clients()["http_client"])
            await run.aclose()
            await run.aclose()
            self.assertTrue(clients["http_client"].is_closed)
            self.assertTrue(clients["http_async_client"].is_closed)
            with self.assertRaises(module.ResearchBudgetError):
                run.reserve_model("gpt-4o-mini", 200, 200)
    async def test_factory_installs_both_scoped_clients_and_disables_sdk_retry(self):
        scope = Mock(mode="enforce")
        scope.http_clients.return_value = {"http_client": object(), "http_async_client": object()}
        constructor = Mock()
        fake_budget = SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: scope), ResearchBudgetError=module.ResearchBudgetError)
        with patch.dict(sys.modules, {
            "gpt_researcher.utils.budget": fake_budget,
            "langchain_openai": SimpleNamespace(ChatOpenAI=constructor),
            "langchain_core.rate_limiters": SimpleNamespace(InMemoryRateLimiter=Mock()),
        }), patch.object(provider_module, "_check_pkg"), patch.dict(provider_module.os.environ, {"OPENROUTER_API_KEY": "offline-fixture"}):
            provider_module.GenericLLMProvider.from_provider("openrouter", model="gpt-4o-mini", max_retries=10)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertIs(constructor.call_args.kwargs["http_client"], scope.http_clients.return_value["http_client"])
        self.assertIs(constructor.call_args.kwargs["http_async_client"], scope.http_clients.return_value["http_async_client"])
        self.assertNotIn("nevel_budget", constructor.call_args.kwargs)

    async def test_factory_rejects_unpriced_providers_before_loading_sdk(self):
        scope = Mock(mode="enforce")
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: scope), ResearchBudgetError=module.ResearchBudgetError)}), patch.object(provider_module, "_check_pkg") as check:
            with self.assertRaises(module.ResearchBudgetError):
                provider_module.GenericLLMProvider.from_provider("openai", model="other-model")
        check.assert_not_called()

    async def test_no_budget_context_preserves_native_sdk_policy(self):
        constructor = Mock()
        with patch.dict(sys.modules, {
            "gpt_researcher.utils.budget": SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: None), ResearchBudgetError=module.ResearchBudgetError),
            "langchain_openai": SimpleNamespace(ChatOpenAI=constructor),
        }), patch.object(provider_module, "_check_pkg"):
            provider_module.GenericLLMProvider.from_provider("openai", model="gpt-4o-mini", max_retries=3)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 3)
        self.assertNotIn("http_async_client", constructor.call_args.kwargs)

    def setup_client(self, handler, mode="enforce"):
        operation = Mock(mode=mode, model_id="openai/gpt-4o-mini", max_output_tokens=200)
        budget = Mock(mode=mode)
        budget.reserve_model.return_value = operation
        native = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = httpx.AsyncClient(transport=module.ResearchBudgetTransport(budget, native), trust_env=False)
        self.addAsyncCleanup(client.aclose)
        return client, budget, operation

    async def test_native_json_is_bounded_and_accounted_before_sdk_returns(self):
        seen = []
        def native(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=b'{"id":"gen-native","usage":{"cost":0.000000123456789}}', headers={"content-type": "application/json"})
        client, budget, operation = self.setup_client(native)
        await client.post(URL, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 900})
        self.assertEqual(seen[0]["max_tokens"], 200)
        self.assertEqual(seen[0]["model"], "openai/gpt-4o-mini")
        self.assertEqual(seen[0]["usage"], {"include": True})
        budget.reserve_model.assert_called_once()
        operation.observe.assert_called_once_with("gen-native")
        operation.finalize.assert_called_once_with("1.23456789E-7")

    async def test_denied_admission_never_reaches_native(self):
        native = Mock()
        client, budget, _ = self.setup_client(native)
        budget.reserve_model.side_effect = module.ResearchBudgetError("budget_exceeded")
        with self.assertRaises(module.ResearchBudgetError):
            await client.post(URL, json={"model": "gpt-4o-mini", "messages": []})
        native.assert_not_called()

    async def test_sse_ids_and_final_cost_are_recorded_before_delivery(self):
        chunks = Chunks([b'data: {"id":"gen-native"}\r', b'\n\ndata: {"usage":{"cost":0}}\n\ndata: [DONE]\n\n'])
        client, _, operation = self.setup_client(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=chunks))
        async with client.stream("POST", URL, json={"model": "gpt-4o-mini", "messages": [], "stream": True}) as response:
            async for value in response.aiter_raw():
                if b"[DONE]" in value:
                    operation.finalize.assert_called_once_with("0")
        operation.observe.assert_called_once_with("gen-native")
        self.assertTrue(chunks.closed)

    async def test_interrupted_native_stream_retains_id_without_zero_or_refund(self):
        chunks = Chunks([b'data: {"id":"gen-native"}\n\n', httpx.ReadError("interrupted")])
        client, budget, operation = self.setup_client(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=chunks))
        with self.assertRaises(httpx.ReadError):
            await client.post(URL, json={"model": "gpt-4o-mini", "messages": [], "stream": True})
        operation.observe.assert_called_once_with("gen-native")
        operation.finalize.assert_not_called()
        self.assertTrue(chunks.closed)
        budget.deny_new_calls.assert_called()

    async def test_unknown_cost_and_non_native_run_ids_are_not_billing_evidence(self):
        client, _, operation = self.setup_client(lambda _: httpx.Response(200, json={"id": "run-langchain", "usage": {"prompt_tokens": 10}}))
        await client.post(URL, json={"model": "gpt-4o-mini", "messages": []})
        operation.observe.assert_not_called()
        operation.finalize.assert_not_called()

    async def test_bypass_preserves_native_body_without_provider_constraints(self):
        seen = []
        client, budget, operation = self.setup_client(lambda request: (seen.append(json.loads(request.content)) or httpx.Response(200, json={"usage": {"cost": 1}})))
        budget.reserve_model.return_value = None
        original = {"model": "gpt-4o-mini", "messages": [], "max_tokens": 700}
        await client.post(URL, json=original)
        self.assertEqual(seen, [original])
        operation.finalize.assert_not_called()

    async def test_redirect_or_wrong_endpoint_cannot_start_an_unscoped_request(self):
        native = Mock(return_value=httpx.Response(302, headers={"location": "https://evil.test"}))
        client, _, _ = self.setup_client(native)
        with self.assertRaises(module.ResearchBudgetError):
            await client.post("https://evil.test/api/v1/chat/completions", json={})
        native.assert_not_called()

    async def test_unpriced_multimodal_input_is_rejected_before_native(self):
        native = Mock()
        client, budget, _ = self.setup_client(native)
        with self.assertRaises(module.ResearchBudgetError):
            await client.post(URL, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image"}}]}]})
        native.assert_not_called()
        budget.reserve_model.assert_not_called()

    async def test_truncated_sse_cost_is_not_proof_of_terminal_billing(self):
        chunks = Chunks([b'data: {"id":"gen-native","usage":{"cost":0.2}}\n\n'])
        client, _, operation = self.setup_client(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=chunks))
        await client.post(URL, json={"model": "gpt-4o-mini", "messages": [], "stream": True})
        operation.observe.assert_called_once_with("gen-native")
        operation.finalize.assert_not_called()


class SyncBudgetHTTPTests(unittest.TestCase):
    def test_sync_chain_uses_the_same_native_reservation_boundary(self):
        operation = Mock(mode="enforce", model_id="openai/gpt-4o-mini", max_output_tokens=200)
        budget = Mock(mode="enforce")
        budget.reserve_model.return_value = operation
        calls = []
        def native(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "gen-sync", "usage": {"cost": 0}})
        inner = httpx.Client(transport=httpx.MockTransport(native))
        with httpx.Client(transport=module.ResearchBudgetSyncTransport(budget, inner), trust_env=False) as client:
            client.post(URL, json={"model": "gpt-4o-mini", "messages": [], "max_tokens": 500})
        self.assertEqual(calls[0]["max_tokens"], 200)
        operation.observe.assert_called_once_with("gen-sync")
        operation.finalize.assert_called_once_with("0")
        self.assertTrue(inner.is_closed)


if __name__ == "__main__":
    unittest.main()
