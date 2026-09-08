"""Embedding budget protocol and native SDK integration, no paid network calls."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import httpx
from test_budget_http import module as transport, budget_module as budget, package, Chunks

CAP = "nbgt1.run." + "r" * 43
RECEIPT = "nbgt1.receipt." + "s" * 43
MODEL = "text-embedding-3-small"
URL = "https://api.openai.com/v1/embeddings"


def admission():
    return {"kind": "tracked_embedding", "receipt": RECEIPT, "mode": "enforce", "modelId": MODEL}


def result(tokens=2, model=MODEL):
    return {"object": "list", "model": model, "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


class EmbeddingBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {"budget_http_fixture": package, "budget_http_fixture.budget": budget,
                                              "budget_http_fixture.budget_http": transport})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def client(self, handler):
        callback = Mock(side_effect=lambda credential, payload: admission() if payload["action"] == "reserve_embedding" else {"kind": "acknowledged"})
        run = budget.ResearchBudget(CAP, "enforce", callback)
        native = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = httpx.AsyncClient(transport=transport.ResearchBudgetTransport(run, native), trust_env=False)
        self.addAsyncCleanup(client.aclose)
        return client, run, callback

    async def test_native_tokens_and_header_identity_are_settled_without_dollar_estimate(self):
        client, _, callback = self.client(lambda _: httpx.Response(200, json=result(), headers={"x-request-id": "req-native"}))
        await client.post(URL, json={"model": MODEL, "input": [[10, 20]]})
        self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_embedding", "observe", "finalize_embedding"])
        self.assertEqual(callback.call_args_list[0].args[1]["inputTokens"], 2)
        self.assertEqual(callback.call_args.args, (RECEIPT, {"action": "finalize_embedding", "modelId": MODEL, "inputTokens": 2, "providerUsageId": "req-native"}))
        self.assertNotIn("providerCostUsd", str(callback.call_args_list))

    async def test_unknown_or_mismatched_usage_stays_held_and_stops_new_calls(self):
        for body in [{"model": MODEL}, result(True), result(-1), result(model="other"), {**result(), "usage": {"prompt_tokens": 2, "total_tokens": 3}}]:
            with self.subTest(body=body):
                native = Mock(return_value=httpx.Response(200, json=body, headers={"x-request-id": "req-unknown"}))
                client, run, callback = self.client(native)
                with self.assertRaises(budget.ResearchBudgetError):
                    await client.post(URL, json={"model": MODEL, "input": "test"})
                self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_embedding", "observe"])
                with self.assertRaises(budget.ResearchBudgetError):
                    run.reserve_embedding(MODEL, 2)
                native.assert_called_once()

    async def test_header_id_survives_interrupted_response_and_explicit_zero_is_valid(self):
        client, _, callback = self.client(lambda _: httpx.Response(200, headers={"x-request-id": "req-interrupted"}, stream=Chunks([httpx.ReadError("interrupted")])) )
        with self.assertRaises(httpx.ReadError):
            await client.post(URL, json={"model": MODEL, "input": [1, 2]})
        self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_embedding", "observe"])
        zero, _, callback = self.client(lambda _: httpx.Response(200, json=result(0)))
        await zero.post(URL, json={"model": MODEL, "input": "test"})
        self.assertEqual(callback.call_args.args[1]["inputTokens"], 0)

    async def test_invalid_inputs_are_rejected_before_admission_or_provider(self):
        native = Mock()
        client, _, callback = self.client(native)
        for fields in [{"input": ""}, {"input": [True]}, {"input": [[-1]]}, {"input": [[1] * 8193]},
                       {"input": ["a"] * 17}, {"input": "a", "dimensions": 1537},
                       {"input": "a", "model": "other"}, {"input": "a", "extra_paid_option": True}]:
            with self.assertRaises(budget.ResearchBudgetError):
                await client.post(URL, json={"model": MODEL, **fields})
        callback.assert_not_called()
        native.assert_not_called()

    async def test_denial_reaches_no_native_provider_and_steps_share_model_namespace(self):
        native = Mock()
        client, run, callback = self.client(native)
        callback.side_effect = budget.ResearchBudgetError("budget_exceeded")
        with self.assertRaises(budget.ResearchBudgetError):
            await client.post(URL, json={"model": MODEL, "input": "test"})
        native.assert_not_called()
        callback = Mock(return_value={"kind": "bypass"})
        run = budget.ResearchBudget(CAP, "enforce", callback)
        run.reserve_model("gpt-4o-mini", 10, 100)
        run.reserve_embedding(MODEL, 2)
        self.assertEqual([call.args[1]["step"] for call in callback.call_args_list], [0, 1])

    async def test_receipt_kind_and_replayed_settlement_are_enforced_locally(self):
        callback = Mock(return_value={"kind": "acknowledged"})
        operation = budget.ResearchBudgetOperation(callback, admission())
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize("0")
        operation.finalize_embedding(MODEL, 2)
        operation.finalize_embedding(MODEL, 2)
        callback.assert_called_once()
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize_embedding(MODEL, 3)

    async def test_shadow_outage_keeps_native_body_but_explicit_denial_is_not_bypassed(self):
        seen = []
        client, run, callback = self.client(lambda req: (seen.append(json.loads(req.content)) or httpx.Response(200, json=result())))
        run.mode = "shadow"
        callback.side_effect = budget.ResearchBudgetError()
        original = {"model": MODEL, "input": "test", "dimensions": 512}
        with self.assertLogs(budget.__name__, level="WARNING"):
            await client.post(URL, json=original)
        self.assertEqual(seen, [original])
        callback.side_effect = budget.ResearchBudgetError("budget_exceeded")
        with self.assertRaises(budget.ResearchBudgetError):
            await client.post(URL, json=original)
        self.assertEqual(seen, [original])

    async def test_factory_rejects_unmetered_overrides_in_enforce_and_preserves_off(self):
        path = Path(__file__).parents[1] / "gpt_researcher/memory/embeddings.py"
        spec = importlib.util.spec_from_file_location("embedding_memory_fixture", path)
        memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(memory)
        constructor = Mock()
        from types import SimpleNamespace
        run = budget.ResearchBudget(CAP, "enforce", Mock())
        token = budget.current_research_budget.set(run)
        try:
            with patch.dict(sys.modules, {"gpt_researcher.utils.budget": budget, "langchain_openai": SimpleNamespace(OpenAIEmbeddings=constructor)}):
                for kwargs in [{"client": object()}, {"http_async_client": object()}, {"base_url": "https://other.test/v1"}]:
                    with self.assertRaises(budget.ResearchBudgetError):
                        memory.Memory("openai", MODEL, **kwargs)
                with self.assertRaises(budget.ResearchBudgetError):
                    memory.Memory("openai", "other-model")
                with self.assertRaises(budget.ResearchBudgetError):
                    memory.Memory("custom", MODEL)
                constructor.assert_not_called()
                run.mode = "shadow"
                with self.assertLogs(memory.__name__, level="WARNING"):
                    memory.Memory("openai", MODEL, base_url="https://other.test/v1", max_retries=3)
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 3)
                self.assertNotIn("http_client", constructor.call_args.kwargs)
                budget.current_research_budget.set(None)
                memory.Memory("openai", MODEL, max_retries=3)
                self.assertNotIn("chunk_size", constructor.call_args.kwargs)
        finally:
            budget.current_research_budget.reset(token)

    @unittest.skipUnless(importlib.util.find_spec("langchain_openai"), "Install researcher SDK dependencies for native integration")
    async def test_real_memory_factory_sync_async_and_sdk_no_retry(self):
        from langchain_openai import OpenAIEmbeddings
        path = Path(__file__).parents[1] / "gpt_researcher/memory/embeddings.py"
        spec = importlib.util.spec_from_file_location("embedding_memory_fixture", path)
        memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(memory)
        callback = Mock(side_effect=lambda credential, payload: admission() if payload["action"] == "reserve_embedding" else {"kind": "acknowledged"})
        run = budget.ResearchBudget(CAP, "enforce", callback)
        seen = []
        def native(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=result(), headers={"x-request-id": "req-sdk"})
        sync = httpx.Client(transport=transport.ResearchBudgetSyncTransport(run, httpx.Client(transport=httpx.MockTransport(native))), trust_env=False)
        asynchronous = httpx.AsyncClient(transport=transport.ResearchBudgetTransport(run, httpx.AsyncClient(transport=httpx.MockTransport(native))), trust_env=False)
        self.addCleanup(sync.close)
        self.addAsyncCleanup(asynchronous.aclose)
        clients = {"http_client": sync, "http_async_client": asynchronous}
        token = budget.current_research_budget.set(run)
        try:
            with patch.dict(sys.modules, {"gpt_researcher.utils.budget": budget}), patch.object(run, "http_clients", return_value=clients):
                model = memory.Memory("openai", MODEL, api_key="offline-fixture", check_embedding_ctx_length=False, max_retries=10).get_embeddings()
            self.assertIsInstance(model, OpenAIEmbeddings)
            self.assertEqual(model.chunk_size, 16)
            self.assertEqual(model.max_retries, 0)
            self.assertEqual(model.embed_query("test"), [1.0, 0.0])
            self.assertEqual(await model.aembed_query("test"), [1.0, 0.0])
            self.assertEqual(len(seen), 2)
            self.assertEqual(sum(call.args[1]["action"] == "finalize_embedding" for call in callback.call_args_list), 2)
            # Exercise the actual length-safe SDK branch with deterministic
            # local tokenization, avoiding tiktoken's first-run cache download.
            model.check_embedding_ctx_length = True
            with patch("tiktoken.encoding_for_model", return_value=Mock(encode_ordinary=Mock(return_value=[10, 20]))):
                self.assertEqual(model.embed_query("test"), [1.0, 0.0])
                self.assertEqual(await model.aembed_query("test"), [1.0, 0.0])
            self.assertEqual(seen[-1]["input"], [[10, 20]])
            self.assertEqual(len(seen), 4)
            callback.side_effect = budget.ResearchBudgetError("budget_exceeded")
            model.check_embedding_ctx_length = False
            with self.assertRaises(Exception):
                await model.aembed_query("denied")
            self.assertEqual(len(seen), 4)
        finally:
            budget.current_research_budget.reset(token)


if __name__ == "__main__":
    unittest.main()
