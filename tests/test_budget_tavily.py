"""Tavily paid boundaries, entirely in-memory HTTP responses and executors."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from test_budget_http import module as transport, budget_module as budget, package

ROOT = Path(__file__).parents[1] / "gpt_researcher"
CAP = "nbgt1.run." + "r" * 43
RECEIPT = "nbgt1.receipt." + "s" * 43
URL = "https://api.tavily.com/search"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admitted(payload):
    return {"kind": "tracked_tavily", "receipt": RECEIPT, "mode": "enforce", "endpoint": payload["endpoint"], "depth": payload["depth"]}


def response(credits=1):
    return {"request_id": "request-native", "usage": {"credits": credits}, "failed_results": [],
            "results": [{"url": "https://example.test", "content": "source", "raw_content": "source"}]}


class TavilyBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        modules = patch.dict(sys.modules, {"budget_http_fixture": package, "budget_http_fixture.budget": budget,
                                          "budget_http_fixture.budget_http": transport, "gpt_researcher.utils.budget": budget})
        modules.start()
        self.addCleanup(modules.stop)

    def setup_run(self, handler, asynchronous=False):
        callback = Mock(side_effect=lambda credential, payload: admitted(payload) if payload["action"] == "reserve_tavily" else {"kind": "acknowledged"})
        run = budget.ResearchBudget(CAP, "enforce", callback)
        if asynchronous:
            native = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            client = httpx.AsyncClient(transport=transport.ResearchBudgetTransport(run, native), trust_env=False)
            self.addAsyncCleanup(client.aclose)
        else:
            native = httpx.Client(transport=httpx.MockTransport(handler))
            client = httpx.Client(transport=transport.ResearchBudgetSyncTransport(run, native), trust_env=False)
            self.addCleanup(client.close)
        return client, run, callback

    async def test_search_entry_meters_before_mapping_and_requests_native_usage(self):
        seen = []
        client, run, callback = self.setup_run(lambda request: (seen.append(json.loads(request.content)) or httpx.Response(200, json=response())))
        search = load("tavily_search_fixture", "retrievers/tavily/tavily_search.py")
        token = budget.current_research_budget.set(run)
        try:
            with patch.object(run, "http_clients", return_value={"http_client": client}):
                result = search.TavilySearch("question", headers={"tavily_api_key": "offline-fixture"}).search()
            self.assertEqual(result, [{"href": "https://example.test", "body": "source"}])
            self.assertTrue(seen[0]["include_usage"])
            self.assertFalse(seen[0]["auto_parameters"])
            self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_tavily", "observe", "finalize_tavily"])
            self.assertEqual(callback.call_args.args[1], {"action": "finalize_tavily", "credits": "1", "providerUsageId": "request-native"})
            self.assertNotIn("offline-fixture", str(callback.call_args_list))
            self.assertNotIn("question", str(callback.call_args_list))
        finally:
            budget.current_research_budget.reset(token)

    async def test_async_extract_preserves_precise_fractional_and_explicit_zero_credits(self):
        for credits in ["0.00012500001", "0"]:
            raw = ('{"request_id":"native-extract","usage":{"credits":' + credits + '},"results":[],"failed_results":[]}').encode()
            client, _, callback = self.setup_run(lambda _: httpx.Response(200, content=raw), asynchronous=True)
            await client.post("https://api.tavily.com/extract", json={"urls": ["https://a.test"] * 6, "extract_depth": "advanced"})
            self.assertEqual(callback.call_args_list[0].args[1]["units"], 6)
            self.assertEqual(callback.call_args.args[1]["credits"], credits)

    async def test_search_does_not_swallow_budget_denial_and_off_keeps_legacy_requests(self):
        search = load("tavily_search_fixture", "retrievers/tavily/tavily_search.py")
        native = Mock()
        client, run, callback = self.setup_run(native)
        callback.side_effect = budget.ResearchBudgetError("budget_exceeded")
        token = budget.current_research_budget.set(run)
        try:
            with patch.object(run, "http_clients", return_value={"http_client": client}):
                with self.assertRaises(budget.ResearchBudgetError):
                    search.TavilySearch("q", headers={"tavily_api_key": "offline-fixture"}).search()
            native.assert_not_called()
        finally:
            budget.current_research_budget.reset(token)
        legacy = Mock(status_code=200)
        legacy.json.return_value = response()
        with patch.object(search.requests, "post", return_value=legacy) as post:
            self.assertEqual(len(search.TavilySearch("q", headers={"tavily_api_key": "offline-fixture"}).search()), 1)
        post.assert_called_once()

    async def test_paid_receipt_cannot_be_settled_as_model_or_embedding_and_replays_are_local(self):
        callback = Mock(return_value={"kind": "acknowledged"})
        operation = budget.ResearchBudgetOperation(callback, admitted({"endpoint": "search", "depth": "basic"}))
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize("0")
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize_embedding("text-embedding-3-small", 0)
        operation.finalize_tavily("1")
        operation.finalize_tavily("1")
        callback.assert_called_once()
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize_tavily("2")

    async def test_missing_usage_retains_native_identity_and_prevents_fallback_paid_call(self):
        native = Mock(return_value=httpx.Response(200, json={"request_id": "unknown-cost", "results": []}))
        client, run, callback = self.setup_run(native)
        with self.assertRaises(budget.ResearchBudgetError):
            client.post(URL, json={"query": "q"})
        self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_tavily", "observe"])
        with self.assertRaises(budget.ResearchBudgetError):
            run.reserve_tavily("search", "basic", 1)
        native.assert_called_once()

    async def test_closed_during_admission_releases_before_sync_and_async_native_send(self):
        for asynchronous in [False, True]:
            native = Mock()
            client, run, callback = self.setup_run(native, asynchronous=asynchronous)
            def admit(credential, payload):
                if payload["action"] == "reserve_tavily":
                    # Represents scope.aclose racing with its pending callback.
                    run._closed = True
                    return admitted(payload)
                return {"kind": "acknowledged"}
            callback.side_effect = admit
            with self.assertRaises(budget.ResearchBudgetError):
                if asynchronous:
                    await client.post(URL, json={"query": "q"})
                else:
                    client.post(URL, json={"query": "q"})
            native.assert_not_called()
            self.assertEqual([call.args[1]["action"] for call in callback.call_args_list], ["reserve_tavily", "release"])

    async def test_depth_auto_parameters_and_url_count_are_bounded_before_native(self):
        native = Mock()
        client, _, callback = self.setup_run(native)
        for url, body in [(URL, {"query": "q", "auto_parameters": True}), (URL, {"query": "q", "search_depth": "other"}),
                          ("https://api.tavily.com/extract", {"urls": ["https://a.test"] * 21}),
                          ("https://api.tavily.com/crawl", {"url": "https://a.test"})]:
            with self.assertRaises(budget.ResearchBudgetError):
                client.post(url, json=body)
        native.assert_not_called()
        callback.assert_not_called()

    async def test_extract_entry_settles_before_unpaid_html_failure(self):
        scraper_utils = SimpleNamespace(get_relevant_images=Mock(), extract_title=Mock())
        with patch.dict(sys.modules, {"bs4": SimpleNamespace(BeautifulSoup=Mock()), "scraper_fixture.utils": scraper_utils}):
            extract = load("scraper_fixture.tavily_extract.tavily_extract", "scraper/tavily_extract/tavily_extract.py")
        client, run, callback = self.setup_run(lambda _: httpx.Response(200, json=response(1)))
        token = budget.current_research_budget.set(run)
        try:
            session = Mock()
            session.get.side_effect = RuntimeError("unpaid HTML unavailable")
            with patch.dict(extract.os.environ, {"TAVILY_API_KEY": "offline-fixture"}), patch.object(run, "http_clients", return_value={"http_client": client}), patch("builtins.print"):
                instance = extract.TavilyExtract("https://example.test", session)
                self.assertIsNone(instance.tavily_client)
                self.assertEqual(instance.scrape(), ("", [], ""))
            self.assertEqual(callback.call_args.args[1]["action"], "finalize_tavily")
        finally:
            budget.current_research_budget.reset(token)

    async def test_actual_scraper_executor_preserves_run_context(self):
        stub = ModuleType("scraper_context_fixture")
        for name in ["ArxivScraper", "BeautifulSoupScraper", "PyMuPDFScraper", "WebBaseLoaderScraper", "BrowserScraper", "NoDriverScraper", "TavilyExtract", "FireCrawl"]:
            setattr(stub, name, Mock())
        with patch.dict(sys.modules, {"scraper_context_fixture": stub, "colorama": SimpleNamespace(Fore=Mock(), init=Mock()),
                                     "gpt_researcher.utils.workers": SimpleNamespace(WorkerPool=Mock())}):
            scraper = load("scraper_context_fixture.scraper", "scraper/scraper.py")
        run = budget.ResearchBudget(CAP, "enforce", Mock())
        seen = []
        class FakeScraper:
            def __init__(self, *args):
                pass
            def scrape(self):
                seen.append(budget.current_research_budget.get())
                return "x" * 120, [], "title"
        @asynccontextmanager
        async def throttle():
            yield
        token = budget.current_research_budget.set(run)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                instance = scraper.Scraper.__new__(scraper.Scraper)
                instance.worker_pool = SimpleNamespace(executor=executor, throttle=throttle)
                instance.logger = Mock()
                instance.get_scraper = Mock(return_value=FakeScraper)
                await instance.extract_data_from_url("https://example.test", Mock())
            self.assertEqual(seen, [run])
        finally:
            budget.current_research_budget.reset(token)


if __name__ == "__main__":
    unittest.main()
