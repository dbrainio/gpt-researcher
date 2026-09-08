"""Exercise the real provider stream loop offline with inert optional imports."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).parents[1]
usage_spec = importlib.util.spec_from_file_location("usage_fixture", ROOT / "gpt_researcher/utils/usage.py")
usage = importlib.util.module_from_spec(usage_spec)
usage_spec.loader.exec_module(usage)
colorama = ModuleType("colorama")
colorama.Fore = SimpleNamespace(GREEN="", YELLOW="", RED="")
colorama.Style = SimpleNamespace(RESET_ALL="")
colorama.init = lambda **kwargs: None
spec = importlib.util.spec_from_file_location("stream_fixture", ROOT / "gpt_researcher/llm_provider/generic/base.py")
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"aiofiles": ModuleType("aiofiles"), "colorama": colorama, "gpt_researcher.utils.usage": usage}):
    spec.loader.exec_module(module)


def chunk(content="", cost=None, tokens=None):
    values = {}
    if cost is not None:
        values["cost"] = cost
    if tokens is not None:
        values["input_tokens"] = tokens
    return SimpleNamespace(content=content, usage_metadata=values, response_metadata={})


class FakeLLM:
    def __init__(self, events):
        self.events = events
        self.calls = 0
        self.closed = False

    async def astream(self, messages, **kwargs):
        self.calls += 1
        try:
            for event in self.events:
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            self.closed = True


class StreamUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_reports_last_cumulative_snapshot_once(self):
        llm = FakeLLM([chunk("first", 0.1), chunk(" last", 0.3)])
        callback = Mock()
        result = await module.GenericLLMProvider(llm, verbose=False).stream_response([], cost_callback=callback)
        self.assertEqual(result, "first last")
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[0]["cost"], 0.3)
        self.assertEqual(llm.calls, 1)

    async def test_provider_failure_preserves_observed_cost_without_retry(self):
        llm = FakeLLM([chunk("partial", 0.2), RuntimeError("native stream failed")])
        callback = Mock()
        with self.assertRaisesRegex(RuntimeError, "native stream failed"):
            await module.GenericLLMProvider(llm, verbose=False).stream_response([], cost_callback=callback)
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[0]["cost"], 0.2)
        self.assertEqual(llm.calls, 1)

    async def test_websocket_failure_preserves_cost_from_the_same_chunk(self):
        class Socket:
            async def send_json(self, payload):
                raise RuntimeError("delivery failed")
        for text in ["paragraph\n", "final paragraph"]:
            with self.subTest(text=text):
                callback = Mock()
                llm = FakeLLM([chunk(text, 0)])
                provider = module.GenericLLMProvider(llm, verbose=False)
                with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                    await provider.stream_response([], Socket(), cost_callback=callback)
                callback.assert_called_once()
                self.assertEqual(callback.call_args.args[0]["cost"], 0)
                self.assertTrue(callback.call_args.args[0]["cost_is_complete"])
                self.assertTrue(llm.closed)

    async def test_cancellation_preserves_evidence_and_remains_cancellation(self):
        callback = Mock()
        provider = module.GenericLLMProvider(FakeLLM([chunk("partial", 0.2), asyncio.CancelledError()]), verbose=False)
        with self.assertRaises(asyncio.CancelledError):
            await provider.stream_response([], cost_callback=callback)
        callback.assert_called_once()

    async def test_missing_usage_is_never_reported_as_zero(self):
        callback = Mock()
        provider = module.GenericLLMProvider(FakeLLM([chunk("partial"), RuntimeError("failed")]), verbose=False)
        with self.assertRaises(RuntimeError):
            await provider.stream_response([], cost_callback=callback)
        callback.assert_not_called()

    async def test_unknown_cost_remains_unknown_on_failure(self):
        callback = Mock()
        provider = module.GenericLLMProvider(FakeLLM([chunk("partial", tokens=4), RuntimeError("failed")]), verbose=False)
        with self.assertRaises(RuntimeError):
            await provider.stream_response([], cost_callback=callback)
        callback.assert_called_once()
        self.assertIsNone(callback.call_args.args[0]["cost"])
        self.assertFalse(callback.call_args.args[0]["cost_is_complete"])

    async def test_failed_cleanup_callback_does_not_mask_cancellation(self):
        callback = Mock(side_effect=RuntimeError("callback failed"))
        provider = module.GenericLLMProvider(FakeLLM([chunk("partial", 0.1), asyncio.CancelledError()]), verbose=False)
        with self.assertLogs("stream_fixture", level="ERROR"):
            with self.assertRaises(asyncio.CancelledError):
                await provider.stream_response([], cost_callback=callback)
        callback.assert_called_once()

    async def test_callback_failure_after_success_propagates_without_reexecution(self):
        callback = Mock(side_effect=RuntimeError("callback failed"))
        llm = FakeLLM([chunk("complete", 0.1)])
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            await module.GenericLLMProvider(llm, verbose=False).stream_response([], cost_callback=callback)
        callback.assert_called_once()
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
