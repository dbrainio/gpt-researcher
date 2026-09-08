import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).parents[1] / "gpt_researcher/utils"
package = ModuleType("budget_scope_fixture")
package.__path__ = [str(ROOT)]
with patch.dict(sys.modules, {"budget_scope_fixture": package}):
    spec = importlib.util.spec_from_file_location("budget_scope_fixture.budget_scope", ROOT / "budget_scope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
SECRET = "offline-fixture-secret-only-at-least-32-chars"
CLAIMS = {"version": 1, "kind": "run", "tool": "deep_research", "mode": "enforce", "runId": "test:1", "subject": {"userId": "user000000000001", "workspaceId": "workspace0000001"}, "issuedAt": 1_000_000, "expiresAt": 2_000_000}


def sign(claims):
    data = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    message = "nbgt1." + data
    key = hmac.new(SECRET.encode(), b"nevel/budget-bridge/v1", hashlib.sha256).digest()
    return message + "." + base64.urlsafe_b64encode(hmac.new(key, message.encode(), hashlib.sha256).digest()).decode().rstrip("=")


class CapabilityTests(unittest.TestCase):
    def test_signed_run_mode_and_expiry(self):
        self.assertEqual(module.verify_run_capability(sign(CLAIMS), SECRET, 1_000_000), CLAIMS)
        with self.assertRaises(module.ResearchBudgetError):
            module.verify_run_capability(sign(CLAIMS), SECRET, 2_000_000)

    def test_changed_mode_without_new_signature_cannot_downgrade_to_shadow(self):
        token = sign(CLAIMS).split(".")
        token[1] = sign({**CLAIMS, "mode": "shadow"}).split(".")[1]
        with self.assertRaises(module.ResearchBudgetError):
            module.verify_run_capability(".".join(token), SECRET, 1_000_000)

    def test_receipts_bad_subject_and_invalid_lifetimes_are_not_runs(self):
        for claims in [{**CLAIMS, "kind": "receipt"}, {**CLAIMS, "admin": True}, {**CLAIMS, "subject": {"userId": "spoofed"}}, {**CLAIMS, "expiresAt": 9_000_000}, {**CLAIMS, "issuedAt": True}, {**CLAIMS, "issuedAt": 1_100_000}]:
            with self.assertRaises(module.ResearchBudgetError):
                module.verify_run_capability(sign(claims), SECRET, 1_000_000)


class ScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_swallowed_inner_denial_cannot_produce_successful_overall_report(self):
        owned = Mock(aclose=AsyncMock())
        owned.raise_if_denied.side_effect = module.ResearchBudgetError("budget_exceeded")
        @module.with_research_budget
        async def run(headers=None):
            return "partial fallback"
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000), patch.object(module, "ResearchBudget", return_value=owned):
            with self.assertRaises(module.ResearchBudgetError) as caught:
                await run(headers={"nevel_budget": {"capability": sign(CLAIMS)}})
        self.assertEqual(caught.exception.code, "budget_exceeded")
        owned.aclose.assert_awaited_once()
        self.assertIsNone(module.current_research_budget.get())

    async def test_private_header_is_removed_and_context_reaches_children_and_threads(self):
        owned = Mock(aclose=AsyncMock())
        headers = {"nevel_budget": {"capability": sign(CLAIMS)}, "nevel_tracking": {"user": "fixture"}}
        @module.with_research_budget
        async def run(headers=None):
            self.assertNotIn("nevel_budget", headers)
            self.assertIn("nevel_tracking", headers)
            self.assertIs(module.current_research_budget.get(), owned)
            self.assertIs(await asyncio.to_thread(module.current_research_budget.get), owned)
            async def child():
                return module.current_research_budget.get()
            self.assertEqual(await asyncio.gather(child(), child()), [owned, owned])
            return "done"
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000), patch.object(module, "ResearchBudget", return_value=owned) as factory:
            self.assertEqual(await run(headers), "done")
            factory.assert_called_once_with(headers["nevel_budget"]["capability"], "enforce")
        self.assertIsNone(module.current_research_budget.get())
        owned.aclose.assert_awaited_once()
        self.assertIn("nevel_budget", headers)

    async def test_cancellation_closes_owned_clients_and_restores_scope(self):
        owned = Mock(aclose=AsyncMock())
        @module.with_research_budget
        async def run(headers=None):
            raise asyncio.CancelledError()
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000), patch.object(module, "ResearchBudget", return_value=owned):
            with self.assertRaises(asyncio.CancelledError):
                await run(headers={"nevel_budget": {"capability": sign(CLAIMS)}})
        owned.aclose.assert_awaited_once()
        self.assertIsNone(module.current_research_budget.get())

    async def test_unauthenticated_shadow_claim_never_reaches_paid_body(self):
        called = Mock()
        @module.with_research_budget
        async def run(headers=None):
            called()
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000):
            with self.assertRaises(module.ResearchBudgetError):
                await run(headers={"nevel_budget": {"capability": "invalid", "mode": "shadow"}})
        called.assert_not_called()

    async def test_only_verified_shadow_may_bypass_missing_callback_configuration(self):
        called = Mock()
        @module.with_research_budget
        async def run(headers=None):
            called()
            self.assertNotIn("nevel_budget", headers)
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000), patch.object(module, "ResearchBudget", side_effect=RuntimeError("missing config")):
            with self.assertLogs("budget_scope_fixture.budget_scope", level="WARNING"):
                await run(headers={"nevel_budget": {"capability": sign({**CLAIMS, "mode": "shadow"})}})
            with self.assertRaises(module.ResearchBudgetError):
                await run(headers={"nevel_budget": {"capability": sign(CLAIMS)}})
        called.assert_called_once()

    async def test_no_capability_preserves_existing_non_budget_run(self):
        @module.with_research_budget
        async def run(headers=None):
            self.assertIsNone(module.current_research_budget.get())
            return headers
        with patch.object(module, "ResearchBudget") as factory:
            self.assertEqual(await run(headers={"custom": "value"}), {"custom": "value"})
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
