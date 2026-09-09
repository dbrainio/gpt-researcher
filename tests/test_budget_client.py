"""Private budget protocol, no network, SDK or production credentials."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
from io import BytesIO
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location("budget_fixture", Path(__file__).parents[1] / "gpt_researcher/utils/budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
CAP = "nbgt1.run." + "r" * 43
RECEIPT = "nbgt1.receipt." + "s" * 43


def tracked(mode="enforce"):
    return {"kind": "tracked", "receipt": RECEIPT, "mode": mode, "modelId": "openai/gpt-4o-mini", "maxOutputTokens": 200,
            "correlation": {"budget_operation_id": "deep-research:run:0", "budget_reservation_id": "reservation00001"}}


class BudgetClientTests(unittest.TestCase):
    def test_requires_safe_explicit_correlation_without_decoding_receipt(self):
        for pair in [None, {}, {"budget_operation_id": RECEIPT, "budget_reservation_id": "reservation00001"},
                     {"budget_operation_id": "deep-research:run:0", "budget_reservation_id": "bad"},
                     {**tracked()["correlation"], "receipt": RECEIPT}]:
            with self.assertRaises(budget.ResearchBudgetError):
                budget.ResearchBudgetOperation(Mock(), {**tracked(), "correlation": pair})
        admission = tracked()
        operation = budget.ResearchBudgetOperation(Mock(), admission)
        admission["correlation"]["budget_operation_id"] = "changed"
        self.assertEqual(operation.correlation["budget_operation_id"], "deep-research:run:0")
        self.assertNotIn(RECEIPT, str(operation.correlation))

    def test_only_provably_unstarted_operation_can_release(self):
        callback = Mock(return_value={"kind": "acknowledged"})
        operation = budget.ResearchBudgetOperation(callback, tracked())
        operation.release_unstarted()
        operation.release_unstarted()
        callback.assert_called_once_with(RECEIPT, {"action": "release", "reason": "provider_not_called"})
        with self.assertRaises(budget.ResearchBudgetError):
            operation.mark_started()
        for action in [lambda op: op.mark_started(), lambda op: op.observe("gen-native"), lambda op: op.finalize("0")]:
            callback.reset_mock()
            operation = budget.ResearchBudgetOperation(callback, tracked())
            action(operation)
            callback.reset_mock()
            with self.assertRaises(budget.ResearchBudgetError):
                operation.release_unstarted()
            callback.assert_not_called()

    def test_wrapped_sdk_errors_preserve_the_budget_code_and_cycles_terminate(self):
        native = budget.ResearchBudgetError("budget_exceeded")
        sdk = RuntimeError("connection wrapper")
        sdk.__cause__ = native
        self.assertIs(budget.find_budget_error(sdk), native)
        cycle = RuntimeError("cycle")
        cycle.__cause__ = cycle
        self.assertIsNone(budget.find_budget_error(cycle))

    def test_reserve_scope_and_controls_then_original_receipt_for_settlement(self):
        callback = Mock(side_effect=[tracked(), {"kind": "acknowledged"}, {"kind": "acknowledged"}])
        run = budget.ResearchBudget(CAP, "enforce", callback)
        operation = run.reserve_model("gpt-4o-mini", 400, 500)
        self.assertEqual(operation.model_id, "openai/gpt-4o-mini")
        self.assertEqual(operation.max_output_tokens, 200)
        self.assertEqual(callback.call_args_list[0].args, (CAP, {"action": "reserve", "step": 0, "provider": "openrouter", "modelId": "gpt-4o-mini", "inputBytes": 400, "maxOutputTokens": 500}))
        operation.observe("gen-native")
        operation.finalize("0.0000001")
        self.assertEqual(callback.call_args_list[-1].args, (RECEIPT, {"action": "finalize", "providerUsageId": "gen-native", "providerCostUsd": "0.0000001"}))
        self.assertNotIn(CAP, repr(run))
        self.assertNotIn(RECEIPT, repr(operation))

    def test_concurrent_children_share_unique_steps(self):
        callback = Mock(return_value={"kind": "bypass"})
        run = budget.ResearchBudget(CAP, "enforce", callback)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: run.reserve_model("gpt-4o-mini", 400, 500), range(100)))
        self.assertEqual(sorted(call.args[1]["step"] for call in callback.call_args_list), list(range(100)))

    def test_bounds_fail_before_callback_and_never_retry(self):
        for args in [("gpt-4o-mini", True, 200), ("gpt-4o-mini", 4_194_305, 200), ("gpt-4o-mini", 400, 0)]:
            callback = Mock()
            with self.assertRaises(budget.ResearchBudgetError):
                budget.ResearchBudget(CAP, "enforce", callback).reserve_model(*args)
            callback.assert_not_called()

    def test_uncertain_admission_fails_closed_without_retry(self):
        callback = Mock(side_effect=RuntimeError("private callback payload"))
        with self.assertRaises(budget.ResearchBudgetError) as caught:
            budget.ResearchBudget(CAP, "enforce", callback).reserve_model("gpt-4o-mini", 400, 500)
        callback.assert_called_once()
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("private callback payload", str(caught.exception))

    def test_shadow_unavailable_is_nonblocking_but_replay_never_reexecutes(self):
        for code, denied in [("budget_internal_error", False), ("budget_exceeded", True), ("budget_invalid_transition", True), ("budget_idempotency_conflict", True)]:
            callback = Mock(side_effect=budget.ResearchBudgetError(code))
            run = budget.ResearchBudget(CAP, "shadow", callback)
            if denied:
                with self.assertRaises(budget.ResearchBudgetError):
                    run.reserve_model("gpt-4o-mini", 400, 500)
                with self.assertRaises(budget.ResearchBudgetError):
                    run.reserve_model("gpt-4o-mini", 400, 500)
            else:
                with self.assertLogs("budget_fixture", level="WARNING"):
                    self.assertIsNone(run.reserve_model("gpt-4o-mini", 400, 500))
            callback.assert_called_once()

    def test_malformed_or_mismatched_admission_is_not_permission(self):
        for result in [{}, {**tracked(), "modelId": "other/model"}, {**tracked(), "receipt": "jwt"}, {**tracked(), "maxOutputTokens": 501}, {**tracked(), "maxOutputTokens": True}]:
            callback = Mock(return_value=result)
            with self.assertRaises(budget.ResearchBudgetError):
                budget.ResearchBudget(CAP, "enforce", callback).reserve_model("gpt-4o-mini", 400, 500)

    def test_observation_without_cost_never_finalizes_or_refunds(self):
        callback = Mock(return_value={"kind": "acknowledged"})
        operation = budget.ResearchBudgetOperation(callback, tracked())
        operation.observe("gen-native")
        operation.observe("gen-native")
        callback.assert_called_once_with(RECEIPT, {"action": "observe", "providerUsageId": "gen-native"})
        with self.assertRaises(budget.ResearchBudgetError):
            operation.observe("gen-other")
        for cost in [None, True, 0, "NaN", "Infinity", "-1", ""]:
            with self.assertRaises(budget.ResearchBudgetError):
                operation.finalize(cost)
        callback.assert_called_once()

    def test_explicit_zero_is_valid_and_duplicate_finalization_is_local_noop(self):
        callback = Mock(return_value={"kind": "acknowledged"})
        operation = budget.ResearchBudgetOperation(callback, tracked())
        operation.finalize("0")
        operation.finalize("0")
        callback.assert_called_once_with(RECEIPT, {"action": "finalize", "providerUsageId": None, "providerCostUsd": "0"})
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize("1")

    def test_failed_shadow_observation_retains_native_identity_for_finalization(self):
        callback = Mock(side_effect=[RuntimeError("failed"), {"kind": "acknowledged"}])
        operation = budget.ResearchBudgetOperation(callback, tracked("shadow"))
        with self.assertLogs("budget_fixture", level="WARNING"):
            operation.observe("gen-native")
        operation.finalize("0.2")
        self.assertEqual(callback.call_args.args[1]["providerUsageId"], "gen-native")

    def test_unknown_ack_fails_closed_and_does_not_cache_settlement(self):
        callback = Mock(side_effect=[{}, {"kind": "acknowledged"}])
        operation = budget.ResearchBudgetOperation(callback, tracked())
        with self.assertRaises(budget.ResearchBudgetError):
            operation.finalize("0.2")
        # Retrying ledger settlement is allowed; no native paid call exists here.
        operation.finalize("0.2")
        self.assertEqual(callback.call_count, 2)

    def test_callback_url_is_private_configuration_not_an_arbitrary_destination(self):
        for url in ["", "http://public.example/api/internal/budget/operations", "https://user:pass@app.test/api/internal/budget/operations", "https://app.test/other", "https://app.test/api/internal/budget/operations?token=x"]:
            with self.assertRaises(budget.ResearchBudgetError):
                budget.BudgetCallback(url)
        for url in ["http://localhost:3000/api/internal/budget/operations", "http://ui.default.svc.cluster.local/api/internal/budget/operations", "https://app.test/api/internal/budget/operations"]:
            budget.BudgetCallback(url)

    def test_http_callback_uses_authorization_and_bounds_response_without_retry(self):
        transport = budget.BudgetCallback("https://app.test/api/internal/budget/operations")
        transport._opener = Mock()
        transport._opener.open.return_value = BytesIO(b'{"kind":"acknowledged"}')
        self.assertEqual(transport(RECEIPT, {"action": "observe", "providerUsageId": "gen-native"}), {"kind": "acknowledged"})
        request = transport._opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer " + RECEIPT)
        self.assertNotIn(RECEIPT.encode(), request.data)
        self.assertEqual(transport._opener.open.call_args.kwargs, {"timeout": 10})
        transport._opener.open.return_value = BytesIO(b"x" * 16_385)
        with self.assertRaises(budget.ResearchBudgetError):
            transport(RECEIPT, {})
        self.assertEqual(transport._opener.open.call_count, 2)

    def test_http_errors_keep_only_allowlisted_code_and_do_not_follow_redirects(self):
        transport = budget.BudgetCallback("https://app.test/api/internal/budget/operations")
        transport._opener = Mock()
        for raw, expected in [(b'{"code":"budget_exceeded","error":"private"}', "budget_exceeded"), (b'{"code":{},"error":"private"}', "budget_internal_error"), (b"not json", "budget_internal_error")]:
            transport._opener.open.side_effect = HTTPError("https://app.test", 500, "private", {}, BytesIO(raw))
            with self.assertRaises(budget.ResearchBudgetError) as caught:
                transport(CAP, {})
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn("private", str(caught.exception))
        for status, expected in [(401, "budget_invalid_transition"), (403, "budget_invalid_transition"), (409, "budget_invalid_transition"), (429, "budget_exceeded")]:
            transport._opener.open.side_effect = HTTPError("https://app.test", status, "private", {}, BytesIO(b"{}"))
            with self.assertRaises(budget.ResearchBudgetError) as caught:
                transport(CAP, {})
            self.assertEqual(caught.exception.code, expected)
        self.assertIsNone(budget._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
