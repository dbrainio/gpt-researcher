"""Offline unit tests; do not import the dependency-heavy researcher package."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "research_usage_evidence", Path(__file__).parents[1] / "gpt_researcher/utils/usage.py"
)
usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usage)


class UsageEvidenceTests(unittest.TestCase):
    def report(self, values, metadata=None):
        return usage.extract_usage_report(SimpleNamespace(usage_metadata=values, response_metadata=metadata or {}), "fixture-model")

    def test_missing_cost_stays_unknown_with_valid_token_usage(self):
        report = self.report({"input_tokens": 12, "output_tokens": 3})
        self.assertEqual(report["total_tokens"], 15)
        self.assertIsNone(report["cost"])
        self.assertFalse(report["cost_is_complete"])
        self.assertEqual(report["cost_source"], "unknown")

    def test_zero_does_not_fall_through_to_another_cost_or_token_alias(self):
        report = self.report({"input_tokens": 0, "prompt_tokens": 99, "total_tokens": 0, "cost": 0, "total_cost": 20}, {"cost": 30})
        self.assertEqual(report["prompt_tokens"], 0)
        self.assertEqual(report["total_tokens"], 0)
        self.assertEqual(report["cost"], 0)
        self.assertTrue(report["cost_is_complete"])

    def test_native_cost_can_exist_without_tokens(self):
        report = self.report({}, {"cost": "0.015"})
        self.assertEqual(report["cost"], 0.015)
        self.assertEqual(report["cost_source"], "provider")

    def test_invalid_cost_is_never_free_or_replaced_by_an_alias(self):
        for value in [float("nan"), float("inf"), -1, True, "NaN", "", {}, []]:
            with self.subTest(value=value):
                report = self.report({"input_tokens": 1, "cost": value, "total_cost": 4})
                self.assertIsNone(report["cost"])
                self.assertFalse(report["cost_is_complete"])

    def test_no_usage_is_not_a_zero_cost_report(self):
        self.assertIsNone(self.report({}))

    def test_fractional_negative_and_unsafe_token_counts_are_rejected(self):
        for value in [-1, 0.5, True, 2**53, float("inf")]:
            with self.subTest(value=value):
                self.assertIsNone(self.report({"input_tokens": value}))

    def test_response_usage_and_normalized_tokens_are_combined(self):
        report = self.report({"input_tokens": 4}, {"usage": {"cost": 0.3, "output_tokens": 2}, "model_name": "actual-model"})
        self.assertEqual(report["total_tokens"], 6)
        self.assertEqual(report["model"], "actual-model")
        self.assertEqual(report["cost"], 0.3)

    def test_unknown_cost_remains_incomplete_after_aggregation(self):
        total = usage.accumulate_research_usage(usage.empty_research_usage(), self.report({"input_tokens": 4}))
        total = usage.accumulate_research_usage(total, self.report({"input_tokens": 3, "cost": 0.2}))
        self.assertEqual(total["cost"], 0.2)
        self.assertEqual(total["tokens_in"], 7)
        self.assertFalse(total["cost_is_complete"])
        self.assertEqual(total["cost_source"], "unknown")

    def test_explicit_provider_zero_is_complete_but_estimated_zero_is_not(self):
        total = usage.accumulate_research_usage(usage.empty_research_usage(), self.report({"cost": 0}))
        self.assertTrue(total["cost_is_complete"])
        self.assertEqual(total["cost_source"], "provider")
        total = usage.accumulate_research_usage(total, 0.0)
        self.assertFalse(total["cost_is_complete"])
        self.assertEqual(total["cost_source"], "mixed")

    def test_child_aggregation_preserves_all_models_and_incomplete_cost(self):
        child = usage.accumulate_research_usage(usage.empty_research_usage(), self.report({"input_tokens": 4, "cost": 0.2}))
        child = usage.accumulate_research_usage(child, 0.1)
        parent = usage.accumulate_research_usage(usage.empty_research_usage(), child)
        self.assertEqual(parent["models"], ["fixture-model"])
        self.assertEqual(parent["tokens_in"], 4)
        self.assertAlmostEqual(parent["cost"], 0.3)
        self.assertFalse(parent["cost_is_complete"])
        self.assertEqual(parent["cost_source"], "mixed")

    def test_old_unlabelled_cost_is_not_upgraded_to_provider_evidence(self):
        result = usage.accumulate_research_usage(usage.empty_research_usage(), {"cost": 0.2})
        self.assertFalse(result["cost_is_complete"])
        self.assertEqual(result["cost_source"], "unknown")

    def test_invalid_estimates_do_not_mutate_the_existing_aggregate(self):
        total = usage.empty_research_usage()
        original = {**total, "models": list(total["models"]), "included_scopes": dict(total["included_scopes"])}
        for value in [True, -1, float("nan"), float("inf"), "0.1"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    usage.accumulate_research_usage(total, value)
        self.assertEqual(total, original)

    def test_parent_declares_which_child_snapshot_is_already_included(self):
        child = usage.accumulate_research_usage(usage.empty_research_usage(), self.report({"cost": 0.2}))
        parent = usage.accumulate_research_usage(usage.empty_research_usage(), child)
        self.assertEqual(parent["included_scopes"], {child["scope_id"]: child["scope_revision"]})
        self.assertNotEqual(parent["scope_id"], child["scope_id"])
        self.assertEqual(parent["scope_revision"], 1)


if __name__ == "__main__":
    unittest.main()
