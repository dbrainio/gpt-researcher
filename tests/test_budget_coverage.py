"""Configuration drift cannot silently introduce unmetered B2C calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_budget_tavily import load, budget, CAP


class CoverageTests(unittest.TestCase):
    def test_default_retriever_and_header_overrides_are_checked_before_import(self):
        with patch.dict("sys.modules", {"gpt_researcher.utils.budget": budget}):
            retrievers = load("retriever_coverage_fixture", "actions/retriever.py")
        run = budget.ResearchBudget(CAP, "enforce", Mock())
        token = budget.current_research_budget.set(run)
        try:
            constructor = Mock()
            with patch.dict("sys.modules", {"gpt_researcher.retrievers": SimpleNamespace(TavilySearch=constructor)}):
                config = SimpleNamespace(retrievers=None, retriever=None)
                self.assertEqual(retrievers.get_retrievers({}, config), [constructor])
                self.assertEqual(retrievers.get_retrievers({"retrievers": "tavily"}, config), [constructor])
                with self.assertRaises(budget.ResearchBudgetError):
                    retrievers.get_retrievers({"retriever": "exa"}, config)
                with self.assertRaises(budget.ResearchBudgetError):
                    run.raise_if_denied()
        finally:
            budget.current_research_budget.reset(token)

    def test_shadow_unpriced_config_warns_safely_and_off_is_unchanged(self):
        private = "https://secret:credential@private.example"
        budget.require_budget_coverage("external_mcp", private, ())
        run = budget.ResearchBudget(CAP, "shadow", Mock())
        token = budget.current_research_budget.set(run)
        try:
            with self.assertLogs(budget.__name__, level="WARNING") as logs:
                budget.require_budget_coverage("external_mcp", private, ())
            self.assertNotIn(private, str(logs.output))
            run.raise_if_denied()
        finally:
            budget.current_research_budget.reset(token)

    def test_enforce_unpriced_component_latches_before_any_admission(self):
        for component, selection in [("external_mcp", "enabled"), ("scraper", "firecrawl"), ("retriever", "serper")]:
            callback = Mock()
            run = budget.ResearchBudget(CAP, "enforce", callback)
            token = budget.current_research_budget.set(run)
            try:
                with self.assertRaises(budget.ResearchBudgetError):
                    budget.require_budget_coverage(component, selection, ())
                with self.assertRaises(budget.ResearchBudgetError):
                    run.reserve_tavily("search", "basic", 1)
                callback.assert_not_called()
            finally:
                budget.current_research_budget.reset(token)


if __name__ == "__main__":
    unittest.main()
