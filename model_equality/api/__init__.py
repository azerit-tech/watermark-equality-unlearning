"""Budget-safe client for the Model Equality Testing API (task 2).

    from model_equality.api import APIClient, Runner, make_plan, load_responses
    plan = make_plan("ref-01..ref-03", {"p1": "Say hi"}, repeats=2, max_tokens=32)
    runner = Runner("model_equality/api/ledger/ledger.jsonl", APIClient())
    print(runner.run(plan).text())                     # dry run
    runner.run(plan, execute=True)                     # ref- only: no confirmation
    runner.run(ep_plan, execute=True, confirm_spend=N) # N = ep- spend from dry run
"""
from .client import (APIClient, AuthError, BudgetExceeded, ConfirmationRequired, PlanItem,
                     Runner, ServerUnavailable, combine_plans, expand_endpoints, make_plan)
from .config import DEFAULT_LEDGER, MissingAPIKey, load_api_key
from .ledger import Ledger, LedgerBusy, PromptConflict, load_responses

__all__ = ["APIClient", "AuthError", "BudgetExceeded", "ConfirmationRequired", "PlanItem",
           "Runner", "ServerUnavailable", "combine_plans", "expand_endpoints", "make_plan",
           "DEFAULT_LEDGER", "MissingAPIKey", "load_api_key", "Ledger", "LedgerBusy",
           "PromptConflict", "load_responses"]
