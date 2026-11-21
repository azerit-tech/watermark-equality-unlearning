"""Tests against a local fake server. Run from the repo root:

    python -m unittest model_equality.api.tests.test_api -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from model_equality.api import config
from model_equality.api.cli import main as cli_main
from model_equality.api.client import (APIClient, BudgetExceeded, ConfirmationRequired,
                                       Runner, ServerUnavailable, expand_endpoints, make_plan)
from model_equality.api.ledger import Ledger, PromptConflict, load_responses
from model_equality.api.tests.fake_server import FakeServer

KEY = "test-key-SECRET-123"
REPO = Path(__file__).resolve().parents[3]
QUIET = lambda s: None  # noqa: E731


class Base(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer(KEY).__enter__()
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.tmp.name) / "ledger.jsonl"

    def tearDown(self):
        self.srv.__exit__()
        self.tmp.cleanup()

    def runner(self, **kw) -> Runner:
        client = APIClient(base_url=self.srv.base_url, api_key=KEY,
                           concurrency=kw.pop("client_concurrency", 4), timeout=(2, 2))
        kw.setdefault("backoff_base", 0.01)
        kw.setdefault("log", QUIET)
        return Runner(self.ledger_path, client, **kw)

    def ledger_records(self):
        return list(Ledger(self.ledger_path).records())

    def assert_no_duplicate_responses(self):
        keys = Counter(r["key"] for r in self.ledger_records() if r.get("type") == "response")
        self.assertTrue(all(v == 1 for v in keys.values()), keys)
        return keys


class TestConfig(unittest.TestCase):
    def test_key_precedence_and_dotenv(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("Secret_judoor=x\nAPI_KEY_BOARD='from-dotenv'")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(config.load_api_key(env), "from-dotenv")
            with mock.patch.dict(os.environ, {"API_KEY_BOARD": "from-env"}, clear=True):
                self.assertEqual(config.load_api_key(env), "from-env")
            with mock.patch.dict(os.environ, {"API_KEY_BOARD": "from-env",
                                              "MODEL_EQUALITY_API_KEY": "override"}, clear=True):
                self.assertEqual(config.load_api_key(env), "override")

    def test_missing_key_is_clear_error(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(config.MissingAPIKey) as cm:
                config.load_api_key(Path(d) / ".env")
            self.assertIn("API_KEY_BOARD", str(cm.exception))

    def test_cli_missing_key(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(config, "REPO_ROOT", Path(d)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = cli_main(["--ledger", str(Path(d) / "l.jsonl"), "budget"])
            self.assertEqual(rc, 2)
            self.assertIn("MissingAPIKey", err.getvalue())

    def test_repr_hides_key(self):
        c = APIClient(base_url="http://127.0.0.1:9", api_key=KEY)
        self.assertNotIn(KEY, repr(c))
        self.assertNotIn(KEY, str(vars(c).get("base_url")))

    def test_plan_helpers(self):
        self.assertEqual(expand_endpoints("ep-01..ep-03,ref-*")[:4], ["ep-01", "ep-02", "ep-03", "ref-01"])
        self.assertEqual(len(expand_endpoints("ep-*")), 33)
        with self.assertRaises(ValueError):
            expand_endpoints("ep-34")
        with self.assertRaises(ValueError):
            make_plan("ep-01", ["x"], 1, 257)
        p = make_plan("ep-01,ep-02", {"a": "A", "b": "B"}, 2, 16)
        self.assertEqual(len(p), 8)
        self.assertEqual([i.endpoint for i in p[:2]], ["ep-01", "ep-02"])  # endpoint-interleaved
        self.assertTrue(config.is_scored("weird-endpoint"))  # unknown -> treated as scored


class TestDryRunAndConfirmation(Base):
    def test_dry_run_sends_nothing(self):
        plan = make_plan("ep-01..ep-03,ref-01", {"p": "hello"}, 3, 16)
        res = self.runner().run(plan)
        self.assertFalse(res.executed)
        self.assertEqual(self.srv.state.posts, 0)
        self.assertEqual(res.report.ep_spend, 9)
        self.assertEqual(res.report.ref_requests, 3)
        self.assertFalse(self.ledger_path.exists())

    def test_ep_requires_exact_confirmation(self):
        plan = make_plan("ep-01,ep-02", {"p": "hello"}, 2, 16)
        r = self.runner()
        with self.assertRaises(ConfirmationRequired):
            r.run(plan, execute=True)
        with self.assertRaises(ConfirmationRequired):
            r.run(plan, execute=True, confirm_spend=5)
        self.assertEqual(self.srv.state.posts, 0)
        res = r.run(plan, execute=True, confirm_spend=4)
        self.assertEqual(res.outcomes, {"response": 4})
        self.assertEqual(self.srv.state.models["ep-01"]["used"], 2)
        # second run: everything already done -> 0 to spend, nothing sent
        res = r.run(plan, execute=True)
        self.assertEqual(res.report.ep_spend, 0)
        self.assertEqual(self.srv.state.posts, 4)

    def test_ref_needs_no_confirmation_and_request_format(self):
        plan = make_plan("ref-01", {"p": "hello"}, 2, 32)
        res = self.runner().run(plan, execute=True)
        self.assertEqual(res.outcomes, {"response": 2})
        body = self.srv.state.requests[0]
        self.assertEqual(body, {"model": "ref-01", "max_tokens": 32,
                                "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(self.srv.state.seen_headers[0].get("X-API-Key"), KEY)
        rec = [r for r in self.ledger_records() if r["type"] == "response"][0]
        for k in ("endpoint", "prompt_id", "prompt", "rep", "max_tokens", "response", "ts", "text"):
            self.assertIn(k, rec)
        self.assertEqual(rec["response"]["budget"]["used"], 1)
        self.assertEqual(rec["finish_reason"], "stop")
        flat = load_responses(self.ledger_path)
        self.assertEqual(len(flat), 2)
        self.assertIsNone(flat[0]["usage"])  # real server sends no "usage"

    def test_ep_refused_when_server_budget_unavailable(self):
        plan = make_plan("ep-01", {"p": "hello"}, 1, 16)
        client = APIClient(base_url="http://127.0.0.1:9/x", api_key=KEY, timeout=(0.5, 0.5))
        r = Runner(self.ledger_path, client, log=QUIET)
        with self.assertRaises(ServerUnavailable):
            r.run(plan, execute=True, confirm_spend=1)

    def test_prompt_conflict(self):
        r = self.runner()
        r.run(make_plan("ref-01", {"p": "hello"}, 1, 16), execute=True)
        with self.assertRaises(PromptConflict):
            r.run(make_plan("ref-01", {"p": "something else"}, 1, 16))


class TestCaps(Base):
    def test_cap_counts_server_used(self):
        self.srv.state.models["ep-02"]["used"] = 45  # teammate spent elsewhere
        plan = make_plan("ep-02", [f"q{i}" for i in range(10)], 1, 16)
        r = self.runner(ep_cap=50)
        with self.assertRaises(BudgetExceeded):
            r.run(plan, execute=True, confirm_spend=10)
        self.assertEqual(self.srv.state.posts, 0)
        res = r.run(plan, execute=True, confirm_spend=5, allow_partial=True)
        self.assertEqual(res.outcomes, {"response": 5})
        self.assertEqual(self.srv.state.models["ep-02"]["used"], 50)
        # now at cap: nothing more can be sent
        res = r.run(plan, allow_partial=True)
        self.assertEqual(res.report.ep_spend, 0)

    def test_cap_counts_local_ledger(self):
        r = self.runner(ep_cap=6)
        r.run(make_plan("ep-03", ["a", "b", "c", "d"], 1, 16), execute=True, confirm_spend=4)
        # server forgets (or reports less); the local ledger still blocks
        self.srv.state.models["ep-03"]["used"] = 0
        with self.assertRaises(BudgetExceeded):
            r.run(make_plan("ep-03", ["e", "f", "g"], 1, 16), execute=True, confirm_spend=3)
        self.assertEqual(self.srv.state.posts, 4)

    def test_cap_never_above_100(self):
        with self.assertRaises(ValueError):
            self.runner(ep_cap=101)

    def test_server_limit_lower_than_cap(self):
        self.srv.state.models["ep-04"]["limit"] = 3
        plan = make_plan("ep-04", ["a", "b", "c", "d"], 1, 16)
        rep = self.runner(ep_cap=50).run(plan).report
        self.assertEqual(rep.blocked[0].over, 1)


class TestServerErrors(Base):
    def test_busy_429_retried_and_not_counted(self):
        self.srv.state.faults["ep-01"] = ["busy", "busy"]
        r = self.runner()
        res = r.run(make_plan("ep-01", ["a"], 1, 16), execute=True, confirm_spend=1)
        self.assertEqual(res.outcomes, {"response": 1})
        st = Ledger(self.ledger_path).state()
        self.assertEqual(st.local_used["ep-01"], 1)
        self.assertEqual(st.errors["ep-01"], 2)
        self.assertEqual(self.srv.state.models["ep-01"]["used"], 1)

    def test_429_without_retry_after_stops_endpoint_only(self):
        self.srv.state.faults["ep-05"] = ["exhaust"]
        plan = make_plan("ep-05,ep-06", ["a", "b", "c"], 1, 16)
        r = self.runner(concurrency=1)
        res = r.run(plan, execute=True, confirm_spend=6)
        self.assertEqual(res.exhausted, ["ep-05"])
        self.assertEqual(res.responses_by_endpoint, {"ep-06": 3})
        self.assertEqual(res.outcomes.get("budget_exhausted"), 1)
        self.assertEqual(res.outcomes.get("not_sent_exhausted"), 2)
        posts_ep05 = sum(1 for b in self.srv.state.requests if b["model"] == "ep-05")
        self.assertEqual(posts_ep05, 1)

    def test_5xx_retried_and_conservatively_counted(self):
        self.srv.state.faults["ep-07"] = ["502", "503", "504"]
        r = self.runner(concurrency=1)
        res = r.run(make_plan("ep-07", ["a"], 1, 16), execute=True, confirm_spend=1)
        # the run may only spend what was confirmed: after a 502 (maybe charged)
        # there is no unit left for a retry
        self.assertEqual(res.outcomes, {"not_sent_budget": 1})
        self.assertEqual(Ledger(self.ledger_path).state().local_used["ep-07"], 1)
        # a larger confirmed budget lets the retries through
        res = r.run(make_plan("ep-07", ["a"], 1, 16), allow_partial=True)
        self.assertEqual(res.report.ep_spend, 1)
        plan2 = make_plan("ep-07", ["a", "b", "c", "d"], 1, 16)
        res = r.run(plan2, execute=True, confirm_spend=4)
        self.assertEqual(res.outcomes.get("response"), 2)  # 2 more 5xx ate 2 units
        st = Ledger(self.ledger_path).state()
        self.assertEqual(st.local_used["ep-07"], 5)
        self.assertEqual(st.responses["ep-07"], 2)
        self.assert_no_duplicate_responses()

    def test_teammate_spending_during_run_is_honoured(self):
        # cap 10; after our 1st answer a teammate spends 6 on ep-14 -> we must stop at 10
        self.srv.state.faults["ep-14"] = ["teammate:6"]
        r = self.runner(ep_cap=10, concurrency=1)
        plan = make_plan("ep-14", [f"t{i}" for i in range(10)], 1, 16)
        res = r.run(plan, execute=True, confirm_spend=10)
        self.assertEqual(self.srv.state.models["ep-14"]["used"], 10)
        self.assertEqual(res.outcomes.get("response"), 4)
        self.assertEqual(res.outcomes.get("not_sent_budget"), 6)

    def test_export_starter_format(self):
        self.runner().run(make_plan("ref-01", {"p1": "hello"}, 2, 16), execute=True)
        out = Path(self.tmp.name) / "answers.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            cli_main(["--ledger", str(self.ledger_path), "export", str(out)])
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(sorted(r["repeat"] for r in rows), [0, 1])
        self.assertEqual(set(rows[0]) - {"max_tokens"}, {"endpoint", "id", "repeat", "prompt", "response"})

    def test_5xx_on_ref_retried(self):
        self.srv.state.faults["ref-02"] = ["503", "502"]
        res = self.runner().run(make_plan("ref-02", ["a"], 1, 16), execute=True)
        self.assertEqual(res.outcomes, {"response": 1})

    def test_dropped_connection_counts_and_retries(self):
        self.srv.state.faults["ep-08"] = ["drop"]
        r = self.runner()
        res = r.run(make_plan("ep-08", ["a", "b"], 1, 16), execute=True, confirm_spend=2)
        st = Ledger(self.ledger_path).state()
        # server charged the dropped one; ledger counts it too ("maybe")
        self.assertEqual(st.local_used["ep-08"], self.srv.state.models["ep-08"]["used"])
        self.assertGreaterEqual(st.local_used["ep-08"], 2)
        self.assertEqual(sum(res.outcomes.values()), 2)

    def test_invalid_request_not_counted(self):
        r = self.runner()
        res = r.run(make_plan("ep-09", ["INVALID prompt"], 1, 16), execute=True, confirm_spend=1)
        self.assertEqual(res.outcomes, {"rejected_400": 1})
        self.assertEqual(Ledger(self.ledger_path).state().local_used["ep-09"], 0)

    def test_auth_error_aborts_run(self):
        client = APIClient(base_url=self.srv.base_url, api_key="wrong", timeout=(2, 2))
        r = Runner(self.ledger_path, client, log=QUIET, concurrency=1)
        res = r.run(make_plan("ref-01", ["a", "b", "c"], 1, 16), execute=True)
        self.assertIn("authentication", res.aborted)
        self.assertEqual(self.srv.state.posts, 1)


class TestConcurrency(Base):
    def _max_inflight(self, conc, tag=""):
        self.srv.state.delay = 0.05
        self.srv.state.max_inflight = 0
        plan = make_plan("ref-01..ref-09", [f"c{conc}{tag}-{i}" for i in range(3)], 1, 8)
        self.runner(concurrency=conc).run(plan, execute=True)
        return self.srv.state.max_inflight

    def test_never_more_than_configured(self):
        self.assertLessEqual(self._max_inflight(2), 2)
        self.assertLessEqual(self._max_inflight(4), 4)
        self.assertEqual(self._max_inflight(4, "b"), 4)  # and it actually uses the slots

    def test_two_runners_share_process_limit(self):
        self.srv.state.delay = 0.05
        l2 = Path(self.tmp.name) / "other.jsonl"
        r1 = self.runner(concurrency=4)
        client2 = APIClient(base_url=self.srv.base_url, api_key=KEY)
        r2 = Runner(l2, client2, concurrency=4, log=QUIET)
        t = threading.Thread(target=r2.run, args=(make_plan("ref-*", ["x", "y"], 1, 8),),
                             kwargs={"execute": True})
        t.start()
        r1.run(make_plan("ref-*", ["z", "w"], 1, 8), execute=True)
        t.join()
        self.assertLessEqual(self.srv.state.max_inflight, 4)

    def test_concurrency_above_4_rejected(self):
        with self.assertRaises(ValueError):
            self.runner(concurrency=5)
        with self.assertRaises(ValueError):
            APIClient(base_url="http://x", api_key=KEY, concurrency=5)


class TestInterruptAndResume(Base):
    def test_stop_event_then_resume(self):
        self.srv.state.delay = 0.05
        plan = make_plan("ep-10,ep-11,ref-03", [f"q{i}" for i in range(6)], 1, 16)
        r = self.runner(concurrency=2)
        stop = threading.Event()
        threading.Timer(0.15, stop.set).start()
        res = r.run(plan, execute=True, confirm_spend=12, stop_event=stop)
        self.assertTrue(res.interrupted)
        n1 = res.outcomes.get("response", 0)
        self.assertGreater(n1, 0)
        self.assertLess(n1, 18)
        # every request the server answered is in the ledger
        self.assertEqual(self.srv.state.posts, n1)
        rep = r.run(plan).report
        res2 = r.run(plan, execute=True, confirm_spend=rep.ep_spend)
        self.assertEqual(n1 + res2.outcomes["response"], 18)
        self.assertEqual(self.srv.state.posts, 18)
        self.assert_no_duplicate_responses()

    def test_kill_9_mid_campaign_then_resume(self):
        """Hard crash of a real CLI process, then resume from the ledger."""
        self.srv.state.delay = 0.1
        plan_file = Path(self.tmp.name) / "plan.json"
        plan_file.write_text(json.dumps({"endpoints": "ep-12,ep-13,ref-04",
                                         "prompts": [{"id": f"k{i}", "text": f"prompt {i}"}
                                                     for i in range(8)],
                                         "repeats": 1, "max_tokens": 16}))
        env = dict(os.environ, MODEL_EQUALITY_API_KEY=KEY,
                   MODEL_EQUALITY_BASE_URL=self.srv.base_url, PYTHONPATH=str(REPO))
        args = [sys.executable, "-m", "model_equality.api", "--ledger", str(self.ledger_path),
                "--concurrency", "3", "run", "--plan", str(plan_file), "--execute"]
        p = subprocess.Popen(args + ["--confirm-spend", "16"], env=env, cwd=REPO,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.time() + 10
        while time.time() < deadline:
            n = sum(1 for r in self.ledger_records() if r.get("type") == "response") \
                if self.ledger_path.exists() else 0
            if n >= 5:
                break
            time.sleep(0.02)
        p.send_signal(signal.SIGKILL)
        out, err = p.communicate()
        self.assertNotIn(KEY.encode(), out + err)
        # simulate a torn final line on top of the crash
        with open(self.ledger_path, "ab") as fh:
            fh.write(b'{"type":"response","trunc')
        st = Ledger(self.ledger_path).state()
        self.assertEqual(st.corrupt_lines, 1)
        n_resp = sum(st.responses.values())
        self.assertLess(n_resp, 24)
        # every POST the server saw was preceded by a durable "attempt" record
        n_attempts = sum(1 for r in self.ledger_records() if r.get("type") == "attempt")
        time.sleep(0.3)  # let the server finish requests orphaned by the kill
        self.assertGreaterEqual(n_attempts, self.srv.state.posts)
        self.assertGreater(sum(st.dangling.values()), 0)
        server_ep_used = sum(self.srv.state.models[e]["used"] for e in ("ep-12", "ep-13"))
        self.assertGreaterEqual(st.local_used["ep-12"] + st.local_used["ep-13"], server_ep_used)

        # dry run on resume shows what is left
        dry = subprocess.run(args[:-1], env=env, cwd=REPO, capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        left_ep = 16 - (st.responses["ep-12"] + st.responses["ep-13"])
        self.assertIn(f"--confirm-spend {left_ep}", dry.stdout)
        res = subprocess.run(args + ["--confirm-spend", str(left_ep)], env=env, cwd=REPO,
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr + res.stdout)
        keys = self.assert_no_duplicate_responses()
        self.assertEqual(len(keys), 24)
        self.assertNotIn(KEY, self.ledger_path.read_text())
        self.assertNotIn(KEY, res.stdout + res.stderr + dry.stdout + dry.stderr)

    def test_second_run_on_same_ledger_refused(self):
        from model_equality.api.ledger import LedgerBusy
        l1 = Ledger(self.ledger_path)
        l1.acquire_run_lock()
        try:
            with self.assertRaises(LedgerBusy):
                self.runner().run(make_plan("ref-01", ["a"], 1, 8), execute=True)
        finally:
            l1.release_run_lock()


class TestCLI(Base):
    def _cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"MODEL_EQUALITY_API_KEY": KEY}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(["--ledger", str(self.ledger_path), "--base-url", self.srv.base_url,
                           *argv])
        self.assertNotIn(KEY, out.getvalue() + err.getvalue())
        return rc, out.getvalue(), err.getvalue()

    def test_budget_status_run(self):
        self.srv.state.models["ep-01"]["used"] = 7
        rc, out, _ = self._cli("budget")
        self.assertEqual(rc, 0)
        self.assertIn("ep-01", out)
        self.assertIn("ref-09", out)
        rc, out, _ = self._cli("run", "--endpoints", "ep-01,ref-01", "--prompt", "hi",
                               "--repeats", "2", "--max-tokens", "8")
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn("--confirm-spend 2", out)
        self.assertEqual(self.srv.state.posts, 0)
        rc, _, err = self._cli("run", "--endpoints", "ep-01,ref-01", "--prompt", "hi",
                               "--repeats", "2", "--max-tokens", "8", "--execute")
        self.assertEqual(rc, 2)
        self.assertIn("ConfirmationRequired", err)
        self.assertEqual(self.srv.state.posts, 0)
        rc, out, _ = self._cli("run", "--endpoints", "ep-01,ref-01", "--prompt", "hi",
                               "--repeats", "2", "--max-tokens", "8", "--execute",
                               "--confirm-spend", "2")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.srv.state.posts, 4)
        rc, out, _ = self._cli("status", "--by-prompt")
        self.assertEqual(rc, 0)
        self.assertIn("ep-01", out)

    def test_cap_flag(self):
        self.srv.state.models["ep-02"]["used"] = 9
        rc, out, _ = self._cli("--cap", "10", "run", "--endpoints", "ep-02", "--prompt", "a",
                               "--prompt", "b", "--max-tokens", "8")
        self.assertIn("OVER CAP by 1", out)
        rc, _, err = self._cli("--cap", "10", "run", "--endpoints", "ep-02", "--prompt", "a",
                               "--prompt", "b", "--max-tokens", "8", "--execute",
                               "--confirm-spend", "2")
        self.assertEqual(rc, 2)
        self.assertIn("BudgetExceeded", err)
        self.assertEqual(self.srv.state.posts, 0)


if __name__ == "__main__":
    unittest.main()
