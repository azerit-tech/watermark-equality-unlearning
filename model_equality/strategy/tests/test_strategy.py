"""Run from the repository root:
    python3.12 -m unittest discover -s model_equality/strategy/tests -t . -v
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_equality.scoring import metrics
from model_equality.scoring.csvio import validate
from model_equality.strategy import cli
from model_equality.strategy import simulate as sim
from model_equality.strategy.core import W_FAMILY, W_MODEL, Belief, Decode, canon, mcc, n_decimals

EPS = sim.EPS


def world(seed):
    rng = np.random.default_rng(seed)
    w = sim.make_world(rng)
    cand = sim.naive_candidate(w)
    return w, cand, sim.Server(w, rng), rng


def true_counts(dec, srv):
    ts, pub = srv.tsame[srv.fsame], srv.pub[srv.fsame]
    m = len(dec.sizes)
    return (np.array([(ts & pub & (dec.cls == c)).sum() for c in range(m)]),
            np.array([(~ts & pub & (dec.cls == c)).sum() for c in range(m)]))


class TestScore(unittest.TestCase):
    def test_mcc_matches_metrics(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            tp, fp, fn, tn = (int(x) for x in rng.integers(0, 12, 4))
            self.assertAlmostEqual(float(mcc(tp, fp, fn, tn)), max(0.0, metrics.mcc_from_counts(tp, tn, fp, fn)), 12)

    def test_server_full_score_matches_metrics(self):
        w, cand, srv, rng = world(3)
        fam = {e: f"F{w['fam'][i]}" for i, e in enumerate(EPS)}
        truth = {e: f"t{w['truth'][i]}" for i, e in enumerate(EPS)}
        for lab in (cand, canon(rng.integers(0, 4, 33) + 10 * w["fam"])):
            pred = {e: f"m{lab[i]}" for i, e in enumerate(EPS)}
            self.assertAlmostEqual(srv.full(lab), metrics.score(fam, pred, fam, truth)["score"], 12)

    def test_constant_prediction_scores_family_term(self):
        # the first real submission: right families, one model per family -> exactly 0.1
        w, cand, srv, _ = world(4)
        self.assertAlmostEqual(srv.public(canon(w["fam"])), W_FAMILY, 12)
        self.assertAlmostEqual(srv.public(np.arange(33)), W_FAMILY, 12)


class TestDecode(unittest.TestCase):
    def test_truth_survives_every_score(self):
        for seed in (11, 12, 13):
            w, cand, srv, rng = world(seed)
            pb = sim.problem_of(w, cand)
            dec = Decode(pb.nw, 0.3)
            lab = pb.cand
            for step in range(6):
                s = pb.pairs_same(lab)
                self.assertTrue(dec.add(s, srv.public(lab)))
                p, d = true_counts(dec, srv)
                hit = (dec.hp == p).all(1) & (dec.hd == d).all(1)
                self.assertEqual(int(hit.sum()), 1, f"seed {seed} step {step}")
                nb = pb.neighbours(pb.cand)
                lab = nb[int(rng.integers(len(nb)))]
            self.assertLessEqual(len(dec.hp), 2000)

    def test_wrong_score_is_rejected(self):
        w, cand, srv, _ = world(14)
        pb = sim.problem_of(w, cand)
        dec = Decode(pb.nw, 0.3)
        self.assertFalse(dec.add(pb.pairs_same(pb.cand), 0.123456789))


class TestBelief(unittest.TestCase):
    def test_worlds_reproduce_the_scores(self):
        w, cand, srv, _ = world(21)
        pb = sim.problem_of(w, cand)
        bel = Belief(pb, n_chains=96, seed=1)
        lab = pb.cand
        for _ in range(5):
            bel.observe(lab, srv.public(lab))
            bel.refresh(6)
            prop = bel.propose(n_rep=2)
            if prop["lab"] is None:
                break
            self.assertNotIn(prop["lab"].tobytes(), {x["lab"].tobytes() for x in bel.subs})
            self.assertTrue(0 < pb.pairs_same(prop["lab"]).sum() < pb.nw)
            lab = prop["lab"]
        self.assertEqual(bel.mode, "exact")
        t, pub = bel.sample_worlds(2)
        ctx = bel._ctx(t, pub)
        ok = np.ones(t.shape[0], bool)
        for x in bel.scored():
            pm = bel.evaluate(x["lab"][None], ctx)[0][:, 0]
            ok &= np.abs(W_FAMILY + W_MODEL * pm - x["score"]) < 1e-9
        self.assertGreater(ok.mean(), 0.8)

    def test_probe_sets_family_term(self):
        w, cand, srv, _ = world(22)
        pb = sim.problem_of(w, cand)
        bel = Belief(pb, n_chains=32, seed=1, prior_sweeps=2)
        sub = bel.observe(canon(w["fam"]), 0.1)
        self.assertEqual(sub["kind"], "probe")
        self.assertIsNone(bel.incumbent())
        self.assertTrue(bel.propose()["first"])

    def test_rounded_scores_fall_back(self):
        w, cand, srv, _ = world(23)
        srv.decimals = 4
        pb = sim.problem_of(w, cand)
        bel = Belief(pb, n_chains=48, seed=1, prior_sweeps=4)
        lab = pb.cand
        for _ in range(3):
            bel.observe(lab, srv.public(lab))
            bel.refresh(4)
            prop = bel.propose(n_rep=2)
            if prop["lab"] is None:
                break
            lab = prop["lab"]
        self.assertEqual(bel.mode, "approx")
        p, d = true_counts(bel.dec, srv)
        self.assertTrue(((bel.dec.hp == p).all(1) & (bel.dec.hd == d).all(1)).any())

    def test_two_decimal_scores(self):
        # as on the real leaderboard so far (0.1 then 0.35): few decimals -> taken as rounded
        self.assertEqual([n_decimals(x) for x in (0.1, 0.35, 0.4123, 0.5836065573770492)], [1, 2, 4, 16])
        w, cand, srv, _ = world(25)
        srv.decimals = 2
        pb = sim.problem_of(w, cand)
        bel = Belief(pb, n_chains=32, seed=1, prior_sweeps=2)
        bel.observe(pb.cand, srv.public(pb.cand))
        self.assertEqual(bel.mode, "approx")
        self.assertAlmostEqual(bel.tol_used, 0.0051)
        p, d = true_counts(bel.dec, srv)
        self.assertTrue(((bel.dec.hp == p).all(1) & (bel.dec.hd == d).all(1)).any())
        bel.observe(pb.neighbours(pb.cand)[0], 0.5836065573770492)     # a long score: no rounding after all
        self.assertIn(bel.mode, ("exact", "sign"))

    def test_scores_on_a_005_grid(self):
        # as on the real leaderboard (0.35, 0.70, 0.75, 0.85...): every score is a multiple of 0.05 ->
        # tolerance of half a step, and a change smaller than the rounding is a tie
        w, cand, srv, _ = world(25)
        pb = sim.problem_of(w, cand)
        s = round(round(srv.public(pb.cand) / 0.05) * 0.05, 2)
        bel = Belief(pb, n_chains=32, seed=1, prior_sweeps=2)
        bel.observe(pb.cand, s)
        self.assertEqual(bel.mode, "approx")
        self.assertAlmostEqual(bel._grid_now(), 0.05)
        self.assertAlmostEqual(bel.tol_used, 0.0255)
        p, d = true_counts(bel.dec, srv)
        self.assertTrue(((bel.dec.hp == p).all(1) & (bel.dec.hd == d).all(1)).any())
        shown = bel._shown(np.array([0.0, 0.66, 0.68, 0.72]))      # 0.1 + 0.9 * MCC = 0.100, 0.694, 0.712, 0.748
        self.assertTrue(np.allclose(shown, [0.10, 0.70, 0.70, 0.75]))
        b2 = Belief(pb, n_chains=32, seed=1, prior_sweeps=2, step=None)     # grid read from the decimals only
        b2.observe(pb.cand, s)
        self.assertAlmostEqual(b2._grid_now(), 10.0 ** -n_decimals(s))

    def test_inconsistent_scores_go_to_sign_mode(self):
        w, cand, srv, _ = world(24)
        pb = sim.problem_of(w, cand)
        bel = Belief(pb, n_chains=48, seed=1, prior_sweeps=4)
        lab = pb.cand
        for k in range(3):
            bel.observe(lab, srv.public(lab) - 0.0234)      # e.g. an unknown, lower family term
            bel.refresh(4)
            prop = bel.propose(n_rep=2)
            self.assertIsNotNone(prop)
            if prop["lab"] is None:
                break
            lab = prop["lab"]
        self.assertEqual(bel.mode, "sign")


class TestCli(unittest.TestCase):
    def test_round_trip(self):
        w, cand, srv, _ = world(31)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "cand.csv").write_text("endpoint,family,model\n" + "".join(
                f"{e},F{w['fam'][i]},F{w['fam'][i]}m{cand[i]}\n" for i, e in enumerate(EPS)))
            with open(tmp / "pairs.csv", "w") as fh:
                fh.write("a,b,p_model\n")
                for i in range(33):
                    for j in range(i + 1, 33):
                        fh.write(f"{EPS[i]},{EPS[j]},{w['P'][i, j]:.4f}\n")
            # a fake submit journal with the families probe (constant prediction, score 0.1)
            subdir = tmp / "submissions"
            subdir.mkdir()
            (subdir / "a.csv").write_text("endpoint,family,model\n" + "".join(
                f"{e},F{w['fam'][i]},F{w['fam'][i]}M\n" for i, e in enumerate(EPS)))
            ts = "2026-11-21T18:34:38+00:00"
            (subdir / "log.jsonl").write_text(
                json.dumps({"kind": "attempt", "ts": ts, "sha256": "x", "copy": "a.csv"}) + "\n" +
                json.dumps({"kind": "result", "attempt_ts": ts, "status": "success", "score": 0.1}) + "\n")
            common = ["--state-dir", str(tmp / "state"), "--chains", "48", "--sweeps", "4", "--rep", "2"]
            files = ["--pairs", str(tmp / "pairs.csv"), "--candidate", str(tmp / "cand.csv")]
            out = tmp / "next" / "partition.csv"

            def run(*args):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rc = cli.main(list(args))
                return rc, buf.getvalue()

            rc, txt = run("update", *common, *files, "--submit-dir", str(subdir))
            self.assertEqual(rc, 0)
            self.assertIn("Familles justes", txt)
            rc, txt = run("update", *common, "--submit-dir", str(subdir))
            self.assertIn("Rien de nouveau", txt)
            for k in range(3):
                rc, txt = run("next", *common, *files, "--out", str(out))
                self.assertEqual(rc, 0, txt)
                if not out.exists():
                    break
                rep = validate(str(out))
                self.assertTrue(rep.ok, rep.format())
                ids: dict = {}
                lab = np.array([ids.setdefault(rep.rows[e], len(ids)) for e in EPS])
                if k == 0:
                    self.assertTrue(np.array_equal(canon(lab), canon(cand)))
                    self.assertIn("partition candidate telle quelle", txt)
                else:
                    self.assertIn("Ce que la soumission teste", txt)
                self.assertTrue((out.parent / "summary.txt").exists())
                rc, txt = run("update", *common, "--csv", str(out), "--score", repr(srv.public(lab)))
                self.assertEqual(rc, 0, txt)
                out.unlink()
            rc, txt = run("final", *common)
            self.assertEqual(rc, 0, txt)
            self.assertIn("RETENUE", txt)
            rc, txt = run("assess", *common, str(tmp / "cand.csv"))
            self.assertEqual(rc, 0, txt)
            self.assertIn("score public attendu", txt)
            rc, txt = run("status", "--state-dir", str(tmp / "state"))
            self.assertIn("#01", txt)
            hist = cli.read_history(tmp / "state")
            self.assertEqual(hist[0]["source"], "submit-log")
            self.assertGreaterEqual(len(hist), 3)

    def test_submission_in_flight(self):
        w, cand, srv, _ = world(32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            subdir = tmp / "submissions"
            subdir.mkdir()
            (subdir / "a.csv").write_text("endpoint,family,model\n" + "".join(
                f"{e},F{w['fam'][i]},F{w['fam'][i]}m{cand[i]}\n" for i, e in enumerate(EPS)))
            ts = "2026-11-21T19:40:00+00:00"
            attempt = json.dumps({"kind": "attempt", "ts": ts, "sha256": "x", "copy": "a.csv"}) + "\n"
            (subdir / "log.jsonl").write_text(attempt)
            hist: list = []
            added = cli.sync_submit_log(tmp / "state", hist, subdir)
            self.assertEqual((len(added), hist[0]["status"], hist[0]["score"]), (1, "pending", None))
            self.assertEqual(cli.sync_submit_log(tmp / "state", hist, subdir), [])
            (subdir / "log.jsonl").write_text(attempt + json.dumps(
                {"kind": "result", "attempt_ts": ts, "status": "success", "score": 0.47}) + "\n")
            added = cli.sync_submit_log(tmp / "state", hist, subdir)
            self.assertEqual((len(added), len(hist), hist[0]["status"], hist[0]["score"]), (1, 1, "scored", 0.47))

    def test_slots(self):
        import datetime as dt
        now = dt.datetime(2026, 11, 21, 19, 0, tzinfo=dt.timezone.utc)       # 20:00 in Paris
        hist = [{"ts": "2026-11-21T18:34:38+00:00"}]
        info = cli.slots_info(hist, "2026-11-22T12:00", now=now)
        self.assertEqual((info["next"].hour, info["next"].minute), (20, 6))
        self.assertEqual(info["left"], 30)


class TestPrior(unittest.TestCase):
    def test_hint_against_one_member_holds_for_the_group(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fam = {e: f"F{i % 3}" for i, e in enumerate(EPS)}
            group = {"F0": ["ep-01", "ep-04", "ep-07"], "F1": ["ep-02", "ep-05"], "F2": ["ep-03", "ep-06"]}
            (tmp / "cand.csv").write_text("endpoint,family,model\n" + "".join(
                f"{e},{fam[e]},{fam[e] + 'G' if e in group[fam[e]] else fam[e] + e}\n" for e in EPS))
            (tmp / "pairs.csv").write_text("a,b,p_model\nep-10,ep-01,0.8\n")
            a = argparse.Namespace(candidate=str(tmp / "cand.csv"), pairs=str(tmp / "pairs.csv"), q=0.3,
                                   pi_hidden=0.12, p_missing=0.3, p_twin=0.05, p_merge=0.05, p_lo=0.03,
                                   p_hi=0.98, no_anchor=False, _warned=True)
            pb = cli.build_problem(a, {})
            i = {e: k for k, e in enumerate(EPS)}
            for mate in group["F0"]:
                self.assertAlmostEqual(pb.P[i["ep-10"], i[mate]], 0.8, 6)       # the hint, for the whole group
                self.assertAlmostEqual(pb.P[i["ep-13"], i[mate]], 0.3, 6)       # no hint: --p-missing
            self.assertAlmostEqual(pb.P[i["ep-10"], i["ep-13"]], 0.05, 6)       # two lone endpoints
            self.assertAlmostEqual(pb.P[i["ep-01"], i["ep-04"]], 0.88, 6)       # inside a candidate group
            self.assertTrue(pb.anchor[i["ep-01"]] and not pb.anchor[i["ep-10"]])
            # prior alone: a lone endpoint joins the group about p of the time, hinted one more often
            bel = Belief(pb, n_chains=600, seed=3, prior_sweeps=30)
            same = bel.L[:, i["ep-01"]][:, None] == bel.L
            self.assertAlmostEqual(same[:, i["ep-13"]].mean(), 0.3, delta=0.08)
            self.assertGreater(same[:, i["ep-10"]].mean(), 0.6)                 # (twin options dilute it a little)
            twin = (bel.L[:, i["ep-13"]] == bel.L[:, i["ep-16"]]) & ~same[:, i["ep-13"]]
            self.assertLess(twin.mean(), 0.1)


class TestSimulation(unittest.TestCase):
    def test_split_scenario(self):
        r = sim.one_world(43, slots=5, chains=64, sweeps=4, scenario="split", greedy=True)
        self.assertEqual(len(r["bayes_traj"]), 5)
        self.assertGreaterEqual(r["bayes_pub"], r["naive_pub"] - 1e-12)

    def test_small_run(self):
        res = [sim.one_world(s, slots=6, chains=64, sweeps=4) for s in (41, 42)]
        txt = sim.report(res, 6)
        self.assertIn("bayes", txt)
        for r in res:
            self.assertEqual(len(r["bayes_traj"]), 6)
            self.assertGreaterEqual(r["bayes_pub"], r["naive_pub"] - 1e-12)   # the kept one is the public best


if __name__ == "__main__":
    unittest.main()
