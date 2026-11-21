"""Tests of the analysis pipeline on synthetic endpoints with known truth.

Run:  python3.12 -m pytest model_equality/analysis/tests -q
(or)  python3.12 -m unittest model_equality.analysis.tests.test_analysis
"""
from __future__ import annotations

import itertools
import tempfile
import unittest

import numpy as np
from scipy import stats as sstats

from model_equality.analysis import synth
from model_equality.analysis.data import Probe, Response, units_of
from model_equality.analysis.decision import best_partition, decide
from model_equality.analysis.extract import parse, parse_dataset
from model_equality.analysis.labels import det_flags, modal_agreement
from model_equality.analysis.pairs import compute_pairs, wm_signature
from model_equality.analysis.stats import PairTables
from model_equality.scoring.metrics import score

D = synth.Deployment


def pair_stat(d_a, d_b, n, seed, B=150, free=False):
    d_a = D(**{**d_a.__dict__, "name": "a"})
    d_b = D(**{**d_b.__dict__, "name": "b"})
    ds = synth.make_dataset([d_a, d_b], n, seed=seed)
    P = parse_dataset(ds)
    pt = PairTables.build(P, "a", "b", ds.probes, units_of(ds.probes), B, np.random.default_rng(seed), free=free)
    return pt.stat()


# ---------------------------------------------------------------- extraction
class TestExtraction(unittest.TestCase):
    def test_stream(self):
        p = Probe("S", "S_stream", "S", {"type": "stream", "alphabet": [str(i) for i in range(10)], "sep": "comma"})
        r = parse(p, Response("Sure! 4, 0, 7, 7, 1, 2, 3, 9, 9, 0, 5, 1"))
        self.assertEqual(r.items, ["4", "0", "7", "7", "1", "2", "3", "9", "9", "0", "5"])  # last token: cut?
        self.assertTrue(r.ok)
        self.assertEqual(r.events[("S|^", "4")], 1)
        self.assertEqual(r.events[("S|7", "7")], 1)
        self.assertLess(r.fmt, 1.0)
        r2 = parse(p, Response("4, 0, 7, 7, 1, 2, 3, 9, 9, 0, 5, 1."))   # ends with '.': complete
        self.assertEqual(len(r2.items), 12)

    def test_binary_stream_order3(self):
        p = Probe("C", "S_stream", "C", {"type": "stream", "alphabet": ["H", "T"], "sep": "comma"})
        r = parse(p, Response("h, t, T, H, H, T, T, H, T, H, H,"))
        self.assertIn(("C|^.^.^", "H"), r.events)
        self.assertIn(("C|H.T.T", "H"), r.events)

    def test_form(self):
        p = Probe("F", "F_form", "F", {"type": "form", "n": 4, "options": ["A", "B"]})
        r = parse(p, Response("Here you go:\n1: A\n**2:** b\n3. C\n3: A\n5: A\n4 - B (coffee)"))
        self.assertEqual(r.items, [(1, "A"), (2, "B"), (4, "B")])   # 3 -> invalid (first answer kept), 5 out of range
        self.assertAlmostEqual(r.fmt, 0.75)
        self.assertFalse(r.ok)

    def test_form_echoed_item_is_not_an_answer(self):
        # real data, ref-04 on F01: the item is copied instead of answered
        p = Probe("F", "F_form", "F", {"type": "form", "n": 3, "options": ["A", "B"]})
        r = parse(p, Response("1: A: tea / B: coffee\n2: B (dogs)\n3: A. Summer / B. Winter"))
        self.assertEqual(r.items, [(2, "B")])
        self.assertFalse(r.ok)
        q = Probe("M", "F_form", "M", {"type": "form", "n": 2, "options": ["A", "B", "C", "D"]})
        self.assertEqual(parse(q, Response("1. B: tungsten\n2. D: carbon dioxide")).items, [(1, "B"), (2, "D")])

    def test_tree_view(self):
        from model_equality.analysis.data import Dataset, add_tree_views
        base = Probe("F", "F_form", "F", {"type": "form", "n": 3, "options": ["yes", "no"]})
        ds = Dataset({"F": base}, {"e1": {"F": [Response("1: Yes\n2: no"), Response("1: Yes")]}})
        self.assertEqual(add_tree_views(ds, K=4), ["F@tree"])
        v = ds.probes["F@tree"]
        self.assertTrue(v.discrete and v.long and v.unit == "F@tree")
        r = parse(v, ds.responses["e1"]["F@tree"][0])
        self.assertEqual(r.items, ["1", ":", "Yes", "\\n"])
        self.assertEqual(r.events[("F@tree|0|", "1")], 1)
        self.assertEqual(r.events[("F@tree|2|1\x1f:", "Yes")], 1)
        r2 = parse(v, ds.responses["e1"]["F@tree"][1])
        self.assertEqual(r2.items, ["1", ":", "Yes", "<eos>"])     # stopping early is an event

    def test_form_open_tools(self):
        p = Probe("T", "F_form", "T", {"type": "form_open", "n": 2, "options": ["gmail_send", "outlook_send", "x_y"]})
        r = parse(p, Response("1: `outlook_send`\n2: I would use gmail_send."))
        self.assertEqual(r.items, [(1, "outlook_send"), (2, "gmail_send")])

    def test_form_open_numbers(self):
        p = Probe("N", "F_form", "N", {"type": "form_open", "n": 2, "options": None})
        r = parse(p, Response("1: 4,042\n2: 1 234."))
        self.assertEqual(r.items, [(1, "4042"), (2, "1234")])

    def test_first_word_and_match(self):
        p = Probe("K1", "K_single", "K", {"type": "first_word", "options": ["yes", "no"]})
        self.assertEqual(parse(p, Response("**Yes.** because")).items, ["yes"])
        self.assertFalse(parse(p, Response("Maybe")).ok)
        q = Probe("KT", "K_single", "KT", {"type": "first_match", "options": ["bing_search", "google_search"]})
        r = parse(q, Response("I choose google_search, not bing_search"))
        self.assertEqual(r.items, ["google_search"])

    def test_open_list(self):
        p = Probe("L", "S_stream_open", "L", {"type": "open_list", "sep": "line"})
        txt = "\n".join(f"{i}. Word{i}" for i in range(1, 13)) + "\nWor"
        r = parse(p, Response(txt))
        self.assertEqual(r.n, 12)        # the cut last item is dropped
        self.assertTrue(r.ok)
        self.assertIn(("L|first", "word1"), r.events)


# ---------------------------------------------------------------- calibration
class TestNullCalibration(unittest.TestCase):
    def test_pvalues_uniform_under_null(self):
        n = synth.default_counts(n_stream=6, n_form=8, n_k=20, n_free=8)
        ps = {k: [] for k in ("raw", "temp", "inv", "mmd")}
        for r in range(60):
            s = pair_stat(D("a", 0), D("b", 0), n, seed=1000 + r, B=100, free=True)
            for k in ps:
                ps[k].append(s[f"p_{k}"])
        for k, v in ps.items():
            v = np.array(v)
            self.assertLessEqual(np.mean(v < 0.05), 0.15, f"{k}: too many rejections under H0")
            self.assertGreater(sstats.kstest(v, "uniform").pvalue, 1e-3, f"{k}: p-values not uniform")

    def test_split_halves_centred(self):
        n = synth.default_counts(n_stream=8, n_form=8, n_k=20)
        z = [pair_stat(D("a", 0), D("b", 0), n, seed=r, B=100)["z_temp"] for r in range(30)]
        self.assertLess(abs(np.mean(z)), 0.6)


# ---------------------------------------------------------------- lures vs real differences
class TestInvariance(unittest.TestCase):
    n = synth.default_counts(n_stream=15, n_form=15, n_k=40)

    def test_temperature_ignored_by_t_temp(self):
        for T in (0.6, 1.4):
            zt = [pair_stat(D("a", 0), D("b", 0, T=T), self.n, seed=r)["z_temp"] for r in range(8)]
            self.assertLess(np.median(zt), 2.0, f"T={T}")
        zr = [pair_stat(D("a", 0), D("b", 0, T=0.6), self.n, seed=r)["z_raw"] for r in range(4)]
        self.assertGreater(np.median(zr), 4.0, "T_raw must see the temperature")

    def test_kappa_estimates_temperature_ratio(self):
        s = pair_stat(D("a", 0), D("b", 0, T=0.5), synth.default_counts(40, 40, 100), seed=3)
        self.assertAlmostEqual(np.log(s["kappa_temp"]), np.log(2.0), delta=0.3)

    def test_watermark_ignored_by_t_inv_and_detected(self):
        for r in range(4):
            s = pair_stat(D("a", 0), D("b", 0, wm_delta=2.0), self.n, seed=r)
            self.assertGreater(s["z_temp"], 4.0)
            self.assertLess(s["z_inv"], 2.0)
            self.assertTrue(wm_signature(s["_inv"])["wm_sig"])

    def test_lora_detected_not_watermark(self):
        for r in range(3):
            s = pair_stat(D("a", 0), D("b", 0, lora=0.8), self.n, seed=r)
            self.assertGreater(s["z_temp"], 4.0)
            self.assertGreater(s["z_inv"], 4.0)
            self.assertFalse(wm_signature(s["_inv"])["wm_sig"])

    def test_quantization_detected_with_enough_draws_and_weaker_than_lora(self):
        n = synth.default_counts(n_stream=30, n_form=30, n_k=80)
        zq = np.median([pair_stat(D("a", 0), D("b", 0, quant=0.25), n, seed=r)["z_temp"] for r in range(4)])
        zl = np.median([pair_stat(D("a", 0), D("b", 0, lora=0.8), n, seed=r)["z_temp"] for r in range(4)])
        self.assertGreater(zq, 3.0)
        self.assertGreater(zl, zq)

    def test_quantization_under_temperature_lure(self):
        """A variant must stay visible through T_temp even when it also runs at another temperature."""
        n = synth.default_counts(n_stream=30, n_form=30, n_k=80)
        z = np.median([pair_stat(D("a", 0), D("b", 0, lora=0.5, T=1.4), n, seed=r)["z_temp"] for r in range(3)])
        self.assertGreater(z, 4.0)

    def test_greedy_det_and_modal_agreement(self):
        deps = [D("base", 0), D("greedy", 0, greedy=True), D("lora", 0, lora=1.5)]
        n = synth.default_counts(n_stream=0, n_form=30, n_k=0, n_free=2)
        n = {k: v for k, v in n.items() if v}
        ds = synth.make_dataset(deps, n, seed=1)
        det = det_flags(ds)
        self.assertTrue(det["greedy"]["det"])
        self.assertFalse(det["base"]["det"])
        P = parse_dataset(ds)
        a_base, nb = modal_agreement(P, ds.probes, "greedy", "base")
        a_lora, _ = modal_agreement(P, ds.probes, "greedy", "lora")
        self.assertGreater(nb, 5)
        self.assertGreaterEqual(a_base, 0.8)   # modal answer != argmax when the top two are close
        self.assertLess(a_lora, a_base)


# ---------------------------------------------------------------- decision
class TestDecision(unittest.TestCase):
    def test_partition_dp_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = 6
            W = rng.normal(0, 2, (n, n))
            W = (W + W.T) / 2
            groups, val = best_partition(W)
            best = -1e9
            for lab in itertools.product(range(n), repeat=n):
                s = sum(W[i, j] for i in range(n) for j in range(i + 1, n) if lab[i] == lab[j])
                best = max(best, s)
            self.assertAlmostEqual(val, best, places=9)
            self.assertEqual(sorted(i for g in groups for i in g), list(range(n)))

    def test_ref_like_family_end_to_end(self):
        deps, truth = synth.ref_like_family(quant=0.4, lora_unit=0.8)
        ds = synth.make_dataset(deps, synth.default_counts(n_stream=40, n_form=40, n_k=100), seed=3)
        P = parse_dataset(ds)
        rows, _ = compute_pairs(P, ds.probes, ds.endpoints, units_of(ds.probes), B=150, seed=1, n_boot=20,
                                per_unit=False)
        res = decide(ds.endpoints, rows, ds, P)
        sc = score(res["family"], res["model"], {e: t[0] for e, t in truth.items()},
                   {e: t[1] for e, t in truth.items()})
        self.assertGreaterEqual(sc["mcc_model"], 0.9, res["families"]["F"]["groups"])
        lab = res["families"]["F"]["label"]
        self.assertEqual(lab["syn-06"], "WM")
        self.assertEqual(lab["syn-03"], "SP")


# ---------------------------------------------------------------- command line
class TestCLI(unittest.TestCase):
    def test_commands_run_on_fake_ledger(self):
        from model_equality.analysis.cli import main
        deps, truth = synth.ref_like_family(quant=0.4, lora_unit=0.8)
        ds = synth.make_dataset(deps, synth.default_counts(n_stream=12, n_form=12, n_k=30, n_free=4), seed=2)
        with tempfile.TemporaryDirectory() as d:
            f = synth.write_fake_files(ds, truth, d)
            com = ["--ledger", str(f["ledger"]), "--probes", str(f["probes"]), "--endpoints", "syn-*", "--jobs", "1"]
            self.assertEqual(main(["extract", *com, "--out-dir", f"{d}/x"]), 0)
            self.assertEqual(main(["pairs", *com, "--out-dir", f"{d}/p", "--B", "60", "--n-boot", "5"]), 0)
            self.assertEqual(main(["decide", *com, "--out-dir", f"{d}/d", "--pairs", f"{d}/p/pairs.csv",
                                   "--families", "one", "--truth", str(f["truth"])]), 0)
            # second run: the previous partition is moved aside, a timestamped copy is written
            self.assertEqual(main(["decide", *com, "--out-dir", f"{d}/d", "--pairs", f"{d}/p/pairs.csv",
                                   "--families", "one"]), 0)
            import glob
            self.assertTrue(glob.glob(f"{d}/d/partition.stale-*.csv"))
            self.assertTrue(glob.glob(f"{d}/d/partition_2*.csv"))
            self.assertEqual(main(["select", *com, "--out-dir", f"{d}/s", "--truth", str(f["truth"]), "--draws", "2",
                                   "--n-sub", "6", "--B", "40", "--hard", "syn-01/syn-02,syn-01/syn-09"]), 0)
            self.assertEqual(main(["simulate", *com, "--out-dir", f"{d}/m", "--truth", str(f["truth"]),
                                   "--battery", "SS:4,SLL:4,SF:4", "--draws", "2", "--B", "40", "--n-boot", "5"]), 0)
            import pandas as pd
            part = pd.read_csv(f"{d}/d/partition.csv")
            self.assertEqual(len(part), 9)
            self.assertTrue((pd.read_csv(f"{d}/s/select_ranked.csv")["unit"].nunique()) >= 3)


class TestAuditSafeguards(unittest.TestCase):
    """Audit points: family collapse, family file, stale CSV, no-data endpoints, full 33-endpoint chain."""

    @classmethod
    def setUpClass(cls):
        deps, truth = synth.ref_like_family(prefix="ep-")      # ep-01..ep-09: not a full deliverable
        cls.ds = synth.make_dataset(deps, synth.default_counts(n_stream=8, n_form=8, n_k=20), seed=4)
        cls.truth = truth

    def _files(self, d):
        return synth.write_fake_files(self.ds, self.truth, d)

    def test_auto_families_collapse_fails(self):
        from model_equality.analysis.cli import main
        with tempfile.TemporaryDirectory() as d:
            f = self._files(d)
            com = ["--ledger", str(f["ledger"]), "--probes", str(f["probes"]), "--endpoints", "ep-*", "--jobs", "1",
                   "--B", "30", "--n-boot", "0", "--out-dir", f"{d}/o"]
            self.assertEqual(main(["decide", *com, "--families", "auto"]), 4)   # no free text -> one family

    def test_invalid_partition_exit_code_and_stale_file(self):
        from pathlib import Path
        from model_equality.analysis.cli import main
        with tempfile.TemporaryDirectory() as d:
            f = self._files(d)
            Path(f"{d}/o").mkdir()
            Path(f"{d}/o/partition.csv").write_text("stale\n")
            fam = Path(d) / "fam.csv"
            fam.write_text("endpoint,family\n" + "".join(f"ep-{i:02d},A\n" for i in range(1, 10)))
            com = ["--ledger", str(f["ledger"]), "--probes", str(f["probes"]), "--endpoints", "ep-*", "--jobs", "1",
                   "--B", "30", "--n-boot", "0", "--out-dir", f"{d}/o"]
            self.assertEqual(main(["decide", *com, "--families", str(fam)]), 3)   # 9 ep- only: invalid deliverable
            self.assertFalse(Path(f"{d}/o/partition.csv").exists())
            import glob
            self.assertTrue(glob.glob(f"{d}/o/partition.stale-*.csv"))
            self.assertTrue(glob.glob(f"{d}/o/partition_draft_*.csv"))

    def test_families_file_validation(self):
        from pathlib import Path
        from model_equality.analysis.cli import read_families
        eps = ["ep-01", "ep-02", "ep-03"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            p.write_text("endpoint,family\nep-01,A\nep-02,A\nep-03,B\n")
            self.assertEqual(read_families(str(p), eps), {"ep-01": "A", "ep-02": "A", "ep-03": "B"})
            for bad in ("ep-01,A\nep-02,A\nep-03,B\n",                       # no header
                        "endpoint,family\nep-01,A\nep-02\nep-03,B\n",          # short line
                        "endpoint,family\nep-01,A\nep-02,A\n",                 # missing endpoint
                        "endpoint,family\nep-01,A\nep-01,B\nep-02,A\nep-03,B\n",  # duplicate
                        "endpoint,family\nep-01,A\nep-02,A\nep-03,B\nep-04,B\n"):  # unknown endpoint
                p.write_text(bad)
                with self.assertRaises(ValueError):
                    read_families(str(p), eps)

    def test_endpoint_without_data_stays_alone(self):
        ds = synth.make_dataset([D("a", 0), D("b", 0), D("c", 0)], synth.default_counts(8, 8, 20), seed=1)
        ds.responses["c"] = {}                                  # no data at all for c
        P = parse_dataset(ds)
        rows, _ = compute_pairs(P, ds.probes, ["a", "b", "c"], units_of(ds.probes), B=50, n_boot=0, per_unit=False)
        res = decide(["a", "b", "c"], rows, ds, P)
        fam = res["families"]["F"]
        self.assertIn("c", fam["no_data"])
        self.assertIn(["c"], fam["groups"])

    def test_full_chain_33_endpoints_to_valid_csv(self):
        """Auditor's smoke test: synthetic 33 endpoints / 4 families -> decide -> CSV -> validator -> score."""
        from model_equality.scoring.csvio import validate, write_submission
        kinds = [dict(), dict(sp=0.6), dict(wm_delta=2.0), dict(T=1.4), dict(quant=0.25), dict(lora=0.35),
                 dict(lora=0.75, lora_dir=2), dict(sp=0.6, sp_key=12)]
        deps, truth, fam = [], {}, {}
        i = 1
        for f, (seed, n) in enumerate([(0, 12), (100, 9), (200, 8), (300, 4)]):
            for k in range(n):
                kw = kinds[k % len(kinds)]
                e = f"ep-{i:02d}"
                i += 1
                deps.append(D(e, seed, **kw))
                variant = any(x in kw for x in ("quant", "lora"))
                truth[e] = (f"T{f}", f"T{f}_{e}" if variant else f"T{f}_base")
                fam[e] = f"T{f}"
        ds = synth.make_dataset(deps, seed=1)
        P = parse_dataset(ds)
        pairs = [(a, b) for a, b in itertools.combinations(sorted(truth), 2) if fam[a] == fam[b]]
        rows, _ = compute_pairs(P, ds.probes, sorted(truth), units_of(ds.probes), B=100, seed=0, n_boot=10,
                                per_unit=False, pairs=pairs)
        res = decide(sorted(truth), rows, ds, P, families=fam)
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/sub.csv"
            write_submission(path, res["family"], res["model"])
            self.assertTrue(validate(path).ok)
        s = score(res["family"], res["model"], {e: t[0] for e, t in truth.items()}, {e: t[1] for e, t in truth.items()})
        self.assertEqual(s["mcc_family"], 1.0)
        self.assertGreater(s["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
