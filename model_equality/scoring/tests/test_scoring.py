"""Run from the repository root:
    python3.12 -m unittest discover -s model_equality/scoring/tests -t . -v
"""
from __future__ import annotations

import datetime as dt
import io
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from model_equality.scoring import csvio, metrics, partition, submit

HERE = Path(__file__).resolve().parent
SCORING = HERE.parent
REPO = SCORING.parent.parent
EPS = csvio.SUBMISSION_ENDPOINTS

# Synthetic 33-endpoint truth: 6 families, several models per family.
_SIZES = [(3, 2, 1), (2, 2, 2), (4, 1), (3, 3), (2, 1, 1, 1), (3, 2)]
TRUE_F, TRUE_M = {}, {}
_it = iter(EPS)
for fi, sizes in enumerate(_SIZES):
    for mi, s in enumerate(sizes):
        for _ in range(s):
            e = next(_it)
            TRUE_F[e], TRUE_M[e] = f"fam{fi}", f"fam{fi}m{mi}"
assert len(TRUE_F) == 33


def sk_mcc(pred, true):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(true, pred) if len(set(true)) > 1 or len(set(pred)) > 1 else 0.0


def write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def csv_text(fam, mod, header=True):
    lines = ["endpoint,family,model"] if header else []
    lines += [f"{e},{fam[e]},{mod[e]}" for e in sorted(fam)]
    return "\n".join(lines) + "\n"


class TestMetrics(unittest.TestCase):
    def test_pair_count(self):
        r = metrics.score(TRUE_F, TRUE_M, TRUE_F, TRUE_M)
        self.assertEqual(r["family_confusion"]["n_pairs"], 528)
        self.assertEqual(r["model_confusion"]["all"]["n_pairs"], 528)

    def test_perfect_is_one(self):
        r = metrics.score(TRUE_F, TRUE_M, TRUE_F, TRUE_M)
        self.assertAlmostEqual(r["mcc_family"], 1.0)
        for v in metrics.VARIANTS:
            self.assertAlmostEqual(r["mcc_model_variants"][v], 1.0)
        self.assertAlmostEqual(r["score"], 1.0)
        self.assertEqual(r["family_confusion"]["false_pairs"], [])

    def test_relabelled_perfect_is_one(self):
        f = {e: "X" + v for e, v in TRUE_F.items()}
        m = {e: "Y" + v for e, v in TRUE_M.items()}
        self.assertAlmostEqual(metrics.score(f, m, TRUE_F, TRUE_M)["score"], 1.0)

    def test_all_same_is_zero(self):
        f = {e: "A" for e in EPS}
        m = {e: "A1" for e in EPS}
        r = metrics.score(f, m, TRUE_F, TRUE_M)
        self.assertEqual(r["mcc_family"], 0.0)
        for v in metrics.VARIANTS:
            self.assertEqual(r["mcc_model_variants"][v], 0.0)
        self.assertEqual(r["score"], 0.0)

    def test_all_different_is_zero(self):
        f = {e: e for e in EPS}
        m = {e: e + "m" for e in EPS}
        r = metrics.score(f, m, TRUE_F, TRUE_M)
        self.assertEqual(r["score"], 0.0)
        self.assertEqual(r["score_variants"]["pred"], 0.0)

    def test_family_only_gives_point_one(self):
        m = {e: TRUE_F[e] + "_single" for e in EPS}  # right families, one model per family
        r = metrics.score(TRUE_F, m, TRUE_F, TRUE_M)
        self.assertAlmostEqual(r["score"], 0.1)

    def test_negative_mcc_clipped(self):
        # anti-correlated prediction on a 4-endpoint toy case
        tf = {"a": "F", "b": "F", "c": "F", "d": "F"}
        tm = {"a": "1", "b": "1", "c": "2", "d": "2"}
        pm = {"a": "1", "c": "1", "b": "2", "d": "2"}
        r = metrics.score(tf, pm, tf, tm)
        self.assertLess(r["model_confusion"]["true"]["mcc_raw"], 0)
        self.assertEqual(r["mcc_model"], 0.0)

    def test_matches_sklearn_random(self):
        rng = random.Random(0)
        for _ in range(50):
            pf = {e: rng.choice("ABCD") for e in EPS}
            pm = {e: pf[e] + rng.choice("12") for e in EPS}
            r = metrics.score(pf, pm, TRUE_F, TRUE_M)
            pairs = list(itertools.combinations(sorted(EPS), 2))
            yp = [pf[a] == pf[b] for a, b in pairs]
            yt = [TRUE_F[a] == TRUE_F[b] for a, b in pairs]
            self.assertAlmostEqual(r["mcc_family"], max(0, sk_mcc(yp, yt)), places=12)
            for v, inc in (("true", lambda a, b: TRUE_F[a] == TRUE_F[b]),
                           ("pred", lambda a, b: pf[a] == pf[b]), ("all", lambda a, b: True)):
                sub = [(a, b) for a, b in pairs if inc(a, b)]
                yp = [pm[a] == pm[b] for a, b in sub]
                yt = [TRUE_M[a] == TRUE_M[b] for a, b in sub]
                self.assertAlmostEqual(r["mcc_model_variants"][v], max(0, sk_mcc(yp, yt)), places=12)
            self.assertAlmostEqual(r["score"], 0.1 * r["mcc_family"] + 0.9 * r["mcc_model"])

    def test_false_pairs_listed(self):
        m = dict(TRUE_M)
        m["ep-01"] = "fam0m1"  # ep-01 moved to the second model of family 0
        r = metrics.score(TRUE_F, m, TRUE_F, TRUE_M)
        fp = r["model_confusion"]["true"]["false_pairs"]
        kinds = {k for _, _, k in fp}
        self.assertEqual(kinds, {"FP", "FN"})
        self.assertTrue(all("ep-01" in (a, b) for a, b, _ in fp))
        c = r["model_confusion"]["true"]
        self.assertEqual(c["fp"] + c["fn"], len(fp))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write(tmp, "p.csv", csv_text(TRUE_F, TRUE_M))
            t = write(tmp, "t.csv", csv_text(TRUE_F, TRUE_M))
            out = subprocess.run([sys.executable, "-m", "model_equality.scoring.metrics", p, t, "--json"],
                                 cwd=REPO, capture_output=True, text=True, check=True).stdout
            self.assertAlmostEqual(json.loads(out)["score"], 1.0)


def _find_practice():
    for d in (SCORING / "data", REPO / "model_equality" / "starter"):
        if (d / "practice_answers.csv").exists():
            return d
    return None


class TestPracticeScorer(unittest.TestCase):
    """Reproduce the organisers' score_practice.py on practice_answers.csv
    (copies in scoring/data/, fetched from the cluster)."""

    def setUp(self):
        self.d = _find_practice()
        if self.d is None:
            self.skipTest("practice_answers.csv not available locally")
        self.answers = str(self.d / "practice_answers.csv")
        self.truth = csvio.read_partition(self.answers)
        self.scorer = self.d / "score_practice.py"

    def test_practice_answers_read(self):
        self.assertEqual(sorted(self.truth), csvio.PRACTICE_ENDPOINTS)
        self.assertEqual(self.truth["ref-02"], ("P", "T1"))  # 4th 'kind' column ignored
        rows = csvio.read_rows(self.answers)
        self.assertEqual(rows["ref-04"][2], "LoRA at strength 0.7")
        rep = csvio.validate(self.answers, expected_endpoints=csvio.PRACTICE_ENDPOINTS)
        self.assertFalse(rep.ok)  # 4 non-empty columns: not a valid *submission* line
        self.assertTrue(any("expected 3 fields" in e for e in rep.errors))

    def test_truth_self_score(self):
        f = {e: v[0] for e, v in self.truth.items()}
        m = {e: v[1] for e, v in self.truth.items()}
        r = metrics.score(f, m, f, m)
        self.assertAlmostEqual(r["mcc_model"], 1.0)
        self.assertEqual(r["mcc_family"], 0.0)  # one family only in practice -> family MCC undefined -> 0

    def _cases(self):
        eps = sorted(self.truth)
        tf = {e: self.truth[e][0] for e in eps}
        tm = {e: self.truth[e][1] for e in eps}
        yield "truth", csv_text(tf, tm)
        yield "all_same", csv_text({e: "A" for e in eps}, {e: "A1" for e in eps})
        yield "all_diff", csv_text({e: "A" for e in eps}, {e: e for e in eps}, header=False)
        # same model labels but split families -> organisers count those as different models
        yield "fam_split", csv_text({e: ("A" if i % 2 else "B") for i, e in enumerate(eps)}, tm)
        yield "with_ep_lines", "\ufeff" + csv_text(tf, tm) + "ep-01,X,X1\nep-02, X , X1 \n"
        rng = random.Random(7)
        for k in range(40):
            f = {e: rng.choice("PQ") for e in eps}
            m = {e: f[e] + rng.choice(["S", "T1", "T2", "T3"]) for e in eps}
            if k % 3 == 0:  # labels reused across families
                m = {e: rng.choice(["S", "T1"]) for e in eps}
            yield f"rand{k}", csv_text(f, m, header=bool(k % 2))

    def test_against_provided_scorer(self):
        if not self.scorer.exists():
            self.skipTest("score_practice.py not available locally")
        eps = sorted(self.truth)
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in self._cases():
                pred = write(tmp, f"{name}.csv", text)
                res = subprocess.run([sys.executable, str(self.scorer), pred, self.answers],
                                     capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stderr)
                ours = metrics.practice_report(pred, self.answers)
                self.assertEqual(ours, res.stdout.rstrip("\n"), name)
                cli = subprocess.run([sys.executable, "-m", "model_equality.scoring.metrics", pred,
                                      self.answers, "--practice-format"], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout
                self.assertEqual(cli, res.stdout, name)
                # the general scorer gives the same model MCC (one family -> all variants agree)
                p = csvio.read_partition(pred, prefix="ref-")
                r = metrics.score({e: p[e][0] for e in eps}, {e: p[e][1] for e in eps},
                                  {e: self.truth[e][0] for e in eps}, {e: self.truth[e][1] for e in eps})
                printed = float(re.search(r": (\d\.\d{3})$", res.stdout.splitlines()[0]).group(1))
                self.assertEqual(f"{r['model_confusion']['all']['mcc']:.3f}", f"{printed:.3f}", name)
                self.assertAlmostEqual(r["mcc_model_variants"]["true"], r["mcc_model_variants"]["all"])

    def test_missing_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = write(tmp, "p.csv", "ref-01,P,S\n")
            with self.assertRaisesRegex(ValueError, "no line for: ref-02"):
                metrics.practice_report(pred, self.answers)


class TestValidator(unittest.TestCase):
    def v(self, text, **kw):
        return csvio.validate(text, is_text=True, **kw)

    def good(self, header=True):
        return csv_text(TRUE_F, TRUE_M, header)

    def test_valid_with_and_without_header(self):
        for h in (True, False):
            r = self.v(self.good(h))
            self.assertTrue(r.ok, r.format())
            self.assertEqual(r.has_header, h)
            self.assertEqual(len(r.rows), 33)

    def test_order_free(self):
        lines = self.good(False).strip().split("\n")
        random.Random(3).shuffle(lines)
        self.assertTrue(self.v("\n".join(lines) + "\n").ok)

    def test_missing_endpoint(self):
        r = self.v(self.good().replace("ep-07,", "#ep-07,", 1).replace("#ep-07", "ep-99"))
        self.assertFalse(r.ok)
        self.assertTrue(any("missing endpoints" in e and "ep-07" in e for e in r.errors))
        self.assertTrue(any("ep-99" in e and "out of range" in e for e in r.errors))

    def test_ref_refused(self):
        r = self.v(self.good() + "ref-01,A,A1\n")
        self.assertFalse(r.ok)
        self.assertTrue(any("ref-01" in e and "practice" in e for e in r.errors))

    def test_duplicate(self):
        r = self.v(self.good() + f"ep-01,{TRUE_F['ep-01']},{TRUE_M['ep-01']}\n")
        self.assertTrue(any("duplicate endpoint ep-01" in e for e in r.errors))

    def test_comma_in_label(self):
        line = f"ep-05,{TRUE_F['ep-05']},{TRUE_M['ep-05']}"
        self.assertIn(line, self.good())
        r = self.v(self.good().replace(line, f"ep-05,{TRUE_F['ep-05']},x,y"))
        self.assertTrue(any("line" in e and "commas" in e for e in r.errors), r.errors)

    def test_model_in_two_families(self):
        txt = self.good().replace(f"ep-01,{TRUE_F['ep-01']},", f"ep-01,famX,")
        r = self.v(txt)
        self.assertTrue(any("same model requires same family" in e for e in r.errors), r.errors)

    def test_empty_and_whitespace_labels(self):
        r = self.v(self.good().replace("ep-02,fam0,fam0m0", "ep-02,,fam0m0"))
        self.assertTrue(any("empty family" in e for e in r.errors))
        # padding spaces are stripped by the organisers' reader: warning only
        r = self.v(self.good().replace("ep-02,fam0,fam0m0", "ep-02, fam0 ,fam0m0"))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.rows["ep-02"], ("fam0", "fam0m0"))
        self.assertTrue(any("spaces" in w for w in r.warnings))

    def test_wrong_field_count_and_blank_line(self):
        r = self.v(self.good().replace("ep-03,fam0,fam0m0\n", "ep-03,fam0\n\n"))
        self.assertTrue(any("expected 3" in e for e in r.errors))
        r = self.v(self.good().replace("ep-03,fam0,fam0m0\n", "ep-03,fam0,fam0m0\n\n  \n"))
        self.assertTrue(r.ok, r.errors)
        self.assertTrue(any("blank" in w for w in r.warnings))

    def test_tolerated_forms(self):
        for text in (("\ufeff" + self.good()).encode("utf-8"),            # BOM
                     self.good().replace("\n", "\r\n"),                     # CRLF
                     self.good().replace("ep-04,fam0,fam0m1", "ep-04,fam0,fam0m1,"),  # trailing comma
                     self.good().replace("ep-04,fam0,fam0m1", '"ep-04","fam0","fam0m1"')):  # quoting
            r = self.v(text)
            self.assertTrue(r.ok, r.errors)
            self.assertEqual(r.rows["ep-04"], ("fam0", "fam0m1"))

    def test_case_and_header(self):
        r = self.v(self.good().replace("ep-04,", "EP-04,"))
        self.assertTrue(any("lower case" in e for e in r.errors))
        r = self.v(self.good().replace("endpoint,family,model", "Endpoint,Family,Model"))
        self.assertTrue(any("header" in e for e in r.errors))
        r = self.v(self.good().replace("ep-04,fam0,fam0m1", 'ep-04,fam0,"fam0,m1"'))
        self.assertTrue(any("comma" in e for e in r.errors))

    def test_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write(tmp, "sub.txt", self.good())
            self.assertTrue(any(".csv" in e for e in csvio.validate(p).errors))

    def test_practice_mode(self):
        f = {e: "A" if i < 5 else "B" for i, e in enumerate(csvio.PRACTICE_ENDPOINTS)}
        m = {e: f[e] + "1" for e in f}
        self.assertTrue(self.v(csv_text(f, m), expected_endpoints=csvio.PRACTICE_ENDPOINTS).ok)

    def test_write_submission_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.csv")
            # model labels reused across families -> writer must disambiguate
            m = {e: "m0" for e in EPS}
            csvio.write_submission(p, TRUE_F, m)
            self.assertTrue(csvio.validate(p).ok)
            rows = csvio.read_partition(p)
            r = metrics.score({e: v[0] for e, v in rows.items()}, {e: v[1] for e, v in rows.items()},
                              TRUE_F, TRUE_M)
            self.assertAlmostEqual(r["mcc_family"], 1.0)


class TestPartition(unittest.TestCase):
    def truth_matrix(self, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        n = len(EPS)
        S = np.array([[1.0 if TRUE_M[a] == TRUE_M[b] else (0.4 if TRUE_F[a] == TRUE_F[b] else 0.0)
                       for b in EPS] for a in EPS])
        F = np.array([[1.0 if TRUE_F[a] == TRUE_F[b] else 0.0 for b in EPS] for a in EPS])
        if noise:
            E = rng.uniform(-noise, noise, (n, n))
            S = np.clip(S + (E + E.T) / 2, 0, 1)
        return S, F

    def test_methods_recover_truth(self):
        S, F = self.truth_matrix(noise=0.2)
        for method in partition.METHODS:
            for kw in ({"family_sim": F}, {"families": TRUE_F}):
                f, m = partition.partition(EPS, S, 0.7, method=method, **kw)
                r = metrics.score(f, m, TRUE_F, TRUE_M)
                self.assertAlmostEqual(r["score"], 1.0, msg=f"{method} {list(kw)}")

    def test_transitive_and_consistent(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            S = rng.uniform(0, 1, (33, 33))
            Fm = rng.uniform(0, 1, (33, 33))
            for method in partition.METHODS:
                f, m = partition.partition(EPS, S, 0.6, method=method, family_sim=Fm, family_threshold=0.5)
                for a, b in itertools.combinations(EPS, 2):
                    if m[a] == m[b]:
                        self.assertEqual(f[a], f[b])
                with tempfile.TemporaryDirectory() as tmp:
                    rep = csvio.write_submission(os.path.join(tmp, "x.csv"), f, m)
                    self.assertTrue(rep.ok)

    def test_components_chain_vs_complete(self):
        # chain a-b-c with a,c dissimilar: components merge all, complete linkage does not
        S = np.array([[1, .9, .1], [.9, 1, .9], [.1, .9, 1.]])
        self.assertEqual(len(set(partition.cluster(S, 0.5, "components"))), 1)
        self.assertEqual(len(set(partition.cluster(S, 0.5, "hierarchical", "complete"))), 2)

    def test_threshold_extremes(self):
        S, _ = self.truth_matrix()
        self.assertEqual(len(set(partition.cluster(S, 1.01, "components"))), 33)
        self.assertEqual(len(set(partition.cluster(S, -0.01, "hierarchical"))), 1)

    def test_cli_pairs_to_csv(self):
        S, F = self.truth_matrix(noise=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            pairs = os.path.join(tmp, "pairs.csv")
            with open(pairs, "w") as fh:
                fh.write("a,b,p_model,p_family\n")
                for i, j in itertools.combinations(range(33), 2):
                    fh.write(f"{EPS[i]},{EPS[j]},{S[i, j]:.4f},{F[i, j]:.4f}\n")
            out = os.path.join(tmp, "sub.csv")
            subprocess.run([sys.executable, "-m", "model_equality.scoring.partition", "--pairs", pairs,
                            "--threshold", "0.7", "--out", out], cwd=REPO, check=True, capture_output=True)
            r = metrics.score_files(out, write(tmp, "t.csv", csv_text(TRUE_F, TRUE_M)))
            self.assertAlmostEqual(r["score"], 1.0)

    def test_sweep(self):
        S, F = self.truth_matrix(noise=0.1)
        res = partition.sweep(EPS, S, {e: (TRUE_F[e], TRUE_M[e]) for e in EPS}, [0.2, 0.7], family_sim=F)
        self.assertLess(res[0][1]["score"], res[1][1]["score"])


class FakeResp:
    status_code = 200
    text = json.dumps({"status": "success", "score": 0.42, "previous_best": 0.3, "improved": True,
                       "message": "ok"})

    def json(self):
        return json.loads(self.text)


class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "subs"
        self.csv = write(self.tmp.name, "sub.csv", csv_text(TRUE_F, TRUE_M))
        self.env = mock.patch.dict(os.environ, {"MODEL_EQUALITY_API_KEY": "SECRET-TEST-KEY"})
        self.env.start()
        self.dotenv = mock.patch.object(submit, "load_dotenv", lambda *a, **k: None)
        self.dotenv.start()
        self.calls = []

    def tearDown(self):
        self.env.stop()
        self.dotenv.stop()
        self.tmp.cleanup()

    def poster(self, url, headers, files, timeout):
        self.calls.append((url, headers, files, timeout))
        return FakeResp()

    def run_submit(self, confirm, now, csv=None):
        buf = io.StringIO()
        rc = submit.submit(csv or self.csv, confirm, log_dir=self.dir, poster=self.poster, now=now,
                           out=lambda s: buf.write(s + "\n"))
        return rc, buf.getvalue()

    def test_dry_run_sends_nothing(self):
        rc, out = self.run_submit(False, dt.datetime(2026, 11, 21, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [])
        self.assertIn("DRY RUN", out)
        self.assertNotIn("SECRET-TEST-KEY", out)
        self.assertFalse((self.dir / "log.jsonl").exists())

    def test_invalid_csv_refused(self):
        bad = write(self.tmp.name, "bad.csv", "ref-01,A,A1\n")
        rc, _ = self.run_submit(True, dt.datetime(2026, 11, 21, 12, tzinfo=dt.timezone.utc), bad)
        self.assertEqual(rc, 1)
        self.assertEqual(self.calls, [])

    def test_no_key_refused(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            rc, out = self.run_submit(True, dt.datetime(2026, 11, 21, 12, tzinfo=dt.timezone.utc))
        self.assertEqual(rc, 3)
        self.assertEqual(self.calls, [])

    def test_fallback_key_var(self):
        with mock.patch.dict(os.environ, {"API_KEY_BOARD": "BOARD-KEY"}, clear=True):
            self.assertEqual(submit.get_api_key(), ("API_KEY_BOARD", "BOARD-KEY"))
        with mock.patch.dict(os.environ, {"API_KEY_BOARD": "B", "MODEL_EQUALITY_API_KEY": "M"}, clear=True):
            self.assertEqual(submit.get_api_key()[0], "MODEL_EQUALITY_API_KEY")

    def test_submit_logs_and_30min_guard(self):
        t0 = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)
        rc, out = self.run_submit(True, t0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(self.calls), 1)
        url, headers, files, timeout = self.calls[0]
        self.assertEqual(url, submit.URL)
        self.assertEqual(headers, {"X-API-Key": "SECRET-TEST-KEY"})
        self.assertIn("file", files)
        self.assertGreaterEqual(timeout[1], 500)
        self.assertNotIn("SECRET-TEST-KEY", out)
        log = submit.read_log(self.dir / "log.jsonl")
        self.assertEqual([e["kind"] for e in log], ["attempt", "result"])
        self.assertEqual(log[1]["score"], 0.42)
        self.assertTrue(log[1]["improved"])
        copy = self.dir / log[0]["copy"]
        self.assertEqual(copy.read_bytes(), Path(self.csv).read_bytes())
        self.assertNotIn("SECRET-TEST-KEY", (self.dir / "log.jsonl").read_text())

        # 29 min later: refused, nothing sent, log unchanged
        rc, out = self.run_submit(True, t0 + dt.timedelta(minutes=29, seconds=59))
        self.assertEqual(rc, 2)
        self.assertIn("30-minute", out)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(submit.read_log(self.dir / "log.jsonl")), 2)
        # dry run is also refused inside the window
        rc, _ = self.run_submit(False, t0 + dt.timedelta(minutes=10))
        self.assertEqual(rc, 2)
        # 30 min later: allowed, log is appended (not rewritten)
        rc, _ = self.run_submit(True, t0 + dt.timedelta(minutes=30))
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(submit.read_log(self.dir / "log.jsonl")), 4)

    def test_rejected_and_failed_responses(self):
        t = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)
        cases = ((422, {"detail": "bad file"}), (200, {"status": "error", "message": "x"}),
                 (200, {"status": "success", "score": 0.5, "message": "first"}))  # no previous_best
        for i, (code, body) in enumerate(cases):
            class R(FakeResp):
                status_code = code
                text = json.dumps(body)
            buf = io.StringIO()
            rc = submit.submit(self.csv, True, log_dir=self.dir, poster=lambda *a, **k: R(),
                               now=t + dt.timedelta(hours=i), out=lambda s: buf.write(s + "\n"))
            self.assertEqual(rc, 4 if i < 2 else 0, buf.getvalue())
        log = submit.read_log(self.dir / "log.jsonl")
        self.assertEqual(log[1]["detail"], "bad file")
        self.assertEqual(log[3]["status"], "error")
        self.assertEqual(log[5]["score"], 0.5)
        self.assertNotIn("previous_best", log[5])

    def test_network_error_still_blocks(self):
        def boom(*a, **k):
            raise TimeoutError("read timed out")
        t0 = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)
        buf = io.StringIO()
        rc = submit.submit(self.csv, True, log_dir=self.dir, poster=boom, now=t0, out=lambda s: buf.write(s))
        self.assertEqual(rc, 4)
        ok, _ = submit.guard(self.dir / "log.jsonl", t0 + dt.timedelta(minutes=5))
        self.assertFalse(ok)

    def test_dotenv_does_not_override(self):
        envf = Path(self.tmp.name) / ".env"
        envf.write_text("MODEL_EQUALITY_API_KEY=from_file\nOTHER='x y'\n")
        self.dotenv.stop()
        try:
            with mock.patch.dict(os.environ, {"MODEL_EQUALITY_API_KEY": "from_env"}, clear=True):
                submit.load_dotenv(envf)
                self.assertEqual(os.environ["MODEL_EQUALITY_API_KEY"], "from_env")
                self.assertEqual(os.environ["OTHER"], "x y")
        finally:
            self.dotenv.start()


if __name__ == "__main__":
    unittest.main()
