"""Command line for the Model Equality analysis (read-only on the ledger; no API call).

    python3.12 -m model_equality.analysis extract  [--endpoints 'ref-*'] [--out-dir DIR]
    python3.12 -m model_equality.analysis pairs    [--endpoints 'ref-*'] [--units SPEC] [--B 500] [--jobs N]
    python3.12 -m model_equality.analysis select   [--n-sub 12] [--draws 20] [--B 200] [--jobs N]
    python3.12 -m model_equality.analysis simulate [--selected DIR/selected.txt | --battery P:n,...] [--draws 10]
    python3.12 -m model_equality.analysis decide   --endpoints 'ep-*' --families FAM.csv [--pairs DIR/pairs.csv]

decide is the official path to the deliverable CSV (it validates with
scoring.csvio.write_submission); do not feed pairs.csv (z-scores) to
scoring/partition.py, which expects probabilities p_model / p_family.

Common options: --ledger (default model_equality/api/ledger/ledger.jsonl),
--probes (default model_equality/probes/probes.jsonl), --out-dir (default
model_equality/analysis/out/<command>), --seed, --min-cell.

--tree K: add for every probe a virtual probe '<probe>@tree' (category T_tree) whose events are
(exact prefix of the first j word tokens -> token j), j < K: exact repeated contexts for any probe type,
free text included.  A view and its base probe are the same responses (do not count both in a budget).

--units SPEC: 'all' (default: every unit with data), 'discrete', a comma list
of unit or probe ids, categories with 'cat:K_single,S_stream', or
'file:PATH' (one unit id per line, e.g. select's selected.txt).
Every command writes CSV tables and a summary.txt in --out-dir and prints the summary.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from .data import (DEFAULT_LEDGER, DEFAULT_PROBES, DEFAULT_TRUTH, add_tree_views, load_dataset, load_probes,
                   read_truth, units_of)
from .extract import conformity_rows, parse_dataset
from .pairs import compute_pairs, default_jobs
from .stats import MIN_CELL

OUT_ROOT = Path(__file__).resolve().parent / "out"


# ----------------------------------------------------------------- helpers
def _units(spec: str | None, ds, parsed) -> dict:
    present = {p for d in parsed.values() for p in d}
    allu = units_of(ds.probes, sorted(present))
    if spec in (None, "", "all"):
        return allu
    if spec == "discrete":
        return {u: ps for u, ps in allu.items() if ds.probes[ps[0]].discrete}
    if spec.startswith("cat:"):
        cats = set(spec[4:].split(","))
        return {u: ps for u, ps in allu.items() if ds.probes[ps[0]].category in cats}
    if spec.startswith("file:"):
        ids = [l.split()[0] for l in Path(spec[5:]).read_text().splitlines() if l.strip() and not l.startswith("#")]
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    out = {}
    for i in ids:
        if i in allu:
            out[i] = allu[i]
        elif i in ds.probes and ds.probes[i].unit in allu:
            u = ds.probes[i].unit
            out.setdefault(u, [])
            if i in allu[u] and i not in out[u]:
                out[u].append(i)
    return out


def _out(a, name) -> Path:
    d = Path(a.out_dir) if a.out_dir else OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(rows, path: Path, cols=None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if cols is not None and not df.empty:
        df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    df.to_csv(path, index=False, float_format="%.4g")
    return df


def _finish(lines, out: Path):
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[written to {out}]")


def _load(a):
    probes = load_probes(a.probes)
    ds = load_dataset(a.ledger, probes, a.endpoints)
    if getattr(a, "tree", 0):
        add_tree_views(ds, a.tree)
    parsed = parse_dataset(ds)
    return ds, parsed


def _fmt(v, nd=2):
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def _matrix(rows, eps, key) -> str:
    M = pd.DataFrame(index=list(eps), columns=list(eps), dtype=object)
    for r in rows:
        v = r.get(key)
        s = _fmt(v, 1) if isinstance(v, float) else str(v)
        M.loc[r["a"], r["b"]] = s
        M.loc[r["b"], r["a"]] = s
    return M.fillna("·").to_string()


# ----------------------------------------------------------------- commands
def cmd_extract(a) -> int:
    ds, parsed = _load(a)
    out = _out(a, "extract")
    rows = conformity_rows(ds, parsed)
    df = _write(rows, out / "conformity.csv")
    lines = [f"ledger: {a.ledger}", f"endpoints: {len(ds.endpoints)} ({', '.join(ds.endpoints)})",
             f"probe responses: {sum(len(v) for d in ds.responses.values() for v in d.values())}; "
             f"non-probe responses: {sum(len(v) for d in ds.other.values() for v in d.values())} "
             f"({', '.join(sorted({p for d in ds.other.values() for p in d}))})"]
    if df.empty:
        lines.append("no probe response yet")
        _finish(lines, out)
        return 0
    g = df.groupby(["probe", "parser"]).agg(n=("n_responses", "sum"), parsable=("n_parsable", "sum"),
                                            fmt=("mean_fmt", "mean"), draws=("mean_draws", "mean"),
                                            worst=("parsable_rate", "min")).reset_index()
    g["rate"] = g["parsable"] / g["n"]
    g.to_csv(out / "conformity_by_probe.csv", index=False, float_format="%.4g")
    piv = df.pivot_table(index="probe", columns="endpoint", values="parsable_rate")
    piv.to_csv(out / "conformity_matrix.csv", float_format="%.3g")
    lines.append("")
    lines.append("per probe: responses, parsable rate (all endpoints), worst endpoint rate, mean format share, mean draws/response")
    for r in g.sort_values(["rate", "probe"]).itertuples():
        lines.append(f"  {r.probe:22s} {r.parser:11s} n={r.n:4d} parsable={r.rate:5.2f} worst={r.worst:5.2f} "
                     f"fmt={r.fmt:5.2f} draws={r.draws:6.1f}")
    _finish(lines, out)
    return 0


def cmd_pairs(a) -> int:
    ds, parsed = _load(a)
    out = _out(a, "pairs")
    units = _units(a.units, ds, parsed)
    eps = ds.endpoints
    rows, urows = compute_pairs(parsed, ds.probes, eps, units, B=a.B, seed=a.seed, jobs=a.jobs,
                                n_boot=a.n_boot, min_cell=a.min_cell)
    cols = ["a", "b", "z_raw", "z_temp", "z_inv", "kappa_temp", "kappa_inv", "z_temp_short", "z_temp_long",
            "z_mmd", "cells_temp", "log10_splits", "low_power", "wm_sig", "wm_delta", "lnk_lo", "lnk_hi"]
    _write(rows, out / "pairs.csv", cols)
    _write(urows, out / "pairs_units.csv", ["a", "b", "unit", "z_raw", "z_temp", "z_inv", "z_mmd", "kappa_temp"])
    lines = [f"endpoints: {len(eps)}; units: {len(units)} ({', '.join(units)}); B={a.B}; min_cell={a.min_cell}",
             f"pairs: {len(rows)}; low power (<100 distinct splits): {sum(bool(r.get('low_power')) for r in rows)}",
             "", "z_temp (aggregated over the units):", _matrix(rows, eps, "z_temp"),
             "", "z_inv:", _matrix(rows, eps, "z_inv"),
             "", "z_raw (G tests):", _matrix(rows, eps, "z_raw"),
             "", "z_mmd (free probes):", _matrix(rows, eps, "z_mmd"),
             "", "kappa_temp (Theta_b ~ kappa Theta_a):", _matrix(rows, eps, "kappa_temp"),
             "", "watermark signature:", _matrix(rows, eps, "wm_sig")]
    _finish(lines, out)
    return 0


def cmd_select(a) -> int:
    from .select import DEFAULT_HARD, run_selection
    ds, parsed = _load(a)
    out = _out(a, "select")
    truth = read_truth(a.truth)
    units = _units(a.units, ds, parsed)
    hard = DEFAULT_HARD if not a.hard else [tuple(p.split("/")) for p in a.hard.split(",")]
    ranked, draws = run_selection(ds, parsed, truth, units, n_sub=a.n_sub, draws=a.draws, B=a.B, seed=a.seed,
                                  jobs=a.jobs, hard=hard, min_cell=a.min_cell)
    _write(ranked, out / "select_ranked.csv")
    _write(draws, out / "select_draws.csv")
    kept = [r for r in ranked if r["kept"]]
    with open(out / "selected.txt", "w") as fh:
        fh.write("# unit  probes  (kept by the R2 criterion; safe_except=SP means: judge SP endpoints on long probes)\n")
        for r in kept:
            fh.write(f"{r['unit']}  {r['probes']}{'  safe_except=SP' if r['safe_except'] else ''}\n")
    lines = [f"units: {len(units)}; draws: {a.draws}; n_sub: {a.n_sub}; B: {a.B}; hard pairs: {hard}",
             f"kept: {len(kept)} -> {' '.join(r['unit'] for r in kept) or '(none)'}", "",
             f"{'unit':20s} {'kept':5s} conf(med/min) safe(minfrac,fails)    sens(max,pair)          "
             f"zmed_hard ctrl  resp lowpow"]
    for r in ranked:
        lines.append(f"{r['unit']:20s} {str(r['kept']):5s} {_fmt(r['conf_median'])}/{_fmt(r['conf_min'])} "
                     f"{_fmt(r['safe_min_frac'])},{r['safe_fail_lures'] or '-':10s}   "
                     f"{_fmt(r['sens_max'])},{r['sens_pair'] or '-':15s} {_fmt(r['zmed_hard']):>6s}  "
                     f"{_fmt(r['control_mean_z']):>5s} {r['resp_per_ep']:4.0f} {r['low_power_share']:.2f}")
    if any(r["low_power_share"] > 0.5 for r in ranked):
        lines.append("\nWARNING: low power (fewer than 100 distinct permutations) for most draws of some units: "
                     "z is coarse, collect more responses before trusting 'sens' and 'safe'.")
    _finish(lines, out)
    return 0


def cmd_simulate(a) -> int:
    from .simulate import default_battery, parse_battery, run_simulation
    ds, parsed = _load(a)
    out = _out(a, "simulate")
    truth = read_truth(a.truth)
    if a.battery:
        battery = parse_battery(a.battery)
        units = units_of(ds.probes, battery)
    else:
        spec = f"file:{a.selected}" if a.selected else a.units
        units = _units(spec, ds, parsed)
        units = {u: ps for u, ps in units.items() if ds.probes[ps[0]].discrete}
        battery = default_battery(units, ds.probes, a.budget)
    if not battery:
        print("empty battery (no selected unit with data)", file=sys.stderr)
        return 2
    rows, summ = run_simulation(ds, parsed, truth, units, battery, draws=a.draws, B=a.B, seed=a.seed, jobs=a.jobs,
                                n_boot=a.n_boot, pi_fp8=a.pi_fp8, min_cell=a.min_cell,
                                forced_labels=read_forced_labels(a.force_labels))
    _write(rows, out / "simulate_draws.csv")
    lines = [f"battery ({sum(battery.values())} responses per endpoint): " +
             ", ".join(f"{p}:{n}" for p, n in battery.items())]
    lines += [f"{k}: {_fmt(v, 3)}" for k, v in summ.items()]
    lines.append("")
    for r in rows:
        lines.append(f"draw {r['draw']}: MCC_model={r['mcc_model']:.3f} base={r['base']} groups=[{r['groups']}] "
                     f"wrong={r['wrong_pairs'] or '-'}")
    if summ["overlap"]:
        lines.append("WARNING: stock too small for disjoint draws; draws overlap (MCC spread underestimated).")
    _finish(lines, out)
    return 0


def read_forced_labels(path: str | None) -> dict | None:
    """CSV ``endpoint,label`` (header optional; further columns ignored); labels DET / WM / SP / TEMP / NONE."""
    if not path:
        return None
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            r = [c.strip() for c in r]
            if len(r) < 2 or not r[0] or r[0].lower() == "endpoint" or r[0].startswith("#"):
                continue
            if r[1].upper() not in ("DET", "WM", "SP", "TEMP", "NONE", "BASE", "VAR"):
                raise ValueError(f"{path}: unknown label {r[1]!r} for {r[0]}")
            out[r[0]] = r[1].upper()
    return out


VERDICT_LABEL = {"BASE_OR_LURE": "BASE", "FP8": "VAR", "VARIANT": "VAR"}


def merge_verdicts(forced: dict | None, path: str | None) -> tuple:
    """White-box verdicts (CSV endpoint,verdict[,...]; BASE_OR_LURE / FP8 / VARIANT / UNSURE) on top of the hand
    labels.  BASE_OR_LURE -> BASE unless a lure label is already set; FP8 / VARIANT -> VAR and it overrides a
    lure label (a lure may sit on a variant); UNSURE and unknown words change nothing.  Returns (labels, notes)."""
    forced = dict(forced or {})
    notes = []
    if not path:
        return forced, notes
    import csv
    col, seen = 1, {}   # verdict column: 1 unless a header names it (wb_verdicts_ep.csv has it in 8th position)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            r = [c.strip() for c in r]
            if r and r[0].lower() == "endpoint":
                low = [c.lower() for c in r]
                col = low.index("verdict") if "verdict" in low else 1
                continue
            if len(r) <= col or not r[0] or r[0].startswith("#"):
                continue
            word = r[col].upper()
            seen[word] = seen.get(word, 0) + 1
            lab = VERDICT_LABEL.get(word)
            if lab == "BASE" and forced.get(r[0]) in (None, "NONE"):
                forced[r[0]] = "BASE"
            elif lab == "VAR":
                if forced.get(r[0]) in ("DET", "WM", "SP", "TEMP"):
                    notes.append(f"{r[0]}: verdict {r[col]} overrides the lure label {forced[r[0]]} (kept alone)")
                forced[r[0]] = "VAR"
    unknown = sorted(w for w in seen if w not in VERDICT_LABEL and w != "UNSURE")
    # INFO, not NOTE: a NOTE line means "a lure label is contradicted by the white box"
    print("INFO: verdicts read (column %d): %s" % (col + 1, ", ".join(f"{w} x{n}" for w, n in sorted(seen.items()))
                                                   or "none"), file=sys.stderr)
    if unknown:
        print("INFO: unknown verdict words ignored: " + ", ".join(unknown), file=sys.stderr)
    return forced, notes


def read_families(path: str, eps: list) -> dict:
    """Hand-validated family file.

    Format: UTF-8 CSV, header ``endpoint,family`` (a third ``model`` column and
    further columns are allowed and ignored), one line per endpoint.  Checked:
    every endpoint of the run appears exactly once, no empty family label, no
    line with fewer than 2 cells, no unknown endpoint.  Raises ValueError.
    """
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows or [c.lower() for c in rows[0][:2]] != ["endpoint", "family"]:
        raise ValueError(f"{path}: first line must be the header 'endpoint,family[,model]'")
    out, errors = {}, []
    for k, r in enumerate(rows[1:], start=2):
        if len(r) < 2 or not r[0] or not r[1]:
            errors.append(f"line {k}: expected 'endpoint,family' with non-empty cells, got {r}")
            continue
        if r[0] in out:
            errors.append(f"line {k}: duplicate endpoint {r[0]}")
            continue
        if r[0] not in eps:
            errors.append(f"line {k}: endpoint {r[0]} not in this run ({len(eps)} endpoints)")
            continue
        out[r[0]] = r[1]
    missing = [e for e in eps if e not in out]
    if missing:
        errors.append(f"missing endpoints: {', '.join(missing)}")
    if errors:
        raise ValueError(f"{path}: invalid families file:\n  " + "\n  ".join(errors))
    return out


def cmd_decide(a) -> int:
    """Exit codes: 0 ok; 2 bad input; 3 partition invalid (no CSV written);
    4 family inference collapsed to one family without --allow-one-family."""
    import datetime
    import shutil
    from model_equality.scoring.csvio import SUBMISSION_ENDPOINTS, write_submission
    from .decision import decide
    from .families import cluster_families
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ds, parsed = _load(a)
    out = _out(a, "decide")
    sub_path = Path(a.out) if a.out else out / "partition.csv"
    if sub_path.exists():   # never leave a stale partition next to new outputs (audit point 2)
        stale = sub_path.with_name(f"{sub_path.stem}.stale-{ts}{sub_path.suffix}")
        sub_path.rename(stale)
        print(f"previous {sub_path.name} moved to {stale.name}", file=sys.stderr)
    eps = ds.endpoints
    units = _units(a.units, ds, parsed)
    if a.families == "one":
        fams, finfo = {e: "F" for e in eps}, {"reason": "one family (--families one, explicit)"}
    elif a.families == "auto":
        fams, finfo = cluster_families(ds, eps)
        if len(set(fams.values())) == 1 and len(eps) > 1 and not a.allow_one_family:
            print(f"ERROR: family inference collapsed to a single family ({finfo.get('reason')}). "
                  f"Give a hand-validated --families FILE.csv (header endpoint,family), or pass "
                  f"--allow-one-family / --families one if a single family is really intended.", file=sys.stderr)
            return 4
    else:
        try:
            fams = read_families(a.families, eps)
        except (OSError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        finfo = {"reason": f"from {a.families}"}
    forced, vnotes = merge_verdicts(read_forced_labels(a.force_labels), a.verdicts)
    for n in vnotes:
        print("NOTE: " + n, file=sys.stderr)
    if a.labels_only:
        rows = []          # the partition comes from the labels: no pair statistic is computed
    elif a.pairs:
        df = pd.read_csv(a.pairs)
        rows = [r for r in df.to_dict("records") if r["a"] in eps and r["b"] in eps]
    else:
        # only within-family pairs are needed
        import itertools
        need = [(x, y) for x, y in itertools.combinations(eps, 2) if fams[x] == fams[y]]
        rows, _ = compute_pairs(parsed, ds.probes, eps, units, B=a.B, seed=a.seed, jobs=a.jobs, n_boot=a.n_boot,
                                min_cell=a.min_cell, per_unit=False, pairs=need)
        _write(rows, out / f"pairs_{ts}.csv")
    res = decide(eps, rows, ds, parsed, families=fams, pi_fp8=a.pi_fp8, prior=a.prior,
                 forced_labels=forced, no_data=a.no_data, labels_only=a.labels_only)
    labels, prow, gray, no_data = [], [], [], []
    for f, r in res["families"].items():
        no_data += r["no_data"]
        for e in r["endpoints"]:
            li = dict(r["labels"][e])
            li["criteria"] = ",".join(li["criteria"])
            li["final_label"] = r["label"][e]
            li["is_base"] = e == r["base"]
            li["family_in"] = f
            li["no_data"] = e in r["no_data"]
            labels.append(li)
        prow += r["pairs"]
        gray += r["gray"]
    _write(labels, out / "labels.csv", ["endpoint", "final_label", "is_base", "criteria", "ref", "no_data"])
    _write(prow, out / "decision_pairs.csv")
    _write(gray, out / "gray_pairs.csv", ["priority", "a", "b", "z_star", "label_a", "label_b", "same_group", "action"])
    # strict deliverable validation as soon as ep- endpoints are involved
    if any(e.startswith("ep-") for e in eps):
        expected = None
    else:
        expected = eps
    rc = 0
    try:
        write_submission(str(sub_path), res["family"], res["model"], expected_endpoints=expected)
        shutil.copyfile(sub_path, out / f"partition_{ts}.csv")
        wrote = f"partition written: {sub_path} (copy: partition_{ts}.csv)"
        if expected is None and sorted(eps) != SUBMISSION_ENDPOINTS:
            wrote += " WARNING: endpoint set differs from ep-01..ep-33"
    except ValueError as e:
        draft = out / f"partition_draft_{ts}.csv"
        pd.DataFrame({"endpoint": sorted(res["family"]), "family": [res["family"][x] for x in sorted(res["family"])],
                      "model": [res["model"][x] for x in sorted(res["family"])]}).to_csv(draft, index=False)
        wrote = f"ERROR: partition NOT written, invalid for the deliverable ({e}); draft only: {draft.name}"
        rc = 3
    lines = [f"run {ts}; endpoints: {len(eps)}; families: {finfo.get('reason', 'auto clustering')} "
             f"({len(set(fams.values()))} families)", wrote, ""]
    if no_data:
        lines.append(f"WARNING: no usable pair statistic for {', '.join(sorted(no_data))}: kept alone (NO_DATA).")
    for f, r in res["families"].items():
        lines.append(f"family {f}: base candidate = {r['base']}; reference = {r['ref']}")
        for g in r["groups"]:
            lines.append("   model group: " + " ".join(f"{e}[{r['label'][e]}]" for e in g))
        lines.append("   gray pairs (reserve priority 1 = first):")
        for gp in r["gray"]:
            lines.append(f"     p{gp['priority']} {gp['a']}-{gp['b']} z*={_fmt(gp['z_star'])} "
                         f"({gp['label_a']}/{gp['label_b']}) same_group={gp['same_group']}")
        if not r["gray"]:
            lines.append("     none")
    lp = sum(bool(p.get("low_power")) for p in prow)
    if lp:
        lines.append(f"\nWARNING: {lp} pairs have low power (<100 distinct permutations): add responses.")
    if a.truth:
        t = read_truth(a.truth)
        if all(e in t for e in eps):
            from model_equality.scoring.metrics import format_report, score
            sc = score(res["family"], res["model"], {e: t[e][0] for e in eps}, {e: t[e][1] for e in eps})
            lines += ["", "score against truth:", format_report(sc)]
    _finish(lines, out)
    if rc:
        print(wrote, file=sys.stderr)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3.12 -m model_equality.analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, endpoints="ref-*"):
        p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
        p.add_argument("--probes", default=str(DEFAULT_PROBES))
        p.add_argument("--endpoints", default=endpoints, help="e.g. 'ref-*', 'ep-01..ep-33', 'ref-01,ref-02'")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--units", default=None, help="unit selection (see module doc)")
        p.add_argument("--min-cell", type=int, default=MIN_CELL)
        p.add_argument("--tree", type=int, default=0, metavar="K",
                       help="add a prefix-tree view '<probe>@tree' of every probe (first K tokens; 0 = off); "
                            "select them with --units cat:T_tree or by name")
        p.add_argument("--jobs", type=int, default=default_jobs())

    p = sub.add_parser("extract", help="parse responses, conformity tables")
    common(p)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("pairs", help="pairwise z_raw / z_temp / z_inv / z_mmd")
    common(p)
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=30, help="bootstrap draws for the ln kappa interval (TEMP label)")
    p.set_defaults(fn=cmd_pairs)
    p = sub.add_parser("select", help="R2 probe selection on ref-")
    common(p)
    p.add_argument("--truth", default=str(DEFAULT_TRUTH))
    p.add_argument("--n-sub", type=int, default=12)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--B", type=int, default=200)
    p.add_argument("--hard", default=None, help="hard pairs 'ref-01/ref-02,ref-01/ref-09'")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("simulate", help="R3 budget simulation on ref-")
    common(p)
    p.add_argument("--truth", default=str(DEFAULT_TRUTH))
    p.add_argument("--selected", default=None, help="selected.txt written by select")
    p.add_argument("--battery", default=None, help="'SL02_digits_long:30,F01_ab_pref:20,...' per endpoint")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--B", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--pi-fp8", type=float, default=None)
    p.add_argument("--force-labels", default=None, help="CSV endpoint,label imposed on the decision")
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("decide", help="labels, z*, weights, partition, gray pairs")
    common(p, endpoints="ep-*")
    p.add_argument("--pairs", default=None, help="pairs.csv from the pairs command (else computed)")
    p.add_argument("--families", default="auto",
                   help="'auto' (stylometry; fails if it collapses to one family), 'one' (explicit single family), "
                        "or a hand-validated CSV with header endpoint,family[,model] covering every endpoint")
    p.add_argument("--allow-one-family", action="store_true",
                   help="accept an 'auto' inference that collapses to a single family")
    p.add_argument("--pi-fp8", type=float, default=None, help="power on fp8 measured by simulate (9.3)")
    p.add_argument("--prior", type=float, default=1.0)
    p.add_argument("--B", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=30)
    p.add_argument("--truth", default=None, help="score the partition against this truth CSV (ref-)")
    p.add_argument("--out", default=None, help="partition CSV path (default OUT_DIR/partition.csv)")
    p.add_argument("--labels-only", action="store_true",
                   help="partition from the labels alone: lures and BASE grouped, everything else alone")
    p.add_argument("--verdicts", default=None,
                   help="white-box verdicts CSV endpoint,verdict (BASE_OR_LURE / FP8 / VARIANT / UNSURE)")
    p.add_argument("--no-data", choices=("alone", "merge", "lures"), default="alone",
                   help="endpoint without any usable pair statistic: kept alone (default); merged into its "
                        "family group ('families only' partition); or 'lures': endpoints labelled as lures "
                        "are grouped (they are the base model), the others stay alone")
    p.add_argument("--force-labels", default=None,
                   help="CSV endpoint,label (DET/WM/SP/TEMP/NONE) imposed by hand, e.g. a persona read in the leaks")
    p.set_defaults(fn=cmd_decide)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
