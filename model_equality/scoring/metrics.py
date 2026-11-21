"""Pairwise MCC scoring for Model Equality Testing (task 2, consigne p. 39).

score = 0.1 * MCC_family + 0.9 * MCC_model, each MCC clipped at 0 from below.

* MCC_family: "same family?" over all unordered pairs (528 for 33 endpoints).
* MCC_model:  "same model?" over the pairs "within one family".  The consigne
  does not say whose family.  Two variants are computed:
    - ``true`` (default): pairs whose TRUE families are equal;
    - ``pred``:           pairs whose PREDICTED families are equal.
  An ``all`` variant (every pair) is also reported for reference.

Convention: when the MCC denominator is 0 (one class absent on either side),
MCC = 0 (same as scikit-learn).  Hence "everything same" and "everything
different" both score 0.

CLI:
    python3.12 -m model_equality.scoring.metrics PRED.csv TRUTH.csv [--variant true|pred|all] [--json]
    python3.12 -m model_equality.scoring.metrics PRED.csv model_equality/scoring/data/practice_answers.csv --practice-format
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable, Mapping

VARIANTS = ("true", "pred", "all")
DEFAULT_VARIANT = "true"
W_FAMILY = 0.1
W_MODEL = 0.9


@dataclass
class Confusion:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    false_pairs: list = field(default_factory=list)  # (a, b, "FP"|"FN")

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def mcc_raw(self) -> float:
        return mcc_from_counts(self.tp, self.tn, self.fp, self.fn)

    def mcc(self) -> float:
        return max(0.0, self.mcc_raw())

    def as_dict(self) -> dict:
        return {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn, "n_pairs": self.n,
                "mcc_raw": self.mcc_raw(), "mcc": self.mcc(),
                "false_pairs": [list(p) for p in self.false_pairs]}


def mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient (not clipped). 0 if the denominator is 0."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denom)


def confusion(endpoints: Iterable[str], pred_same, true_same, include=None) -> Confusion:
    """Confusion over unordered pairs of ``endpoints`` (sorted, i<j).

    pred_same/true_same: callables (a, b) -> bool.  include: optional callable
    (a, b) -> bool selecting the pairs that count.
    """
    eps = sorted(endpoints)
    c = Confusion()
    for a, b in itertools.combinations(eps, 2):
        if include is not None and not include(a, b):
            continue
        p, t = bool(pred_same(a, b)), bool(true_same(a, b))
        if p and t:
            c.tp += 1
        elif not p and not t:
            c.tn += 1
        elif p and not t:
            c.fp += 1
            c.false_pairs.append((a, b, "FP"))
        else:
            c.fn += 1
            c.false_pairs.append((a, b, "FN"))
    return c


def _check_keys(pred: Mapping, truth: Mapping, what: str) -> None:
    missing = sorted(set(truth) - set(pred))
    extra = sorted(set(pred) - set(truth))
    if missing or extra:
        raise ValueError(f"{what}: endpoints differ between prediction and truth "
                         f"(missing in prediction: {missing}; not in truth: {extra})")


def score(pred_family: Mapping[str, str], pred_model: Mapping[str, str],
          true_family: Mapping[str, str], true_model: Mapping[str, str],
          variant: str = DEFAULT_VARIANT) -> dict:
    """Full local score.  All mappings are endpoint -> label.

    Model equality is decided on the (family, model) label pair, exactly as the
    organisers' score_practice.py does ("same model = same family and same model
    label"); for a valid CSV this equals model-label equality.
    Returns a dict with mcc_family, mcc_model (selected variant),
    mcc_model_variants, score, score_variants, confusions and false pairs.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    _check_keys(pred_family, true_family, "family")
    _check_keys(pred_model, true_model, "model")
    _check_keys(pred_family, pred_model, "prediction family/model")
    eps = sorted(true_family)

    fam = confusion(eps, lambda a, b: pred_family[a] == pred_family[b],
                    lambda a, b: true_family[a] == true_family[b])
    pm = lambda a, b: (pred_family[a], pred_model[a]) == (pred_family[b], pred_model[b])  # noqa: E731
    tm = lambda a, b: (true_family[a], true_model[a]) == (true_family[b], true_model[b])  # noqa: E731
    models = {
        "true": confusion(eps, pm, tm, include=lambda a, b: true_family[a] == true_family[b]),
        "pred": confusion(eps, pm, tm, include=lambda a, b: pred_family[a] == pred_family[b]),
        "all": confusion(eps, pm, tm),
    }
    mf = fam.mcc()
    scores = {v: W_FAMILY * mf + W_MODEL * models[v].mcc() for v in VARIANTS}
    return {
        "variant": variant,
        "n_endpoints": len(eps),
        "mcc_family": mf,
        "mcc_model": models[variant].mcc(),
        "mcc_model_variants": {v: models[v].mcc() for v in VARIANTS},
        "score": scores[variant],
        "score_variants": scores,
        "family_confusion": fam.as_dict(),
        "model_confusion": {v: models[v].as_dict() for v in VARIANTS},
    }


def score_files(pred_csv: str, truth_csv: str, variant: str = DEFAULT_VARIANT) -> dict:
    """Score two CSVs (endpoint,family,model).

    The truth defines the endpoints (e.g. ref-01..ref-09 in practice_answers.csv,
    whose 4th 'kind' column is ignored).  Like score_practice.py, prediction
    lines for other ids are ignored; missing ids are an error.  If the truth is
    on ep-01..ep-33 the prediction is fully validated as a submission."""
    from .csvio import read_partition, validate, SUBMISSION_ENDPOINTS
    truth = read_partition(truth_csv)
    ids = sorted(truth)
    if ids == SUBMISSION_ENDPOINTS:
        rep = validate(pred_csv)
        if not rep.ok:
            raise ValueError("prediction CSV invalid:\n" + rep.format())
    pred = read_partition(pred_csv)
    missing = [e for e in ids if e not in pred]
    if missing:
        raise ValueError(f"Your file has no line for: {', '.join(missing)}")
    tf = {e: truth[e][0] for e in ids}
    tm = {e: truth[e][1] for e in ids}
    return score({e: pred[e][0] for e in ids}, {e: pred[e][1] for e in ids}, tf, tm, variant)


def practice_report(pred_csv: str, truth_csv: str) -> str:
    """Text identical to the stdout of the organisers' score_practice.py
    (same-model MCC over ALL pairs of the answer ids, i.e. variant "all";
    on the practice set there is one family, so "all" == "true")."""
    from .csvio import read_rows
    answers = read_rows(truth_csv, prefix="ref-")
    mine = read_rows(pred_csv, prefix="ref-")
    missing = sorted(set(answers) - set(mine))
    if missing:
        raise ValueError(f"Your file has no line for: {', '.join(missing)}")
    eps = sorted(answers)
    c = confusion(eps, lambda a, b: mine[a][:2] == mine[b][:2], lambda a, b: answers[a][:2] == answers[b][:2])
    lines = [f"Practice score (MCC over {c.n} pairs, same model or not): {c.mcc():.3f}",
             f"Right: {c.tp + c.tn} pairs, wrong: {c.fp + c.fn} ({c.fp} wrongly called the same, "
             f"{c.fn} wrongly called different)"]
    if c.false_pairs:
        lines.append("Wrong pairs:")
        for a, b, k in c.false_pairs:
            guess = k == "FP"
            lines.append(f"  {a} ({answers[a][2]}) and {b} ({answers[b][2]}): you said "
                         f"{'same' if guess else 'different'}, they are "
                         f"{'the same model' if not guess else 'different models'}")
    return "\n".join(lines)


def format_report(r: dict) -> str:
    v = r["variant"]
    lines = [
        f"endpoints: {r['n_endpoints']}",
        f"MCC_family            = {r['mcc_family']:.6f}",
    ]
    for var in VARIANTS:
        tag = " <- used" if var == v else ""
        lines.append(f"MCC_model[{var:4s}]      = {r['mcc_model_variants'][var]:.6f}{tag}")
    for var in ("true", "pred"):
        tag = " <- used" if var == v else ""
        lines.append(f"score[{var:4s}]          = {r['score_variants'][var]:.6f}{tag}")
    fc = r["family_confusion"]
    lines.append(f"family confusion: TP={fc['tp']} TN={fc['tn']} FP={fc['fp']} FN={fc['fn']} (pairs={fc['n_pairs']})")
    for var in VARIANTS:
        mc = r["model_confusion"][var]
        lines.append(f"model[{var}] confusion: TP={mc['tp']} TN={mc['tn']} FP={mc['fp']} FN={mc['fn']} (pairs={mc['n_pairs']})")
    if fc["false_pairs"]:
        lines.append("wrong family pairs:")
        lines += [f"  {a} {b} {k}" for a, b, k in fc["false_pairs"]]
    mc = r["model_confusion"][v]
    if mc["false_pairs"]:
        lines.append(f"wrong model pairs [{v}]:")
        lines += [f"  {a} {b} {k}" for a, b, k in mc["false_pairs"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("truth")
    ap.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                    help="pairs counted for MCC_model (default: true family)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--practice-format", action="store_true",
                    help="print exactly what the organisers' score_practice.py prints")
    a = ap.parse_args(argv)
    try:
        if a.practice_format:
            print(practice_report(a.pred, a.truth))
            return 0
        r = score_files(a.pred, a.truth, a.variant)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(r, indent=2) if a.json else format_report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
