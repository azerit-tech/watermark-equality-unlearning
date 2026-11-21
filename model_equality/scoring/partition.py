"""From pairwise "same model" / "same family" scores to a transitive partition.

Input: a symmetric matrix S (n x n) of similarities or probabilities in which
larger means "more likely the same".  Two methods:

  * ``components``: edge (i, j) iff S[i, j] >= threshold, clusters = connected
    components (single-linkage at that threshold; prone to chaining);
  * ``hierarchical``: scipy agglomerative clustering on distance 1 - S, cut at
    distance 1 - threshold; linkage ``average`` (default), ``complete`` or ``single``.

Families and models:
  * families come from a family matrix if given, else from a fixed mapping,
    else they equal the model clusters;
  * models are always clustered *within* each family (pairs across families
    are never merged), so the output satisfies "same model => same family".

CLI (pairs file with a header naming the columns a,b,p_model[,p_family];
columns are read by name, other columns ignored, missing pairs = 0):
    python3.12 -m model_equality.scoring.partition --pairs P.csv \
        --threshold 0.5 [--family-threshold 0.5] [--method hierarchical --linkage average] \
        [--families FAM.csv] --out submission.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .csvio import write_submission

METHODS = ("components", "hierarchical")


def _check_matrix(S, n=None) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"matrix must be square, got shape {S.shape}")
    if n is not None and S.shape[0] != n:
        raise ValueError(f"matrix has {S.shape[0]} rows, expected {n}")
    if np.isnan(S).any():
        raise ValueError("matrix contains NaN")
    return (S + S.T) / 2.0  # enforce symmetry


def cluster(S, threshold: float, method: str = "hierarchical", linkage: str = "average") -> np.ndarray:
    """Cluster ids (0..k-1, ordered by first appearance) for a similarity matrix."""
    S = _check_matrix(S)
    n = S.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    if method == "components":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        adj = (S >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        _, lab = connected_components(csr_matrix(adj), directed=False)
    elif method == "hierarchical":
        from scipy.cluster.hierarchy import linkage as _linkage, fcluster
        from scipy.spatial.distance import squareform
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        Z = _linkage(squareform(D, checks=False), method=linkage)
        # merge iff merge distance <= 1 - threshold (i.e. similarity >= threshold)
        lab = fcluster(Z, t=1.0 - threshold + 1e-12, criterion="distance")
    else:
        raise ValueError(f"method must be one of {METHODS}")
    return _relabel(lab)


def _relabel(lab) -> np.ndarray:
    mapping: dict = {}
    return np.array([mapping.setdefault(x, len(mapping)) for x in lab], dtype=int)


def partition(endpoints, model_sim, threshold: float, method: str = "hierarchical",
              linkage: str = "average", family_sim=None, family_threshold=None,
              families: dict | None = None):
    """Return (family, model) dicts endpoint -> label, transitive and consistent.

    Labels: families F01, F02, ...; models F01M01, F01M02, ... (globally unique).
    """
    eps = list(endpoints)
    n = len(eps)
    M = _check_matrix(model_sim, n)
    if family_sim is not None:
        ft = threshold if family_threshold is None else family_threshold
        fam_ids = cluster(_check_matrix(family_sim, n), ft, method, linkage)
    elif families is not None:
        missing = [e for e in eps if e not in families]
        if missing:
            raise ValueError(f"families mapping misses {missing}")
        fam_ids = _relabel([families[e] for e in eps])
    else:
        fam_ids = None

    fam_of: dict = {}
    mod_of: dict = {}
    if fam_ids is None:
        mids = cluster(M, threshold, method, linkage)
        for i, e in enumerate(eps):
            fam_of[e] = f"F{mids[i] + 1:02d}"
            mod_of[e] = f"F{mids[i] + 1:02d}M01"
        return fam_of, mod_of
    for f in sorted(set(fam_ids)):
        idx = np.flatnonzero(fam_ids == f)
        sub = M[np.ix_(idx, idx)]
        mids = cluster(sub, threshold, method, linkage)
        for k, i in enumerate(idx):
            fam_of[eps[i]] = f"F{f + 1:02d}"
            mod_of[eps[i]] = f"F{f + 1:02d}M{mids[k] + 1:02d}"
    return fam_of, mod_of


def sweep(endpoints, model_sim, truth: dict, thresholds, variant: str = "true", **kw):
    """Score a range of thresholds against a truth (endpoint -> (family, model)),
    e.g. on the ref-* practice endpoints.  Returns [(threshold, score_dict)]."""
    from .metrics import score
    out = []
    tf = {e: truth[e][0] for e in endpoints}
    tm = {e: truth[e][1] for e in endpoints}
    for t in thresholds:
        f, m = partition(endpoints, model_sim, t, **kw)
        out.append((t, score(f, m, tf, tm, variant)))
    return out


PAIR_A = ("a", "endpoint_a")
PAIR_B = ("b", "endpoint_b")


def load_pairs(path: str):
    """Load a pairs CSV selected BY COLUMN NAME (never by position).

    Required header: an endpoint column ``a`` (or ``endpoint_a``), ``b`` (or
    ``endpoint_b``) and ``p_model``; optional ``p_family``; other columns
    (e.g. z_raw, z_temp) are ignored.  Values of p_* must be finite and in
    [0, 1] (larger = more likely the same); missing pairs count as 0; a pair
    given twice is an error.  Any other layout is refused with ValueError.
    Returns (endpoints, model_matrix, family_matrix_or_None).
    """
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in df.columns}
    ca = next((cols[k] for k in PAIR_A if k in cols), None)
    cb = next((cols[k] for k in PAIR_B if k in cols), None)
    if ca is None or cb is None or "p_model" not in cols:
        raise ValueError(
            f"{path}: pairs file must have a header with columns a,b,p_model[,p_family] "
            f"(a/b may be endpoint_a/endpoint_b); found columns {list(df.columns)}. "
            f"Columns are selected by name, never by position: produce an explicit p_model column.")
    cm, cf = cols["p_model"], cols.get("p_family")
    if df.empty:
        raise ValueError(f"{path}: no pairs")
    eps = sorted(set(df[ca].str.strip()) | set(df[cb].str.strip()))
    pos = {e: i for i, e in enumerate(eps)}
    n = len(eps)
    M = np.eye(n)
    F = np.eye(n) if cf is not None else None
    seen = set()

    def val(x, col, line):
        try:
            v = float(x)
        except ValueError:
            raise ValueError(f"{path}: line {line}: {col}={x!r} is not a number")
        if not np.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError(f"{path}: line {line}: {col}={v} outside [0, 1] "
                             f"(p_* columns are probabilities/similarities)")
        return v

    for k, row in enumerate(df.to_dict("records")):
        line = k + 2
        ea, eb = row[ca].strip(), row[cb].strip()
        if ea == eb:
            raise ValueError(f"{path}: line {line}: self pair {ea}")
        key = tuple(sorted((ea, eb)))
        if key in seen:
            raise ValueError(f"{path}: line {line}: pair {key} given twice")
        seen.add(key)
        i, j = pos[ea], pos[eb]
        M[i, j] = M[j, i] = val(row[cm], "p_model", line)
        if F is not None:
            F[i, j] = F[j, i] = val(row[cf], "p_family", line)
    return eps, M, F


def read_families(path: str) -> dict:
    """endpoint -> family from a CSV whose first two columns are endpoint,family
    (a submission-style endpoint,family,model file works; extra columns are
    ignored; header optional).  Rows with fewer than 2 cells are an error."""
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for k, r in enumerate(csv.reader(fh)):
            r = [c.strip() for c in r]
            if not any(r):
                continue
            if k == 0 and r[:2] == ["endpoint", "family"]:
                continue
            if len(r) < 2 or not r[0] or not r[1]:
                raise ValueError(f"{path}: line {k + 1}: expected endpoint,family[,...], got {r}")
            if r[0] in out:
                raise ValueError(f"{path}: line {k + 1}: duplicate endpoint {r[0]}")
            out[r[0]] = r[1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True,
                    help="pairs CSV with header a,b,p_model[,p_family] (columns read by name, others ignored)")
    ap.add_argument("--threshold", type=float, required=True, help="'same model' threshold on similarity")
    ap.add_argument("--family-threshold", type=float, default=None)
    ap.add_argument("--method", choices=METHODS, default="hierarchical")
    ap.add_argument("--linkage", choices=("average", "complete", "single"), default="average")
    ap.add_argument("--families", help="CSV whose first two columns are endpoint,family (header optional, "
                    "extra columns such as model ignored); fixes the families and overrides p_family")
    ap.add_argument("--out", required=True)
    ap.add_argument("--practice", action="store_true", help="ids are ref-01..ref-09 (for calibration only)")
    a = ap.parse_args(argv)
    try:
        eps, M, F = load_pairs(a.pairs)
        fams = None
        if a.families:
            fams = read_families(a.families)
            F = None
        f, m = partition(eps, M, a.threshold, a.method, a.linkage, family_sim=F,
                         family_threshold=a.family_threshold, families=fams)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    from .csvio import PRACTICE_ENDPOINTS
    try:
        rep = write_submission(a.out, f, m, expected_endpoints=PRACTICE_ENDPOINTS if a.practice else None)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(rep.format())
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
