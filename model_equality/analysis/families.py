"""Family clustering (METHOD 7 "Familles"), simplified.

Only the stylometric variable is implemented: cosine of character-trigram
TF-IDF profiles of each endpoint's free-text answers (D and C probes, plus any
non-probe prompts), average-linkage clustering cut at the largest gap of the
merge heights.  If the largest gap is below ``min_gap``, everything is one
family (doubt -> merge, METHOD 1).  Tokenizer signatures (D11/D12), token
counts and F05 profiles are not implemented: use ``--families FILE.csv`` to
impose a manual family mapping.
"""
from __future__ import annotations

import numpy as np


def endpoint_texts(ds) -> dict:
    out = {}
    for e in ds.endpoints:
        parts = []
        for pid, rs in sorted(ds.responses[e].items()):
            if ds.probes[pid].ptype == "free" and ds.probes[pid].category in ("D_diagnostic", "C_continuation"):
                parts += [r.text for r in rs]
        for pid, rs in sorted(ds.other.get(e, {}).items()):
            parts += [r.text for r in rs]
        out[e] = "\n".join(parts)
    return out


def cluster_families(ds, endpoints: list, min_gap: float = 0.08) -> tuple:
    """(endpoint -> family label, info dict)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = endpoint_texts(ds)
    eps = [e for e in sorted(endpoints)]
    docs = [texts.get(e, "") for e in eps]
    if len(eps) < 3 or sum(bool(d.strip()) for d in docs) < len(eps):
        return {e: "F" for e in eps}, {"reason": "not enough free text; one family"}
    X = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True).fit_transform(docs)
    S = (X @ X.T).toarray()
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    h = Z[:, 2]
    gaps = np.diff(h)            # jumps between consecutive merge heights
    k = int(np.argmax(gaps))
    if gaps[k] < min_gap:
        return {e: "F" for e in eps}, {"reason": f"largest gap between merge heights {gaps[k]:.3f} < {min_gap}; "
                                                 f"one family", "heights": h}
    cut = (h[k] + h[k + 1]) / 2
    lab = fcluster(Z, t=cut, criterion="distance")
    return {e: f"F{int(l)}" for e, l in zip(eps, lab)}, {"cut": float(cut), "heights": h, "sim": S}
