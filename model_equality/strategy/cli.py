"""Command line of the submission strategy (never sends anything, never calls the API).

    python3.12 -m model_equality.strategy update          # read the submit journal (scores received)
    python3.12 -m model_equality.strategy next            # partition to submit next, and what it tests
                                                          # (--max-moves 1: one endpoint only)
    python3.12 -m model_equality.strategy assess FILE.csv [FILE2.csv ...]   # value of partitions of our own
    python3.12 -m model_equality.strategy final           # submission kept, expected private score
    python3.12 -m model_equality.strategy status          # slots used / left, history
    python3.12 -m model_equality.strategy simulate --scenario split --worlds 48 --slots 27

First call: add  --candidate PARTITION.csv [--pairs PAIRS.csv]  (both are then remembered in state/config.json).

Formats (same as analysis decide / scoring submit):
  PARTITION.csv  endpoint,family,model   the candidate partition: its families are taken as right, its groups
                                          as "same model unless shown otherwise"
  PAIRS.csv      a,b,p_model             optional; probability 'same model' for a pair, columns read by name.
                                          One line "lone endpoint, one member of a group, p" holds for the whole
                                          group.  A decision_pairs.csv with a 'weight' column also works
                                          (p = sigmoid(weight), NO_DATA = unknown).
  Pairs without a line: 1 - pi_hidden inside a group of the candidate, --p-missing (0.3) between a lone
  endpoint and a group, --p-twin (0.05) between two lone endpoints.

State: model_equality/strategy/state/ (history.jsonl + a copy of every recorded CSV, config.json).
``next`` writes model_equality/strategy/out/next/partition.csv with a summary.txt next to it (submit.py
archives it with the submission) and posterior_pairs.csv (updated p_model per pair).

``update`` without option imports the new entries of model_equality/scoring/submissions/log.jsonl (read only).
``update --csv FILE --score S`` records by hand a submission made elsewhere (``--failed``: sent, no score).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from model_equality.scoring.csvio import SUBMISSION_ENDPOINTS, validate, write_submission

from .core import SCORE_STEP, W_FAMILY, W_MODEL, Belief, Problem, canon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state"
OUT = HERE / "out" / "next"
SUBMIT_DIR = REPO / "model_equality" / "scoring" / "submissions"
PARIS = ZoneInfo("Europe/Paris")
DEADLINE = "2026-11-22T12:00"
GUARD_MIN = 32


# ----------------------------------------------------------------- state
def _now():
    return dt.datetime.now(dt.timezone.utc)


def read_history(state: Path) -> list:
    p = state / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_history(state: Path, hist: list) -> None:
    state.mkdir(parents=True, exist_ok=True)
    tmp = state / "history.jsonl.tmp"
    tmp.write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in hist), encoding="utf-8")
    tmp.replace(state / "history.jsonl")


def read_config(state: Path) -> dict:
    p = state / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def record(state: Path, hist: list, csv_path: Path, score, ts: str, source: str, note: str = "",
           extra: dict | None = None) -> dict:
    rep = validate(str(csv_path))
    if not rep.ok:
        raise ValueError(f"{csv_path}: CSV invalide:\n{rep.format()}")
    data = Path(csv_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    state.mkdir(parents=True, exist_ok=True)
    copy = state / f"sub_{len(hist) + 1:03d}_{sha[:12]}.csv"
    copy.write_bytes(data)
    entry = dict(n=len(hist) + 1, ts=ts, csv=copy.name, sha256=sha, score=score,
                 status="scored" if score is not None else "failed", source=source, note=note,
                 rows={e: list(v) for e, v in rep.rows.items()})
    entry.update(extra or {})
    hist.append(entry)
    return entry


def sync_submit_log(state: Path, hist: list, submit_dir: Path = SUBMIT_DIR) -> list:
    """Import the attempts of scoring/submissions/log.jsonl that are not yet in the history.
    An attempt whose result is not in the journal yet (request in flight) is recorded as 'pending'
    (it already uses the slot) and completed by a later call."""
    log = submit_dir / "log.jsonl"
    if not log.exists():
        return []
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = {e.get("attempt_ts"): e for e in entries if e.get("kind") == "result"}
    known = {h["ts"]: h for h in hist}
    added = []
    for e in entries:
        if e.get("kind") != "attempt":
            continue
        r = results.get(e["ts"])
        ok = bool(r) and r.get("status") == "success" and isinstance(r.get("score"), (int, float))
        extra = {k: r[k] for k in ("improved", "previous_best", "http_status") if r and k in r}
        note = str(r.get("message") or r.get("error") or r.get("detail") or "") if r else "résultat pas encore au journal"
        status = "scored" if ok else ("failed" if r else "pending")
        if e["ts"] in known:
            h = known[e["ts"]]
            if h.get("status") == "pending" and r:               # the result has arrived since
                h.update(score=float(r["score"]) if ok else None, status=status, note=note, **extra)
                added.append(h)
            continue
        h = record(state, hist, submit_dir / e["copy"], float(r["score"]) if ok else None, e["ts"],
                   "submit-log", note=note, extra=extra)
        h["status"] = status
        added.append(h)
    return added


# ----------------------------------------------------------------- inputs
def load_pairs(path: str, eps: list) -> np.ndarray:
    """(n x n) p_model matrix aligned with ``eps``; NaN where the file says nothing
    (pair absent, empty cell, or a NO_DATA weight of decide)."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    ca = next((cols[k] for k in ("a", "endpoint_a") if k in cols), None)
    cb = next((cols[k] for k in ("b", "endpoint_b") if k in cols), None)
    if ca is None or cb is None:
        raise ValueError(f"{path}: colonnes a,b introuvables (trouvé {list(df.columns)})")
    if "p_model" in cols:
        val = pd.to_numeric(df[cols["p_model"]], errors="coerce").to_numpy(float)
    elif "weight" in cols:
        w = pd.to_numeric(df[cols["weight"]], errors="coerce").to_numpy(float)
        val = np.where(w <= -1e5, np.nan, 1.0 / (1.0 + np.exp(-np.clip(w, -30, 30))))
        print(f"NOTE: {path} n'a pas de colonne p_model ; p_model = sigmoid(weight), NO_DATA = inconnu.",
              file=sys.stderr)
    else:
        raise ValueError(f"{path}: il faut une colonne p_model (ou weight) ; trouvé {list(df.columns)}")
    ok = np.isfinite(val)
    if (val[ok] < 0).any() or (val[ok] > 1).any():
        raise ValueError(f"{path}: p_model doit être dans [0, 1]")
    pos = {e: i for i, e in enumerate(eps)}
    M = np.full((len(eps), len(eps)), np.nan)
    for a, b, v in zip(df[ca].astype(str).str.strip(), df[cb].astype(str).str.strip(), val):
        if a in pos and b in pos and a != b:
            M[pos[a], pos[b]] = M[pos[b], pos[a]] = v
    return M


def build_problem(a, cfg: dict) -> Problem:
    cand_path, pairs_path = a.candidate or cfg.get("candidate"), a.pairs or cfg.get("pairs")
    if not cand_path:
        raise ValueError("donner --candidate PARTITION.csv (mémorisée ensuite dans config.json)")
    rep = validate(cand_path)
    if not rep.ok:
        raise ValueError(f"{cand_path}: partition candidate invalide:\n{rep.format()}")
    eps = list(SUBMISSION_ENDPOINTS)
    n = len(eps)
    fam = {e: rep.rows[e][0] for e in eps}
    mod = {e: rep.rows[e][1] for e in eps}
    M = load_pairs(pairs_path, eps) if pairs_path else np.full((n, n), np.nan)
    key = [(fam[e], mod[e]) for e in eps]
    block = {k: [i for i in range(n) if key[i] == k] for k in set(key)}
    unknown = np.isnan(M)
    F = M.copy()

    def lgt(v):
        v = min(max(float(v), 1e-6), 1 - 1e-6)
        return math.log(v / (1 - v))

    keys = sorted(block)
    for ka in keys:
        A = block[ka]
        for i in A:                                       # inside a candidate block
            for j in A:
                if i != j and unknown[i, j]:
                    F[i, j] = 1.0 - a.pi_hidden
        for kb in keys:
            if kb <= ka or ka[0] != kb[0]:
                continue
            B = block[kb]
            known = [M[i, j] for i in A for j in B if not unknown[i, j]]
            if known:                                     # a hint against one member holds for the block
                fill = 1.0 / (1.0 + math.exp(-float(np.mean([lgt(v) for v in known]))))
            elif len(A) == 1 and len(B) == 1:
                fill = a.p_twin                           # two lone endpoints
            elif len(A) == 1 or len(B) == 1:
                fill = a.p_missing                        # a lone endpoint against a group
            else:
                fill = a.p_merge                          # two groups
            for i in A:
                for j in B:
                    if unknown[i, j]:
                        F[i, j] = F[j, i] = fill
    F = np.where(np.isnan(F), a.p_lo, F)
    pb = Problem(eps, fam, mod, F, q=a.q, pi_hidden=a.pi_hidden, p_lo=a.p_lo, p_hi=a.p_hi,
                 anchored=not a.no_anchor)
    missing = int(unknown[pb.ia, pb.ib].sum())
    if missing and not getattr(a, "_warned", False):
        a._warned = True
        print(f"NOTE: {missing} paires intra-famille sur {pb.nw} sans p_model. A priori utilisé : "
              f"{1 - a.pi_hidden:.2f} dans une grappe de la partition candidate ; {a.p_missing} entre un endpoint "
              f"seul et une grappe (--p-missing) ; {a.p_twin} entre deux endpoints seuls (--p-twin).", file=sys.stderr)
    return pb


def replay(pb: Problem, hist: list, a, upto: int | None = None, out=print):
    """Belief after the scored submissions of the history (in order)."""
    step = getattr(a, "step", SCORE_STEP)
    bel = Belief(pb, n_chains=a.chains, seed=a.seed, step=step if step and step > 0 else None)
    used = []
    for h in hist[:upto]:
        if h.get("score") is None:
            continue
        lab = pb.labels_from({e: r[0] for e, r in h["rows"].items()}, {e: r[1] for e, r in h["rows"].items()})
        if lab is None:
            out(f"ATTENTION: soumission #{h['n']} ignorée (ses familles diffèrent de la partition candidate).")
            continue
        sub = bel.observe(lab, h["score"])
        sub["n"] = h["n"]
        used.append(h)
        if sub["kind"] == "model" and not sub["dup"]:
            bel.refresh(max(2, a.sweeps // 3), rescue=0)       # light pass; the full one comes at the end
    if used and any(x["kind"] == "model" for x in bel.subs):
        bel.refresh(a.sweeps)
    return bel, used


# ----------------------------------------------------------------- text helpers
def _fmt_group(names: list, k: int = 8) -> str:
    return "{" + ", ".join(names[:k]) + (", …" if len(names) > k else "") + "}"


def describe(pb: Problem, a_lab, b_lab) -> list:
    out = []
    for kind, x, y in pb.diff(a_lab, b_lab):
        if kind == "split":
            who = x[0] if len(x) == 1 else _fmt_group(x)
            out.append(f"isole {who} de {_fmt_group(y)}")
        else:
            out.append("réunit " + " + ".join(_fmt_group(p) if len(p) > 1 else p[0] for p in x))
    return out or ["(aucune différence)"]


def mode_text(bel: Belief) -> str:
    if bel.mode == "exact":
        return "exact (chaque score est décodé en comptes de paires publiques)"
    if bel.mode == "approx":
        g = bel._grid_now()
        return (f"approché (aucun score n'a plus de 6 décimales : scores supposés arrondis"
                f"{'' if g is None else f' au pas de {g:g}'}, tolérance {bel.tol_used:.1e})")
    return ("signe seul (les scores ne collent pas au modèle de score : formule, familles ou arrondi ; "
            "seul le sens des variations est exploité)")


def decoded_text(bel: Belief, k: int | None = None) -> str:
    d = bel.decoded(k)
    if d is None:
        return "comptes publics : non décodés"
    if len(d) == 1:
        tp, fp, fn, tn = (int(v) for v in d[0])
        n = tp + fp + fn + tn
        return (f"comptes publics décodés : TP={tp} FP={fp} FN={fn} TN={tn} "
                f"({n} paires publiques sur {bel.pb.nw} intra-famille, soit {n / bel.pb.nw:.0%})")
    fp, fn = d[:, 1], d[:, 2]
    return (f"comptes publics : {len(d)} décodages possibles, FP entre {fp.min()} et {fp.max()}, "
            f"FN entre {fn.min()} et {fn.max()}")


def slots_info(hist: list, deadline: str, now=None) -> dict:
    now = now or _now()
    dl = dt.datetime.fromisoformat(deadline)
    if dl.tzinfo is None:
        dl = dl.replace(tzinfo=PARIS)
    last = max((dt.datetime.fromisoformat(h["ts"]) for h in hist), default=None)
    nxt = now if last is None else max(now, last + dt.timedelta(minutes=GUARD_MIN))
    left = 0 if nxt > dl else int((dl - nxt).total_seconds() // (GUARD_MIN * 60)) + 1
    return dict(next=nxt.astimezone(PARIS), deadline=dl.astimezone(PARIS), left=left, used=len(hist))


# ----------------------------------------------------------------- commands
def cmd_update(a) -> int:
    state = Path(a.state_dir)
    hist = read_history(state)
    if a.csv:
        if a.score is None and not a.failed:
            print("ERREUR: --csv demande --score S (ou --failed pour un envoi sans score).", file=sys.stderr)
            return 2
        added = [record(state, hist, Path(a.csv), None if a.failed else float(a.score),
                        a.ts or _now().isoformat(), "manual", note=a.note or "")]
    else:
        added = sync_submit_log(state, hist, Path(a.submit_dir))
    if not added:
        print("Rien de nouveau dans le journal des soumissions.")
        return 0
    write_history(state, hist)
    new_ns = {h["n"] for h in added}
    for h in added:
        print(f"Enregistré #{h['n']} ({h['ts']}) : " +
              (f"score public {h['score']!r}" if h["score"] is not None
               else f"{'EN ATTENTE' if h.get('status') == 'pending' else 'ÉCHEC'} ({h['note']})"))
    known_best = max((h["score"] for h in hist if h["n"] not in new_ns and h.get("score") is not None),
                     default=None)
    for h in added:
        pb_ = h.get("previous_best")
        if pb_ is not None and known_best is not None and abs(pb_ - known_best) > 1e-9:
            print(f"ATTENTION: le serveur annonce previous_best={pb_} mais notre historique donne {known_best} : "
                  f"une soumission faite ailleurs manque (l'ajouter avec update --csv ... --score ...).")
        if h.get("score") is not None:
            known_best = h["score"] if known_best is None else max(known_best, h["score"])
    cfg = read_config(state)
    try:
        pb = build_problem(a, cfg)
    except ValueError as e:
        print(f"(pas d'interprétation : {e})")
        return 0
    bel, used = replay(pb, hist, a)
    print(f"Décodage : {mode_text(bel)}")
    subs = bel.scored()
    last = added[-1]
    if last.get("score") is None or not used or used[-1]["n"] != last["n"]:
        return 0
    cur = bel.subs[-1]
    if cur["kind"] == "probe":
        print(f"Soumission à prédiction constante dans les familles : le score est le terme famille "
              f"(MCC_famille public = {cur['score'] / W_FAMILY:.4f}). " +
              ("Familles justes sur toutes les paires publiques." if abs(cur["score"] - W_FAMILY) < 1e-9
               else "ATTENTION : familles imparfaites, le module les suppose justes."))
        return 0
    if cur["dup"]:
        print("Partition déjà soumise : aucune information nouvelle.")
        return 0
    print(decoded_text(bel))
    prev = [x for x in subs[:-1]]
    if prev:
        ref = prev[0]
        for x in prev:
            if x["score"] > ref["score"] + 1e-12:
                ref = x
        d = cur["score"] - ref["score"]
        print(f"Par rapport à la référence #{ref['n']} ({ref['score']:.6f}) : {d:+.6f}  [" +
              " ; ".join(describe(pb, ref["lab"], cur["lab"])) + "]")
        if d > 1e-9:
            print("  -> le score public monte : cette partition devient la référence retenue par le serveur.")
        elif abs(d) <= 1e-9:
            m = int((ref["s"] != cur["s"]).sum())
            p0 = (1 - pb.q) ** m
            if bel.mode == "exact":
                print(f"  -> score identique : aucune des {m} paires changées n'est publique (cela arrive avec "
                      f"probabilité {p0:.2f}) ; rien d'appris, la référence reste la première arrivée à ce score.")
            else:
                print(f"  -> score identique à l'arrondi : soit aucune des {m} paires changées n'est publique "
                      f"(probabilité {p0:.2f}), soit des paires justes et des paires fausses se compensent "
                      f"(au moins un changement juste et un faux). La référence reste la première arrivée.")
        else:
            print("  -> le score public baisse : changement rejeté sur les paires publiques, la référence reste.")
    return 0


def cmd_next(a) -> int:
    state = Path(a.state_dir)
    hist = read_history(state)
    cfg = read_config(state)
    try:
        pb = build_problem(a, cfg)
    except ValueError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 2
    cfg.update(candidate=str(Path(a.candidate or cfg["candidate"]).resolve()))
    if a.pairs or cfg.get("pairs"):
        cfg.update(pairs=str(Path(a.pairs or cfg["pairs"]).resolve()))
    state.mkdir(parents=True, exist_ok=True)
    (state / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    lines = []
    say = lines.append
    info = slots_info(hist, a.deadline)
    say(f"Créneaux : {info['used']} utilisés ; prochain possible à {info['next']:%H:%M} (garde {GUARD_MIN} min) ; "
        f"{info['left']} restants avant {info['deadline']:%d/%m %H:%M}.")
    bel, used = replay(pb, hist, a, out=say)
    say(f"Décodage : {mode_text(bel)}")
    inc = bel.incumbent()
    out_dir = Path(a.out).parent if a.out else OUT
    out_path = Path(a.out) if a.out else OUT / "partition.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.probe_families:
        lab = np.arange(pb.n)
        say("Sonde familles : chaque endpoint seul dans son modèle. Score attendu = 0,1 x MCC_famille public "
            "(0,1 si les familles sont justes). À envoyer avec --allow-degenerate.")
        prop = dict(lab=lab, first=False, probe=True)
    else:
        prop = bel.propose(n_rep=a.rep, v_min=a.v_min, risk=a.risk, depth=a.max_moves)
    if inc is not None:
        say(f"Référence (retenue par le serveur) : #{inc['n']}, score public {inc['score']:.6f}. " + decoded_text(
            bel, [x["n"] for x in bel._items()].index(inc["n"]) if bel.dec is not None else None))
        if inc["score"] >= 1 - 1e-9:
            say("Score public = 1 : plus aucune erreur visible sur les paires publiques, le public ne peut plus rien apprendre.")
    if prop["lab"] is None:
        say("AUCUNE soumission utile : aucun changement n'a d'espérance de gain privé positive "
            f"(meilleure valeur {prop.get('V', 0):.5f} < {a.v_min}). Garder le créneau.")
        _alternatives(pb, bel, inc, prop, say)
        if out_path.exists():
            out_path.unlink()                              # no stale proposal left behind
        (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0
    lab = prop["lab"]
    family, model = pb.names(lab)
    if out_path.exists():
        out_path.unlink()
    write_submission(str(out_path), family, model)
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    shutil.copyfile(out_path, state / f"proposal_{stamp}.csv")
    say(f"PROPOSITION écrite : {out_path}")
    if prop.get("probe"):
        pass
    elif prop.get("first"):
        say("Première soumission modèle : la partition candidate telle quelle (notre meilleure estimation). "
            "Elle fixe la référence ; tous les tests suivants en sont des variantes à un changement.")
    else:
        say("Ce que la soumission teste (par rapport à la référence) :")
        for d in describe(pb, inc["lab"], lab):
            say(f"  - {d}")
        say(f"  probabilité a posteriori que ce changement soit juste : {prop['p_right']:.2f}")
        say("Lecture du résultat :")
        say(f"  - score > {inc['score']:.6f} (probabilité {prop['p_adopt']:.2f}) : changement confirmé sur les paires "
            f"publiques, la soumission devient la référence ; gain privé attendu alors {prop['gain_if_adopted']:+.4f} ; "
            f"risque que le privé baisse quand même {prop['p_harm_if_adopted']:.0%}.")
        say(f"  - score identique (probabilité {prop['p_tie']:.2f}) : aucune paire publique touchée, rien d'appris.")
        say(f"  - score plus bas (probabilité {max(0.0, 1 - prop['p_adopt'] - prop['p_tie']):.2f}) : changement rejeté, "
            f"la référence reste ; le décodage dit quand même combien de paires publiques étaient justes.")
        say(f"  valeur du test (gain privé espéré) : {prop['V']:+.5f}")
        _alternatives(pb, bel, inc, prop, say)
    _uncertain(pb, bel, inc, say)
    marg = bel.marginals()
    with open(out_dir / "posterior_pairs.csv", "w", encoding="utf-8") as fh:
        fh.write("a,b,p_model_prior,p_model\n")
        for k, (i, j) in enumerate(zip(pb.ia, pb.ib)):
            fh.write(f"{pb.eps[i]},{pb.eps[j]},{pb.P[i, j]:.4f},{marg[k]:.4f}\n")
    say(f"Croyances à jour par paire : {out_dir / 'posterior_pairs.csv'}")
    say("Envoi (après accord) : python3.12 -m model_equality.scoring.submit " + str(out_path) +
        (" --allow-degenerate" if prop.get("probe") else "") + "   puis --confirm-submit ; ensuite : "
        "python3.12 -m model_equality.strategy update")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _alternatives(pb, bel, inc, prop, say):
    alts = [x for x in prop.get("alternatives", []) if x["V"] > 0]
    if not alts or inc is None:
        return
    say("Autres tests possibles (valeur, P(le public monte), P(juste)) :")
    for x in alts[:5]:
        say(f"  {x['V']:+.5f}  {x['p_adopt']:.2f}  {x['p_right']:.2f}  " + " ; ".join(describe(pb, inc["lab"], x["lab"])))


def _uncertain(pb, bel, inc, say, k: int = 8):
    """Endpoints whose place in the reference partition is the least certain."""
    if inc is None:
        return
    marg = np.zeros((pb.n, pb.n))
    m = bel.marginals()
    marg[pb.ia, pb.ib] = m
    marg[pb.ib, pb.ia] = m
    lab = inc["lab"]
    rows = []
    for i in range(pb.n):
        mates = np.flatnonzero((lab == lab[i]) & (np.arange(pb.n) != i))
        if len(mates):
            rows.append((1 - marg[i, mates].mean(), pb.eps[i], f"distinct de sa grappe ({len(mates) + 1} endpoints)"))
        else:
            others = np.flatnonzero((pb.fam == pb.fam[i]) & (np.arange(pb.n) != i))
            if len(others):
                j = others[marg[i, others].argmax()]
                rows.append((marg[i, j], pb.eps[i], f"en réalité le même modèle que {pb.eps[j]}"))
    rows.sort(reverse=True)
    say("Doutes restants les plus forts (probabilité a posteriori) :")
    for p, e, what in rows[:k]:
        if p >= 0.03:
            say(f"  {p:.2f}  {e} {what}")


def cmd_final(a) -> int:
    state = Path(a.state_dir)
    hist = read_history(state)
    cfg = read_config(state)
    try:
        pb = build_problem(a, cfg)
    except ValueError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 2
    bel, used = replay(pb, hist, a)
    inc = bel.incumbent()
    if inc is None:
        print("Aucune soumission modèle notée dans l'historique.")
        return 0
    print(f"Décodage : {mode_text(bel)}")
    rows = bel.summary(n_rep=a.rep)["rows"]
    subs = bel.scored()
    top = [x for x in subs if abs(x["score"] - inc["score"]) <= 1e-9]
    print(f"{'#':>3s} {'public':>9s} {'privé estimé':>12s} {'[10% ; 90%]':>17s} {'P(pire que la 1re)':>18s}")
    for x, r in zip(subs, rows):
        mark = "  <- RETENUE" if x is inc else ("  (ex æquo)" if any(x is y for y in top) else "")
        print(f"{x['n']:3d} {x['score']:9.6f} {r['priv_mean']:12.4f}   [{r['priv_lo']:.3f} ; {r['priv_hi']:.3f}] "
              f"{r['p_worse_than_first']:18.2f}{mark}")
    print(f"Le serveur retient #{inc['n']} (meilleur score public ; en cas d'égalité on suppose la première). "
          f"Fichier : {state / next(h['csv'] for h in hist if h['n'] == inc['n'])}")
    if len(top) > 1:
        print(f"ATTENTION : {len(top)} soumissions ex æquo au meilleur score public ({[x['n'] for x in top]}). "
              "Si le serveur retient la dernière et non la première, re-soumettre la partition voulue avec --force.")
    r_inc = rows[next(k for k, x in enumerate(subs) if x is inc)]
    print(f"Score privé estimé de la retenue : {r_inc['priv_mean']:.3f} ; score public {inc['score']:.3f} "
          f"(écart d'optimisme attendu {inc['score'] - r_inc['priv_mean']:+.3f}).")
    prop = bel.propose(n_rep=a.rep, v_min=a.v_min, risk=a.risk, depth=a.max_moves)
    info = slots_info(hist, a.deadline)
    if prop["lab"] is None:
        print("Aucun test restant n'a de valeur positive : s'arrêter là.")
    else:
        print(f"Il reste un test utile (valeur {prop['V']:+.5f}) et {info['left']} créneaux : lancer `next`.")
    return 0


def cmd_assess(a) -> int:
    state = Path(a.state_dir)
    hist = read_history(state)
    try:
        pb = build_problem(a, read_config(state))
    except ValueError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 2
    bel, used = replay(pb, hist, a)
    inc = bel.incumbent()
    labs, names = [], []
    for path in a.csv:
        rep = validate(path)
        if not rep.ok:
            print(f"ERREUR: {path}: CSV invalide:\n{rep.format()}", file=sys.stderr)
            return 2
        lab = pb.labels_from({e: r[0] for e, r in rep.rows.items()}, {e: r[1] for e, r in rep.rows.items()})
        if lab is None:
            print(f"ERREUR: {path}: familles différentes de la partition candidate.", file=sys.stderr)
            return 2
        labs.append(lab)
        names.append(path)
    print(f"Décodage : {mode_text(bel)}")
    if inc is not None:
        print(f"Référence : #{inc['n']}, score public {inc['score']:.6f}")
    for path, x in zip(names, bel.assess(labs, n_rep=a.rep, risk=a.risk)):
        print(f"\n{path}")
        if inc is not None:
            for d in describe(pb, inc["lab"], x["lab"]):
                print(f"  - {d}")
        print(f"  score public attendu {x['exp_public']:.3f} [10 % : {x['pub_lo']:.3f} ; 90 % : {x['pub_hi']:.3f}] ; "
              f"score privé attendu {x['exp_private']:.3f}")
        if inc is not None:
            print(f"  P(le public monte) {x['p_adopt']:.2f} ; P(score identique) {x['p_tie']:.2f} ; "
                  f"P(baisse) {max(0.0, 1 - x['p_adopt'] - x['p_tie']):.2f}")
            print(f"  si le public monte : gain privé attendu {x['gain_if_adopted']:+.4f}, "
                  f"risque que le privé baisse {x['p_harm_if_adopted']:.0%}")
            print(f"  valeur du créneau (gain privé espéré) : {x['V']:+.5f} ; P(meilleure que la référence sur "
                  f"toutes les paires) {x['p_right']:.2f}")
    return 0


def cmd_status(a) -> int:
    hist = read_history(Path(a.state_dir))
    info = slots_info(hist, a.deadline)
    print(f"Créneaux utilisés : {info['used']} ; prochain possible à {info['next']:%d/%m %H:%M} ; "
          f"restants avant {info['deadline']:%d/%m %H:%M} : {info['left']}")
    best, kept = None, None
    for h in hist:
        if h.get("score") is not None and (best is None or h["score"] > best + 1e-12):
            best, kept = h["score"], h["n"]
    for h in hist:
        s = h.get("score")
        flag = "  <- meilleur score public (retenue)" if h["n"] == kept else ""
        print(f"#{h['n']:02d} {h['ts'][:19]} {'score ' + format(s, '.6f') if s is not None else ('EN ATTENTE' if h.get('status') == 'pending' else 'ÉCHEC'):16s} "
              f"{h['csv']}{flag}")
    return 0


def cmd_simulate(a) -> int:
    from . import simulate as sim
    kw = dict(slots=a.slots, q_true=a.q_true, q_model=a.q, chains=a.chains, sweeps=a.sweeps,
              decimals=a.decimals, greedy=not a.no_greedy, scenario=a.scenario)
    if a.sizes:
        kw["sizes"] = tuple(int(x) for x in a.sizes.split(","))
    res = sim.run_many(a.worlds, jobs=a.jobs, seed0=a.seed0, **kw)
    text = sim.report(res, a.slots)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{k: v for k, v in r.items()} for r in res]), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3.12 -m model_equality.strategy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--state-dir", default=str(STATE))
        p.add_argument("--pairs", default=None,
                       help="CSV a,b,p_model, facultatif (mémorisé) ; une ligne endpoint-contre-un-membre vaut "
                            "pour toute la grappe")
        p.add_argument("--candidate", default=None, help="partition candidate endpoint,family,model (mémorisée)")
        p.add_argument("--q", type=float, default=0.3, help="part publique des paires (défaut 0,3)")
        p.add_argument("--step", type=float, default=SCORE_STEP,
                       help="pas d'arrondi du score du serveur (défaut 0,05) ; 0 : lu sur les décimales seulement")
        p.add_argument("--pi-hidden", type=float, default=0.12,
                       help="probabilité plancher qu'un membre d'une grappe soit un modèle distinct indétectable")
        p.add_argument("--p-missing", type=float, default=0.3,
                       help="sans mesure : probabilité qu'un endpoint seul appartienne à une grappe donnée")
        p.add_argument("--p-twin", type=float, default=0.05,
                       help="sans mesure : probabilité que deux endpoints seuls servent le même modèle")
        p.add_argument("--p-merge", type=float, default=0.05,
                       help="sans mesure : probabilité que deux grappes de la candidate soient le même modèle")
        p.add_argument("--no-anchor", action="store_true",
                       help="a priori par paires pur (sans l'ancrage sur les grappes de la candidate)")
        p.add_argument("--p-lo", type=float, default=0.03)
        p.add_argument("--p-hi", type=float, default=0.98)
        p.add_argument("--chains", type=int, default=512)
        p.add_argument("--sweeps", type=int, default=10)
        p.add_argument("--rep", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--risk", type=float, default=2.0, help="aversion à la perte pour les changements groupés")
        p.add_argument("--v-min", type=float, default=1e-4, help="valeur minimale d'un test pour proposer")
        p.add_argument("--max-moves", type=int, default=3,
                       help="nombre maximal de changements élémentaires par proposition (1 = un seul endpoint)")
        p.add_argument("--deadline", default=DEADLINE, help="fin de l'épreuve, heure de Paris")

    p = sub.add_parser("next", help="prochaine partition à soumettre et ce qu'elle teste")
    common(p)
    p.add_argument("--out", default=None, help="CSV proposé (défaut strategy/out/next/partition.csv)")
    p.add_argument("--probe-families", action="store_true", help="proposer la sonde 'familles seules'")
    p.set_defaults(fn=cmd_next)
    p = sub.add_parser("update", help="enregistre les scores reçus et les interprète")
    common(p)
    p.add_argument("--submit-dir", default=str(SUBMIT_DIR))
    p.add_argument("--csv", default=None, help="soumission faite hors de submit.py")
    p.add_argument("--score", type=float, default=None)
    p.add_argument("--failed", action="store_true", help="envoi sans score (compte quand même un créneau)")
    p.add_argument("--ts", default=None, help="horodatage ISO de l'envoi (défaut : maintenant)")
    p.add_argument("--note", default=None)
    p.set_defaults(fn=cmd_update)
    p = sub.add_parser("final", help="partition retenue et estimation du score privé")
    common(p)
    p.set_defaults(fn=cmd_final)
    p = sub.add_parser("assess", help="ce qu'apporterait une partition donnée si on la soumettait maintenant")
    common(p)
    p.add_argument("csv", nargs="+", help="partition(s) endpoint,family,model à évaluer")
    p.set_defaults(fn=cmd_assess)
    p = sub.add_parser("status", help="créneaux et historique")
    p.add_argument("--state-dir", default=str(STATE))
    p.add_argument("--deadline", default=DEADLINE)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("simulate", help="compare les stratégies sur des vérités synthétiques")
    p.add_argument("--scenario", choices=("merge", "split"), default="split",
                   help="split : leurres reconnus groupés, le reste seul (situation réelle) ; "
                        "merge : la meilleure estimation fusionne les variantes indétectables")
    p.add_argument("--worlds", type=int, default=100)
    p.add_argument("--slots", type=int, default=20)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--q", type=float, default=0.3, help="part publique supposée par l'outil")
    p.add_argument("--q-true", type=float, default=0.3, help="part publique réelle du monde simulé")
    p.add_argument("--chains", type=int, default=256)
    p.add_argument("--sweeps", type=int, default=8)
    p.add_argument("--decimals", type=int, default=None, help="arrondi du score renvoyé par le serveur simulé")
    p.add_argument("--sizes", default=None, help="tailles des familles, ex. 11,11,11")
    p.add_argument("--no-greedy", action="store_true")
    p.add_argument("--out", default=None, help="JSON des résultats par monde")
    p.set_defaults(fn=cmd_simulate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
