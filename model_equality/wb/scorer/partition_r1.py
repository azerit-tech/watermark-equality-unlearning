"""White-box-only partition of ref- (rules written BEFORE computing the R1 statistics and before
scoring against practice_answers.csv; they were written after the E1/E3 results were seen).

  python partition_r1.py R1.json WM.txt [ANSWERS.csv]

Rules (base hypothesis = SmolLM2-360M-Instruct, default template, T0 = 0.7):
  R1  watermark: max default-key detector z >= 5 (8 configs, Bonferroni-ish)  -> same model as base
  R2  fp8: LLR z of the chosen fp8 emulation vs base >= 4                       -> own model
  R3  unexplained weight residual: unigram residual energy z >= 4               -> weight-change candidate
      otherwise                                                                  -> same model as base
  R4  candidates are merged only if direction cosine >= 0.7 and energy ratio within [0.7, 1.43];
      otherwise each candidate is its own model.
"""
import json
import sys


def partition(r1: dict, wm: dict) -> dict:
    q1, q2, cos = r1["Q1"]["per_endpoint_chosen_emulation"], r1["Q2_Q3"], r1["unigram_cosines"]
    lab, why, cands = {}, {}, []
    for ep in sorted(q2):
        wz = max(v for k, v in wm.get(ep, {}).items() if k != "n_pairs") if ep in wm else 0.0
        if wz >= 5:
            lab[ep], why[ep] = "base", f"R1 wm z={wz}"
        elif q1[ep]["z"] >= 4:
            lab[ep], why[ep] = f"fp8:{ep}", f"R2 fp8 z={q1[ep]['z']}"
        elif q2[ep]["unigram_energy_z"] >= 4:
            cands.append(ep)
            why[ep] = f"R3 energy z={q2[ep]['unigram_energy_z']}"
        else:
            lab[ep], why[ep] = "base", f"none (fp8 z={q1[ep]['z']}, energy z={q2[ep]['unigram_energy_z']})"
    for ep in cands:
        lab[ep] = f"w:{ep}"
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            c = cos.get(f"{a}|{b}", cos.get(f"{b}|{a}"))
            ea, eb = q2[a]["unigram_energy"], q2[b]["unigram_energy"]
            if c is not None and c >= 0.7 and 0.7 <= ea / eb <= 1.43:
                lab[b] = lab[a]
    return {"labels": lab, "why": why}


def main():
    r1 = json.load(open(sys.argv[1]))
    wm = {}
    for line in open(sys.argv[2]):
        ep, js = line.split(" ", 1)
        wm[ep] = json.loads(js)
    p = partition(r1, wm)
    print(json.dumps(p, indent=1))
    if len(sys.argv) > 3:
        sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else ".")
        import csv
        truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(sys.argv[3]))}
        eps = sorted(p["labels"])
        tp = tn = fp = fn = 0
        for i, a in enumerate(eps):
            for b in eps[i + 1:]:
                ps, ts = p["labels"][a] == p["labels"][b], truth[a] == truth[b]
                tp += ps and ts; tn += (not ps) and (not ts); fp += ps and not ts; fn += (not ps) and ts
        d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        print(json.dumps({"tp": tp, "tn": tn, "fp": fp, "fn": fn, "mcc": round(max(0.0, (tp * tn - fp * fn) / d) if d else 0.0, 3)}))


if __name__ == "__main__":
    main()
