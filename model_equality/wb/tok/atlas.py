"""Build the tokenizer atlas: out/atlas.json and out/atlas.md.

Usage: .venv/bin/python atlas.py
Lineages:
  * bytes   : tokenizer.json identical byte for byte
  * core    : same vocab+merges+normalizer+pre-tokenizer+non-special added tokens
  * lineage : identical token ids on a probe corpus (ledger texts + multilingual/code strings);
              this is the grouping that matters for counting tokens of generated text.
"""
from __future__ import annotations

import json
from collections import defaultdict

import toklib as T

PROBE = [
    "Hello world! The quick brown fox jumps over the lazy dog.",
    "  leading spaces\n\n\ttabs and\r\nCRLF   trailing  ",
    "Ça coûte 12,50 € — naïve façade, Größe, ñandú, øre.",
    "東京は日本の首都です。北京是中国的首都。서울은 한국의 수도입니다.",
    "Здравствуйте, мир! مرحبا بالعالم שלום עולם नमस्ते दुनिया",
    "def f(x):\n    return [i**2 for i in range(x) if i % 3 == 0]  # comment\n",
    "1234567890 3.14159 2026-11-21 0x1F4A9 1e-9 100000000000",
    "emoji 🤖🎉👍🏽 and symbols ∑∫√∞ ≠ ≤ ≥ → ← ™ ©",
    "XK7-qp93-ZZ_m4 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ........ ----- =====",
]


FRIENDLY = {  # member -> lineage name (also used as representative)
    "qwen2.5-0.5b-instruct": "qwen2", "llama-3.2-1b-instruct": "llama3", "llama-2-7b-chat": "llama2-sp",
    "mistral-7b-instruct-v0.2": "mistral-v1", "mistral-7b-instruct-v0.3": "mistral-v3",
    "mistral-nemo-instruct": "tekken", "gemma-2-2b-it": "gemma1-2", "gemma-3-1b-it": "gemma3",
    "smollm2-135m-instruct": "smollm", "olmo-2-1b-instruct": "olmo2-cl100k", "phi-4-mini-instruct": "phi4-o200k",
    "stablelm-2-zephyr-1.6b": "stablelm2", "falcon3-1b-instruct": "falcon3", "granite-3.1-2b-instruct": "granite3",
    "h2o-danube3-500m-chat": "danube3", "gpt-neox-20b": "gpt-neox", "gpt2": "gpt2",
}


def gen_params(name: str) -> dict:
    p = T.HF / name / "generation_config.json"
    g = json.loads(p.read_text()) if p.exists() else {}
    return {k: g.get(k) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                                  "eos_token_id", "max_new_tokens")} | {"_present": p.exists()}


def template_summary(name: str) -> dict:
    try:
        s = T.render_chat(name, "USER_MSG")
        err = None
    except Exception as e:
        s, err = None, f"{type(e).__name__}: {e}"
    try:
        ssys = T.render_chat(name, "USER_MSG", system="SYS_MSG")
        sys_ok = ssys is not None and "SYS_MSG" in ssys
    except Exception:
        sys_ok = False
    cfg = json.loads((T.HF / name / "tokenizer_config.json").read_text())
    return {"rendered": s, "error": err, "supports_system": sys_ok,
            "default_system_injected": bool(s and len(s.split("USER_MSG")[0]) > 60),
            "prompt_tokens_hi": T.prompt_token_count(name, "Hi") if s else None,
            "add_bos_token": cfg.get("add_bos_token")}


def main():
    import toklib
    srcs = json.loads((T.HF / "_sources.json").read_text())
    names = T.models()
    texts = PROBE + [r["text"] for r in T.load_responses() if r.get("text")][:400]
    sig = {}
    for n in names:
        tk = T.tokenizer(n)
        ids = tuple(tuple(tk.encode(t, add_special_tokens=False).ids) for t in texts)
        sig[n] = T._sha(json.dumps(ids).encode())
    groups = defaultdict(list)
    for n in names:
        groups[sig[n]].append(n)
    lineages = {}
    for s, mem in sorted(groups.items(), key=lambda kv: sorted(kv[1])[0]):
        rep = next((k for k in FRIENDLY if k in mem), sorted(mem)[0])
        lid = f"L-{FRIENDLY.get(rep, rep)}"
        bytes_groups = defaultdict(list)
        core_groups = defaultdict(list)
        for m in mem:
            bytes_groups[T.bytes_hash(m)].append(m)
            core_groups[T.core_hash(m)].append(m)
        j = json.loads((T.HF / rep / "tokenizer.json").read_bytes())
        lineages[lid] = {
            "representative": rep, "members": sorted(mem),
            "vocab_size": sorted({T.tokenizer(m).get_vocab_size(with_added_tokens=True) for m in mem}),
            "model_type": (j.get("model") or {}).get("type"),
            "bytes_groups": sorted(sorted(v) for v in bytes_groups.values()),
            "core_groups": sorted(sorted(v) for v in core_groups.values()),
            "corpus_sig": s,
        }
    models = {}
    for n in names:
        lid = next(l for l, v in lineages.items() if n in v["members"])
        models[n] = {"repo": srcs[n]["repo"], "lineage": lid, "bytes_hash": T.bytes_hash(n),
                     "core_hash": T.core_hash(n), "gen": gen_params(n), "template": template_summary(n)}
    conf = confusability(lineages)
    failed = {n: i["errors"] for n, i in srcs.items() if not i.get("repo")}
    T.OUT.mkdir(exist_ok=True)
    (T.OUT / "atlas.json").write_text(json.dumps({"lineages": lineages, "models": models, "failed": failed,
                                                  "confusability": conf},
                                                 indent=1, ensure_ascii=False))
    write_md(lineages, models, failed, conf)
    print(f"{len(names)} models, {len(lineages)} lineages -> out/atlas.json, out/atlas.md")


DISCRIM = {
    "digits": "The numbers are 1234567 and 89012345, then 2026, 3.14159265358979, 1000000.",
    "zh": "从前，在一个小渔村里，住着一位老灯塔看守人。他每天晚上都点亮灯塔。",
    "fr": "Il était une fois un gardien de phare qui trouva une bouteille échouée sur la grève.",
    "code": "for (int i = 0; i < n; ++i) {\n        sum += a[i] * b[i];\n    }\n",
}


def confusability(lineages):
    """Share of ledger texts (English, mostly) on which two lineages give the same count,
    plus counts on discriminating probes."""
    import itertools
    reps = {l: v["representative"] for l, v in lineages.items()}
    texts = [r["text"] for r in T.load_responses() if r.get("text")]
    C = {l: [T.count(n, x) for x in texts] for l, n in reps.items()}
    pairs = []
    for x, y in itertools.combinations(reps, 2):
        s = sum(p == q for p, q in zip(C[x], C[y])) / max(1, len(texts))
        if s >= 0.5:
            pairs.append((x, y, round(s, 3)))
    probes = {l: {k: T.count(n, t) for k, t in DISCRIM.items()} for l, n in reps.items()}
    return {"n_texts": len(texts), "pairs": sorted(pairs, key=lambda p: -p[2]), "probes": probes}


def write_md(lineages, models, failed, conf=None):
    L = ["# Atlas des tokenizers", "",
         "Généré par `atlas.py`. Lignée = ids identiques sur un corpus de sondes (textes du registre + "
         "multilingue/code). `octets` = tokenizer.json identique octet pour octet.", "",
         "## Lignées", "", "| Lignée | V | type | membres (groupes identiques octet pour octet entre crochets) |",
         "|---|---|---|---|"]
    for lid, v in lineages.items():
        bg = " ".join("[" + ", ".join(g) + "]" for g in v["bytes_groups"])
        L.append(f"| {lid} | {'/'.join(map(str, v['vocab_size']))} | {v['model_type']} | {bg} |")
    L += ["", "## Modèles", "",
          "| Modèle | Dépôt | Lignée | do_sample | T | top_p | top_k | rep. pen. | tokens prompt « Hi » | system par défaut |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, m in models.items():
        g = m["gen"]
        t = m["template"]
        L.append(f"| {n} | {m['repo']} | {m['lineage']} | {g['do_sample']} | {g['temperature']} | {g['top_p']} | "
                 f"{g['top_k']} | {g['repetition_penalty']} | {t['prompt_tokens_hi']} | "
                 f"{'oui' if t['default_system_injected'] else 'non'} |")
    L += ["", "## Gabarits (message utilisateur `USER_MSG`, add_generation_prompt)", ""]
    seen = {}
    for n, m in models.items():
        r = m["template"]["rendered"]
        key = r or m["template"]["error"] or "(pas de gabarit)"
        seen.setdefault(key, []).append(n)
    for r, ns in seen.items():
        L += [f"**{', '.join(ns)}**", "", "```", r, "```", ""]
    if conf:
        L += ["## Lignées confondables par le compte de tokens", "",
              f"Part des {conf['n_texts']} textes du registre (anglais) où deux lignées donnent le même compte "
              "(seuil 0,5). Au-dessus de 0,95, l'empreinte par troncature ne les sépare pas sur de l'anglais : "
              "il faut des sorties en chiffres, en chinois, etc.", "", "| Lignée A | Lignée B | comptes identiques |", "|---|---|---|"]
        L += [f"| {a} | {b} | {s:.2f} |" for a, b, s in conf["pairs"]]
        ks = list(DISCRIM)
        L += ["", "Comptes sur des sondes discriminantes : " + "; ".join(f"`{k}` = {DISCRIM[k]!r}" for k in ks), "",
              "| Lignée | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
        L += [f"| {l} | " + " | ".join(str(v[k]) for k in ks) + " |" for l, v in conf["probes"].items()]
        L.append("")
    if failed:
        L += ["## Non récupérés", ""] + [f"- {n}: {e}" for n, e in failed.items()]
    (T.OUT / "atlas.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
