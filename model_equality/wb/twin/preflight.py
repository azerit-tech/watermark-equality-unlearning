"""Fail fast before using GPUs: models, corpora, probes, imports."""
from __future__ import annotations

import argparse
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="Q,O")
    a = ap.parse_args()
    from .run import PROBES, hyps_for
    from .spec import PLAN, family_specs, load_probes
    from .train_lora import load_corpus
    import mescore.hyp  # noqa
    import peft  # noqa
    import transformers  # noqa
    hf = os.environ["HF_HOME"]
    errs = []
    probes = load_probes(PROBES)
    errs += [f"probe {p} missing" for p in PLAN if p not in probes]
    specs = family_specs("/tmp")
    models = set()
    corpora = set()
    for f in a.fams.split(","):
        s = specs[f]
        models.add(s["model"])
        models |= {h["model"] for h in hyps_for(f, s, "/tmp")}
        corpora |= {c["data"] for c in s["loras"].values()}
    for m in sorted(models):
        d = glob.glob(f"{hf}/hub/models--{m.replace('/', '--')}/snapshots/*/")
        if not d or not (glob.glob(d[0] + "*.safetensors") and os.path.exists(d[0] + "tokenizer_config.json")):
            errs.append(f"model {m} not in cache")
    for c in sorted(corpora):
        try:
            n = len(load_corpus(c))
            print(f"corpus {c}: {n} examples")
        except Exception as e:
            errs.append(f"corpus {c}: {e}")
    if errs:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(errs), file=sys.stderr)
        sys.exit(3)
    print(f"preflight ok: models {sorted(models)}")


if __name__ == "__main__":
    main()
