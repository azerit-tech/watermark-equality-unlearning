"""Build wave-0 and fallback plan files from probes/probes.jsonl (ref- only)."""
import json
from pathlib import Path
here = Path(__file__).parent
ps = [json.loads(l) for l in open(here.parent / "probes" / "probes.jsonl")]
REF = "ref-01..ref-09"
def blocks(cat_prefix, repeats):
    return [{"endpoints": REF, "prompts": [{"id": p["id"], "text": p["text"]}],
             "repeats": repeats, "max_tokens": p["max_tokens"]}
            for p in ps if p["category"].startswith(cat_prefix)]
json.dump(blocks("X", 1), open(here / "W0a_X01.json", "w"), ensure_ascii=False, indent=1)
json.dump(blocks("D_", 2) + blocks("C_", 3), open(here / "W0b_D2_C3.json", "w"), ensure_ascii=False, indent=1)
json.dump(blocks("K_", 5), open(here / "W1_fallback_K5.json", "w"), ensure_ascii=False, indent=1)
