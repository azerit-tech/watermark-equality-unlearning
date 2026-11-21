"""Build the hypothesis list for base-model identification (every cached candidate x default template
x default-system-prompt variants).  Writes hyps/ident_all.jsonl.

  python ident_hyps.py [--out hyps/ident_all.jsonl]

Naming: <model_tag>__<variant>.  ident_report.py groups variants by <model_tag>.
"""
import argparse
import json
import os

WB = "/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb"
NOSYS = f"file:{WB}/scorer/templates/chatml_nosys.jinja"
TEMPS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.5]
QWEN_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
HELPFUL = "You are a helpful assistant."

# (tag, repo, lineage, variants)   variant = (name, extra hypothesis fields)
D = ("def", {})
CANDS = [
    ("smol2_135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "cosmo2", [D, ("nosys", {"chat_template": NOSYS})]),
    ("smol2_360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "cosmo2", [D, ("nosys", {"chat_template": NOSYS})]),
    ("smol2_1p7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "cosmo2", [D, ("nosys", {"chat_template": NOSYS})]),
    ("smol1_135m", "HuggingFaceTB/SmolLM-135M-Instruct", "cosmo2", [D]),
    ("smol1_360m", "HuggingFaceTB/SmolLM-360M-Instruct", "cosmo2", [D]),
    ("smol1_1p7b", "HuggingFaceTB/SmolLM-1.7B-Instruct", "cosmo2", [D]),
    ("smol3_3b", "HuggingFaceTB/SmolLM3-3B", "llama3",
     [D, ("nothink", {"template_kwargs": {"enable_thinking": False}})]),
]
for tag, repo in [("qwen25_0p5b", "Qwen/Qwen2.5-0.5B-Instruct"), ("qwen25_1p5b", "Qwen/Qwen2.5-1.5B-Instruct"),
                  ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"), ("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct"),
                  ("qwen25_7b_base", "Qwen/Qwen2.5-7B")]:
    CANDS.append((tag, repo, "qwen2", [D, ("qwensys", {"system_prompt": QWEN_SYS}),
                                         ("helpful", {"system_prompt": HELPFUL}),
                                         ("nosys", {"chat_template": NOSYS})]))
for tag, repo in [("qwen3_0p6b", "Qwen/Qwen3-0.6B"), ("qwen3_1p7b", "Qwen/Qwen3-1.7B"),
                  ("qwen3_4b", "Qwen/Qwen3-4B"), ("qwen3_8b", "Qwen/Qwen3-8B")]:
    CANDS.append((tag, repo, "qwen2", [("nothink", {"template_kwargs": {"enable_thinking": False}}), D]))
for tag, repo in [("llama32_1b", "unsloth/Llama-3.2-1B-Instruct"), ("llama32_3b", "unsloth/Llama-3.2-3B-Instruct"),
                  ("llama31_8b", "unsloth/Llama-3.1-8B-Instruct")]:
    CANDS.append((tag, repo, "llama3", [D, ("date2407", {"template_kwargs": {"date_string": "26 Jul 2024"}})]))
CANDS += [
    ("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama2", [D]),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct", "olmo2", [D]),
    ("olmo2_7b", "allenai/OLMo-2-1124-7B-Instruct", "olmo2", [D]),
    ("granite33_2b", "ibm-granite/granite-3.3-2b-instruct", "granite", [D]),
    ("phi35_mini", "microsoft/Phi-3.5-mini-instruct", "llama2", [D]),
    ("phi4_mini", "microsoft/Phi-4-mini-instruct", "o200k", [D]),
    ("gemma3_270m", "unsloth/gemma-3-270m-it", "gemma3", [D]),
    ("gemma3_1b", "unsloth/gemma-3-1b-it", "gemma3", [D]),
    ("gemma3_4b", "unsloth/gemma-3-4b-it", "gemma3", [D]),
    ("gemma2_2b", "unsloth/gemma-2-2b-it", "gemma2", [D]),
    ("gemma2_9b", "unsloth/gemma-2-9b-it", "gemma2", [D]),
    ("mistral_7b", "unsloth/mistral-7b-instruct-v0.3", "mistral", [D]),
]
BIG = ("7b", "8b", "9b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyps", "ident_all.jsonl"))
    ap.add_argument("--lineages", help="comma list: keep only these tokenizer lineages")
    ap.add_argument("--dtype", default="bfloat16", help="float16 is ~10x faster on Turing (RTX 8000)")
    a = ap.parse_args()
    keep = set(a.lineages.split(",")) if a.lineages else None
    n = 0
    with open(a.out, "w") as fh:
        for tag, repo, lin, variants in CANDS:
            if keep and lin not in keep:
                continue
            for vname, extra in variants:
                h = {"name": f"{tag}__{vname}", "model": repo, "dtype": a.dtype, "temps": TEMPS, "topk": 20,
                     "finish_policy": "length", "batch_tokens": 4096 if any(b in tag for b in BIG) else 8192}
                if tag.startswith("gemma2"):
                    h["attn_impl"] = "eager"
                h.update(extra)
                fh.write(json.dumps(h) + "\n")
                n += 1
    print(f"{n} hypotheses -> {a.out}")


if __name__ == "__main__":
    main()
