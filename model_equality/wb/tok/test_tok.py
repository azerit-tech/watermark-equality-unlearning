"""Sanity tests. Run: .venv/bin/python -m pytest -q test_tok.py  (or .venv/bin/python test_tok.py)"""
import toklib as T
import trunc

BASE = [
    "Once upon a time, in a small fishing village perched on the edge of a stormy sea, there lived an old "
    "lighthouse keeper named Elias who had spent forty years tending the great lamp every single night.",
    "Autumn arrives quietly, painting the maple trees in shades of amber and crimson while the air grows crisp "
    "and the days shorten, inviting long walks, warm soup, and evenings spent reading by the fire.",
    "The capital of Australia is Canberra, a planned city chosen as a compromise between Sydney and Melbourne, "
    "and it hosts Parliament House, the High Court, and many national institutions and museums.",
    "从前，在一个小渔村里，住着一位老灯塔看守人。他每天晚上都点亮灯塔，并且记录下每一艘经过的船只和每一次风暴的情况。",
    "Inventory 2026: item 1234567 costs 89.99, item 7654321 costs 1049.50, total 31415926 units shipped in 365 days.",
    "To compute the factorial of a number in Python you can write def fact(n): return 1 if n < 2 else n * fact(n-1), "
    "although for large inputs an iterative loop or math.factorial is faster and avoids recursion limits.",
]


def synth(model, mts=(16, 24, 32)):
    tk = T.tokenizer(model)
    rows = []
    for t in BASE:
        ids = tk.encode(t, add_special_tokens=False).ids
        for mt in mts:
            rows.append({"text": tk.decode(ids[:mt], skip_special_tokens=True), "max_tokens": mt})
    return rows


def test_unfinished():
    assert trunc.looks_unfinished("there lived a keeper named Jack who")
    assert not trunc.looks_unfinished("The capital is Canberra.")


def test_truth_ranks_first():
    atlas = T.load_lineages()
    reps = [v["representative"] for v in atlas["lineages"].values()]
    for truth in ["smollm2-135m-instruct", "qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct",
                  "gemma-3-1b-it", "mistral-7b-instruct-v0.3", "phi-3.5-mini-instruct"]:
        lid = atlas["models"][truth]["lineage"]
        res = trunc.analyse(synth(truth), reps)
        rep = atlas["lineages"][lid]["representative"]
        top = max(v["score"] for v in res.values())
        assert res[rep]["score"] == top and res[rep]["over"] == 0, (truth, list(res.items())[:2])
        # with Chinese + digits in the outputs, the cl100k-derived families separate
        if truth.startswith(("qwen", "llama")):
            assert sum(v["score"] == top for v in res.values()) == 1, (truth, [k for k, v in res.items() if v["score"] == top])


def test_prompt_count_smollm2():
    # system default injected by the SmolLM2 template
    assert T.prompt_token_count("smollm2-135m-instruct", "What is the capital of France?") == 37


if __name__ == "__main__":
    test_unfinished(); test_truth_ranks_first(); test_prompt_count_smollm2(); print("ok")
