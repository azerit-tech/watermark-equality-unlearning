#!/usr/bin/env python3
"""Writes the ref- wave plans (client plan format: a list of blocks) into waves/. No network access.

    python3 build_waves.py

A block is {"endpoints", "prompts": [{"id","text"}], "repeats", "max_tokens", "rep_start"}; the client runs the
blocks of a file in order, and inside a block rep-major, then prompt, then endpoint. Blocks are therefore cut in
small rounds of repetitions, so that an interruption leaves every probe and endpoint evenly covered.
max_tokens always equals the value in probes.jsonl (it is part of the ledger key).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PROBES = {p["id"]: p for p in (json.loads(l) for l in (HERE / "probes.jsonl").read_text(encoding="utf-8").splitlines())}
# Supplementary probes written after the R1_a checkpoint (analysis/build_extra_probes.py); the bank wins on a clash.
_EXTRA = HERE.parent / "analysis" / "probes_extra.jsonl"
if _EXTRA.exists():
    for _l in _EXTRA.read_text(encoding="utf-8").splitlines():
        if _l.strip():
            PROBES.setdefault(json.loads(_l)["id"], json.loads(_l))
ALL = "ref-01..ref-09"
HARD = "ref-01,ref-02,ref-09"          # plain, fp8, LoRA 0.3: the pairs that need depth
EP = "ep-01..ep-33"                    # scored endpoints: every accepted answer is charged (100 per endpoint)


def rounds(ids, start, stop, step, endpoints=ALL):
    """Blocks covering repetitions start..stop-1 of the probes `ids`, `step` repetitions per round."""
    blocks = []
    for r in range(start, stop, step):
        n = min(step, stop - r)
        by_mt = {}
        for pid in ids:
            by_mt.setdefault(PROBES[pid]["max_tokens"], []).append(pid)
        for mt in sorted(by_mt, reverse=True):
            blocks.append({"endpoints": endpoints, "max_tokens": mt, "rep_start": r, "repeats": n,
                           "prompts": [{"id": pid, "text": PROBES[pid]["text"]} for pid in by_mt[mt]]})
    return blocks


def n_endpoints(spec):
    return {ALL: 9, EP: 33}.get(spec) or len(spec.split(","))


def ep_budget(waves):
    """Scored answers per ep- endpoint over a set of plans (identical ledger keys counted once)."""
    keys = set()
    for blocks in waves.values():
        for b in blocks:
            if b["endpoints"].startswith("ep-"):
                for p in b["prompts"]:
                    if p["id"] != "X01_oversize":          # rejected, not charged
                        for r in range(b["rep_start"], b["rep_start"] + b["repeats"]):
                            keys.add((b["endpoints"], p["id"], r, b["max_tokens"]))
    return len({k[1:] for k in keys})


def cost(blocks):
    """(responses, minutes of API at concurrency 4), latency = 6.2 s + 0.053 s per generated token."""
    n = sec = 0
    for b in blocks:
        k = len(b["prompts"]) * b["repeats"] * n_endpoints(b["endpoints"])
        n += k
        sec += k * (6.2 + 0.053 * b["max_tokens"])
    return n, sec / 4 / 60


BREADTH = ["SL02_digits_long", "S02_digits", "SL01_coin_long", "S01_coin", "SL06_abcd_long", "S05_dice",
           "F01_ab_pref", "F02_yesno", "F11_genre_fr", "F03_rating", "F20_tools_a", "F05_mcq"]
# Revision of Saturday 18:15, after the first round of R1_a (1 answer per probe and endpoint, bench = SmolLM2-360M):
#   * S02, S01, S05 (short streams with a "like this: 4, 0, 7" example): the model echoes the example and stops
#     (S02 0/9, S01 1/9, S05 2/9 parsable) -> removed from every file not yet started;
#   * F11 (0/9) and F01 (1/9: stops after "1: A" or copies the items) -> removed; F03: half the answers stop after
#     line 1 and the others are constant ("4" everywhere) -> removed;
#   * SL02: parsable but it is a verbatim copy of the log of the prompt (no entropy on base-like endpoints) -> removed;
#   * held: SL01 (copies ~10-20 items, then a genuine stream), SL06, F05, F20, F02 (6/9).
HELD = ["SL01_coin_long", "SL06_abcd_long", "F05_mcq", "F20_tools_a", "F02_yesno"]
# Second screening: candidates without an example to echo, a second coin log, open lists, other forms.
# SX01 / SX06 (analysis/probes_extra.jsonl): space-separated logs, one token per item, up to 256 draws per answer.
# Revision 19:10 (12 repetitions of the five held probes on the 9 endpoints: only LoRA 1.5 and the watermark stand
# out, forms have too few draws): the four extra forms are dropped from the screening, and the four free-text
# prompts of R1_c are extended to the six other endpoints (lures and strong LoRA on repeated free text).
SCREEN = ["SX01_coin_space", "SX06_abcd_space", "SL11_coin_long_b", "S24_numbers100", "S21_words", "S22_animals"]
FREE = ["tp_story", "C08_story", "L24_tweet", "L22_opinion"]
OTHERS = "ref-03,ref-04,ref-05,ref-06,ref-07,ref-08"
WAVES = {
    # 12 candidates x 3 repetitions: format compliance on the 9 endpoints, first look at every question.
    "R1_a_breadth.json": rounds(BREADTH, 0, 3, 1),
    # Core depth: the five probes that hold, to 12 repetitions on the 9 endpoints.
    "R1_b_core.json": rounds(HELD, 3, 12, 3),
    # Hard pairs only (plain, fp8, LoRA 0.3): two streams and the MCQ form to 40 repetitions (size of epsilon,
    # split-half null of ref-01), then the two other forms to 30.
    # Third part (added 18:55): four free-text prompts repeated 25 more times on the same three endpoints. Free text
    # is what the local replay track needs (it separates fp8 there), and the prefix-tree view (analysis --tree) makes
    # repeated free prompts usable in black box: this measures what they give on the hard pairs.
    "R1_c_hardpairs.json": (rounds(["SL01_coin_long", "SL06_abcd_long", "F05_mcq"], 12, 40, 7, endpoints=HARD)
                            + rounds(["F02_yesno", "F20_tools_a"], 12, 30, 9, endpoints=HARD)
                            + rounds(["tp_story", "C08_story", "L24_tweet", "L22_opinion"], 3, 28, 5, endpoints=HARD)),
    # Free-text prompts to 12 repetitions on the six endpoints R1_c does not cover, then a second screening
    # (3 repetitions x 9 endpoints) of 6 stream / list candidates for the R2 / ep- battery.
    "R1_d_second.json": rounds(FREE, 3, 12, 3, endpoints=OTHERS) + rounds(SCREEN, 0, 3, 1),
    # ref- baseline of the three new cut-diagnostic probes of wave E1 (tp_story at 64 tokens is already in the ledger).
    "R1_d2_e1_baseline.json": rounds(["E01_identity_long", "E02_pi_digits", "E03_story_zh"], 0, 2, 1),
    # Optional filler (outside the budget): keeps the API busy while R1 is analysed. Delete it if R2 is ready.
    "R1_e_filler.json": rounds(HELD, 12, 16, 2),
}

# ---- Wave R2 (ref- only, written Saturday 19:25).  What R1 showed: at 60 to 75 responses the discrete probes separate
# only LoRA 1.5 and the watermark; streams carry no signal for fp8 or the weak LoRA, neither here nor in the local
# replay.  R2 therefore rehearses, at the real budget, the battery planned for ep- (91 responses after the 9 of E1):
# 7 repeated non-stream prompts x 13 repetitions.  Repetition indices start at 100: an R2 response is recognisable,
# and repetitions 100-112 / 113-125 / 150-162 are three disjoint "budgets" of the same endpoint.
BATTERY = ["tp_story", "LX01_story_long", "LX02_essay_long", "C08_story", "C01_en", "E01_identity_long", "F20_tools_a"]
IDENT = ["D01_identity", "D07_intro", "D15_adjectives", "D06_persona"]
HARD4 = "ref-01,ref-02,ref-04,ref-09"      # plain, fp8, LoRA 0.7 (positive control), LoRA 0.3
REST5 = "ref-03,ref-05,ref-06,ref-07,ref-08"
R2_WAVES = {
    # 1. One real budget of the battery on the 9 endpoints (819 responses).
    "R2_a_battery.json": rounds(BATTERY, 100, 113, 1),
    # 2. Second and third budgets on plain / fp8 / LoRA 0.7 / LoRA 0.3 (560): null of ref-01 against itself,
    #    effect size of fp8 and of the weak LoRA at 33 repetitions per prompt.
    "R2_b_replicate_hard.json": rounds(BATTERY, 113, 133, 2, endpoints=HARD4),
    # 3. Second budget on the five other endpoints (455): every endpoint then has two disjoint budgets
    #    (selection on one, measure on the other; lure labels at the real budget).
    "R2_c_replicate_rest.json": rounds(BATTERY, 113, 126, 1, endpoints=REST5),
    # 4. Identity / self-description prompts at depth on the hard endpoints (480, short answers): the only place
    #    where a LoRA showed a concentrated effect ("My name is Emily ..." on ref-04 / ref-07). Does 0.3 show it?
    "R2_d_identity_depth.json": rounds(IDENT, 100, 130, 3, endpoints=HARD4),
    # 5. Third budget on the 9 endpoints (819): spread of the MCC between budgets.
    "R2_e_battery_third.json": rounds(BATTERY, 150, 163, 1),
}

# ---- First ep- diagnostic (wave E1). SCORED BUDGET: exactly 9 answers per endpoint, 297 in total.
# Written to waves/ep_pending/, not waves/: "E1" sorts before "R1", and a runner sweeping waves/ for the ref- bench
# must not reach scored endpoints by accident. Move the files up when the spend is decided.
E2_ENDPOINTS = EP    # every sampled ep- endpoint; replaced by an explicit list if E1 shows deterministic ones
E1_FIRST = ["tp_story", "E01_identity_long", "E02_pi_digits", "E03_story_zh"]
E1_SECOND = ["tp_story", "E02_pi_digits", "E03_story_zh"]
EP_WAVES = {
    # Free: the over-long prompt is rejected (HTTP 400, "not counted"). Reachability check of the 33 endpoints.
    "E1_1_free_oversize.json": rounds(["X01_oversize"], 0, 1, 1, endpoints=EP),
    # Canary: one charged answer on ep-01. Same ledger key as the first item of E1_3, which then skips it.
    "E1_2_canary.json": rounds(["tp_story"], 0, 1, 1, endpoints="ep-01"),
    # 7 answers per endpoint, all certainly cut: determinism (tp_story twice), tokenizer lineage, identity, style.
    "E1_3_cut.json": rounds(E1_FIRST, 0, 1, 1, endpoints=EP) + rounds(E1_SECOND, 1, 2, 1, endpoints=EP),
    # 2 answers per endpoint: system prompt leak, asked twice (a role text that changes between the two = persona).
    "E1_4_leak.json": rounds(["D03_leak_repeat"], 0, 2, 1, endpoints=EP),
    # ---- ep- battery (wave E2, written Saturday 19:30). SCORED BUDGET: 70 answers per endpoint (2310 in total);
    # with the 9 of E1 that makes 79, i.e. a reserve of 21 per endpoint. No stream: the 7 repeated non-stream prompts
    # of BATTERY x 10 repetitions (100-109). Three files in priority order, each cut in rounds of one repetition, so
    # an interruption leaves all endpoints evenly covered: 28 answers per endpoint after E2_a (the local replay asks
    # for 30 to 35 with a simulated reference), 49 after E2_b, 70 after E2_c (about 60 without it).
    # E2_ENDPOINTS is narrowed after reading E1 (deterministic endpoints get distinct prompts instead: E2_det).
    "E2_a_battery.json": rounds(BATTERY, 100, 104, 1, endpoints=E2_ENDPOINTS),
    "E2_b_battery.json": rounds(BATTERY, 104, 107, 1, endpoints=E2_ENDPOINTS),
    "E2_c_battery.json": rounds(BATTERY, 107, 110, 1, endpoints=E2_ENDPOINTS),
}

if __name__ == "__main__":
    out = HERE / "waves"
    out.mkdir(exist_ok=True)
    tot_n = tot_m = 0
    (out / "ep_pending").mkdir(exist_ok=True)
    for name, blocks in list(WAVES.items()) + list(R2_WAVES.items()) + list(EP_WAVES.items()):
        path = out / "ep_pending" / name if name in EP_WAVES else out / name
        text = json.dumps(blocks, ensure_ascii=False, indent=1)
        if not path.exists() or path.read_text(encoding="utf-8") != text:   # never rewrite an unchanged plan
            path.write_text(text, encoding="utf-8")
        n, m = cost(blocks)
        tot_n += n
        tot_m += m
        print(f"{name}: {len(blocks)} blocks, {n} responses, about {m:.0f} min")
    used = ep_budget(EP_WAVES)
    assert used <= 80, f"ep- plans spend {used} answers per endpoint: the reserve of 20 is broken"
    print(f"ep- plans: {used} scored answers per endpoint (E1 + E2), reserve {100 - used}")
    print(f"total: {tot_n} responses, about {tot_m:.0f} min (upper bound: assumes every answer reaches max_tokens)")
