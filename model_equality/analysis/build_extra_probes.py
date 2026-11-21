#!/usr/bin/env python3
"""Supplementary probes written after the R1_a checkpoint (same schema as probes/probes.jsonl).

    python3.12 model_equality/analysis/build_extra_probes.py     # writes analysis/probes_extra.jsonl

Why: on the ref- bench (SmolLM2-360M) the only streams that behave as repeated random draws are the
"continue the log" probes with a small alphabet (SL01, SL06).  With a comma separator an item costs two
tokens ("ĠT" + ","), i.e. 128 draws per 256-token answer; with single spaces an item is one token in
every BPE vocabulary we looked at, i.e. up to 256 draws per answer.  These variants are candidates only:
they have to pass the same selection as the others.

The file is merged into the bank by analysis.data.load_probes and by probes/build_waves.py.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent


def seq(alpha, n, seed, sep=" "):
    rng = random.Random(seed)
    return sep.join(rng.choice(alpha) for _ in range(n))


H = ("Flux dilué à séparateur espace : un token par item, donc jusqu'à 256 tirages par réponse au lieu de 128. "
     "Même mécanique que SL01 / SL06 (copie courte du journal puis flux aléatoire).")
R = ("Température : pente κ. Watermark de contexte 1 : le token précédent est l'item précédent, donc le décalage "
     "dépend du contexte et n'est PAS retiré par le double centrage d'ordre 1 ; à vérifier sur ref-06. "
     "System prompt : dilution par le contexte long.")
SPECS = [
    ("SX01_coin_space", "coin flips (H or T)", ["H", "T"], 200, 201),
    ("SX06_abcd_space", "random letters among A, B, C and D", list("ABCD"), 200, 206),
]


# Long free-text probes for the ep- battery (added 19:20).  The local replay track counts tokens, the budget counts
# responses: a 256-token answer costs one response like a 64-token one.  Two prompts, repeated, with open first tokens.
LONG_FREE = [
    ("LX01_story_long", "Write a long, detailed story about a clockmaker who discovers a hidden door at the back of "
                        "her workshop."),
    ("LX02_essay_long", "Write a detailed essay on whether cities should ban cars from their centres. Give arguments "
                        "on both sides, then your own conclusion."),
]


def build():
    out = [{"id": pid, "category": "L_domain", "group": pid, "lang": "en", "priority": 1,
            "phases": ["ref_R2", "ep_battery"], "max_tokens": 256, "parser": {"type": "free"}, "wm_block": None,
            "est_prompt_tokens": int(len(text) / 4) + 30,
            "hypothesis": "Texte libre long et répété : matière du rejeu local (fp8, LoRA forts) et arbre de préfixes.",
            "resists": "Leurres jugés par le rejeu au T propre de l'endpoint ; répétitions = signature du watermark.",
            "text": text} for pid, text in LONG_FREE]
    for pid, what, alpha, n, seed in SPECS:
        text = (f"Below is a log of {what}, separated by single spaces.\n\n{seq(alpha, n, seed)}\n\n"
                f"Continue the log with 250 more entries in exactly the same format. Output only the new entries, "
                f"on one line, with no introduction and no comment.")
        out.append({"id": pid, "category": "SL_stream_diluted", "group": pid, "lang": "en", "priority": 2,
                    "phases": ["ref_R2", "ep_battery"], "max_tokens": 256,
                    "parser": {"type": "stream", "alphabet": alpha, "sep": "space"}, "wm_block": pid,
                    "est_prompt_tokens": int(len(text) / 2.6), "hypothesis": H, "resists": R, "text": text})
    return out


if __name__ == "__main__":
    path = HERE / "probes_extra.jsonl"
    rows = build()
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{path}: {len(rows)} probes ({', '.join(r['id'] for r in rows)})")
