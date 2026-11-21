"""Teacher-forcing scorer for Model Equality Testing (GPU track, tasks T1/T4).

Modules
-------
data     responses loading (simple JSONL or the API ledger), context building
hyp      hypothesis spec + model construction (dtype, quant emulation, noise, LoRA scale)
score    forward pass + per-token stats; CLI: python -m mescore.score
store    on-disk format (npz + json sidecar), summaries, comparisons; numpy only
worker   multi-GPU, idempotent, resumable runner over a list of hypotheses
selftest end-to-end test on a small open model
"""
FORMAT_VERSION = 1
