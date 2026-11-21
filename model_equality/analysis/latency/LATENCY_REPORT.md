# Latency channel: verdict

2026-11-21. Data: a copy of the ledger (ledger_copy.jsonl), 3,294 responses with status 200. No API calls were made.

## Conclusion

The latency channel does not tell the weights apart. Every endpoint is classed UNSURE. The rule was fixed in RULES.txt before it was applied to the ep- endpoints.

## Why

1. The response time is set by max_tokens, not by the model. When the server is idle (the early ref- runs), elapsed_s = 5.89 s + 53.75 ms × max_tokens, with a spread of about 0.08 s. This holds for every endpoint and every output length. For example, ep-01 took the same 23.9 s for an 11-character reply as its neighbours did for 160 characters at the same max_tokens. The time is either padded or spent generating up to max_tokens every time, so it carries no real per-token throughput.

2. The bench fails. The model is a robust log-time regression with 3-minute time slots, a max_tokens effect, and an endpoint effect plus its interaction with max_tokens; CIs come from a bootstrap over the slots. All of ref-02 to ref-09 fall between -0.44% and -0.25% of ref-01, with CIs of about ±0.25%. The fp8 endpoint (-0.29%), the LoRA endpoints (-0.27%, -0.25%, -0.34%) and the lures (-0.25% to -0.44%) overlap. A multi-LoRA slowdown or an fp8 speedup of even 1% would have been visible, and none is there.

3. The ep- runs took place under load (4 to 14 s above the floor, varying with the time slot), so they are noisier. After adjusting for the time slot and max_tokens, every ep- endpoint is within -1.0% to +1.5% of its family median. No endpoint meets the anomaly threshold (more than 3% with a CI that excludes ±1%). The per-token and first-token effects are also inside the noise. ep-03 has one CI that excludes 0 (+4.5 ms/token, and -0.8 s on the first token). That is what chance alone gives across 33 endpoints and several tests, and the bench offers no calibration for it.

4. The quick-look ms/char signal is a text artefact. Across endpoints it correlates at 0.95 with 1/(median length of the replies over 300 characters). Because elapsed_s is fixed by max_tokens, ms/char measures characters per token in the truncated output: a style feature, not a speed. On the bench it does not separate LoRA from the base group either (the ref-03 and ref-05 lures at 34, the ref-04 LoRA at 37, ref-01, ref-02 and ref-09 all at 52, reflecting the run mix).

## Limits

There is no tokenizer in the local cache, so this works in characters. finish_reason is always "stop", so it is unusable. The ep- runs never overlap the ref- runs in time, so the bench validates the method but not the loaded regime.

## Files

RULES.txt, latency_ep.csv, and the scripts load.py, fit.py, bench.py and apply_ep.py.
