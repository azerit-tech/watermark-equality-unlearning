"""Builds t1/bigtrain/{train_sg.py,train_tiles_all.py} from t1/train/train.py and t1/tiles2/train_tiles.py (run from TASK).
Changes: --val_fold -1 = train on ALL real train folds (no real validation); --seed; final.pt / ema_final.pt (EMA of the
LAST state) saved at every checkpoint; tiles: --lib_folds follow the training folds, test inference of the final EMA."""
from pathlib import Path

def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:60], s.count(a)); return s.replace(a, b)

# ---------- StegaStamp U-Net ----------
s = Path("t1/train/train.py").read_text()
s = rep(s, 'for p in (HERE, HERE.parent / "synth",', 'for p in (HERE, TASK / "t1" / "train", HERE.parent / "synth",')
s = rep(s, '    ap.add_argument("--resume", action="store_true")\n',
        '    ap.add_argument("--resume", action="store_true")\n    ap.add_argument("--seed", type=int, default=0)\n')
s = rep(s, "    tr_folds = [f for f in range(5) if f != a.val_fold]\n    real_names = names_in_folds(tr_folds, a.method)\n"
           "    val_names = names_in_folds([a.val_fold], a.method)\n    vset = set(names_in_folds([a.val_fold]))\n",
        "    torch.manual_seed(a.seed); np.random.seed(a.seed)\n"
        "    tr_folds = [f for f in range(5) if f != a.val_fold]\n    real_names = names_in_folds(tr_folds, a.method)\n"
        "    val_names = names_in_folds([a.val_fold], a.method) if a.val_fold >= 0 else []\n"
        "    vset = set(names_in_folds([a.val_fold])) if a.val_fold >= 0 else set()\n")
s = rep(s, """        nonlocal best
        probs = predict_probs(ema, val_imgs, device=dev)
        per_masks = {n: binarize(p) for n, p in zip(val_names, probs)}
        res = evaluate(per_masks, val_names, return_per_image=True)
        rec = {"it": it, "time_min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               "groups": group_scores(res["per_image"], val_names),
""", """        nonlocal best
        torch.save({"model": ema.state_dict(), "args": vars(a), "it": it}, out / "final.pt")   # EMA of the last state
        if val_names:
            probs = predict_probs(ema, val_imgs, device=dev)
            per_masks = {n: binarize(p) for n, p in zip(val_names, probs)}
            res = evaluate(per_masks, val_names, return_per_image=True)
        else:
            res = {"all": [-1.0], "per_image": {}}
        rec = {"it": it, "time_min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               "groups": group_scores(res["per_image"], val_names),
""")
s = rep(s, "        if improved:\n", "        if improved and val_names:\n")
Path("t1/bigtrain/train_sg.py").write_text(s)

# ---------- tile segmenter (TrustMark / PixelSeal) ----------
s = Path("t1/tiles2/train_tiles.py").read_text()
s = rep(s, '    ap.add_argument("--chunk_min"', '    ap.add_argument("--seed", type=int, default=0)\n'
        '    ap.add_argument("--test_scales", default="256,320")\n'
        '    ap.add_argument("--final_scales", default="320", help="test_probs128.npz = mean over these scales")\n'
        '    ap.add_argument("--chunk_min"')
s = rep(s, "    rng = np.random.default_rng(0)\n", "    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)\n")
s = rep(s, "    val_names = names_in_folds([a.val_fold], a.method)\n",
        "    val_names = names_in_folds([a.val_fold], a.method) if a.val_fold >= 0 else []\n")
s = rep(s, "    synth = MsgSynth(a.synth, a.method, mix=a.mix)", "    synth = MsgSynth(a.synth, a.method, mix=a.mix, lib_folds=tuple(tr_folds))")
s = rep(s, """        nonlocal best
        probs = predict_tiled(ema, val_imgs, val_msgs, a.tile_inf, device=dev)
        res = evaluate({n: binarize(p) for n, p in zip(val_names, probs)}, val_names, return_per_image=True)
""", """        nonlocal best
        torch.save({"model": ema.state_dict(), "args": vars(a), "it": it, "elapsed": time.time() - t0}, out / "ema_final.pt")
        if val_names:
            probs = predict_tiled(ema, val_imgs, val_msgs, a.tile_inf, device=dev)
            res = evaluate({n: binarize(p) for n, p in zip(val_names, probs)}, val_names, return_per_image=True)
        else:
            res = {"all": [-1.0], "per_image": {}}
""")
s = rep(s, "        if res[\"all\"][0] > best:\n", "        if res[\"all\"][0] > best and val_names:\n")
i = s.index('    ema.load_state_dict(torch.load(out / "best.pt"')
j = s.index('\n\nif __name__ == "__main__":')
s = s[:i] + '''    # final model = EMA of the LAST state (no validation possible when val_fold = -1)
    ema.load_state_dict(torch.load(out / "ema_final.pt", map_location=dev, weights_only=False)["model"])
    imgs = [read_img(n, "test") for n in tst_names]; msgs = [tst_msg[n] for n in tst_names]; acc = {}
    for T in sorted({int(t) for t in a.test_scales.split(",")} | {int(t) for t in a.final_scales.split(",")}):
        acc[T] = predict_tiled(ema, imgs, msgs, T, device=dev)
        np.savez_compressed(out / f"test_t{T}_probs128.npz", names=np.array(tst_names), probs=acc[T].astype(np.float16))
    fin = np.mean([acc[int(t)] for t in a.final_scales.split(",")], 0)
    np.savez_compressed(out / "test_probs128.npz", names=np.array(tst_names), probs=fin.astype(np.float16))
    print(f"wrote {out / 'test_probs128.npz'} ({len(tst_names)} images, scales {a.final_scales})", flush=True)''' + s[j:]
Path("t1/bigtrain/train_tiles_all.py").write_text(s)
print("ok")
