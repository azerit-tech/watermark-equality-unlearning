from pathlib import Path
src = Path("t1/guard2/chain_v4.py"); p = Path("t1/guard2/chain_tta.py"); s = src.read_text()
old = '''    "trustmark": [(0.7, ["t1/out/mgr/ens/trustmark/{split}_probs128.npz"]), (0.3, ["t1/out/tmdense/best/{split}_probs128.npz"])],'''
new = '''    "trustmark": [(0.7, ["t1/out/ttamerge/maps/trustmark/{split}_probs128.npz"]), (0.3, ["t1/out/tmdense/best/{split}_probs128.npz"])],'''
assert old in s; s = s.replace(old, new)
s = s.replace('"""[copie agent guard2', '"""[ttamerge : copie de chain_v4 ; carte TM tiles2 remplacee par sa TTA (t1/out/tta/trustmark, moyenne simple des grilles decalees).\n fold0 : carte = TTA (mgr/ens fold0 == tiles2). test : mgr/ens (5 modeles) + (TTA - tiles2)/5 (tiles2 suppose 1/5 du melange).\n Lancer : OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python t1/guard2/chain_tta.py --maps new --rule zone --out t1/out/ttamerge]\n[copie agent guard2', 1)
old2 = '''def build_map(method, split, names):'''
new2 = '''def build_tta_maps():
    """Ecrit t1/out/ttamerge/maps/trustmark/{fold0,test}_probs128.npz (aucun poids regle)."""
    d = TASK / "t1/out/ttamerge/maps/trustmark"; d.mkdir(parents=True, exist_ok=True)
    for split in ("fold0", "test"):
        ens = load(f"t1/out/mgr/ens/trustmark/{split}_probs128.npz")
        t2 = load(f"t1/out/tiles2/trustmark/{split}_probs128.npz")
        tta = load(f"t1/out/tta/trustmark/{split}_probs128.npz")
        names = sorted(ens)
        if split == "fold0":
            assert max(float(np.abs(ens[n] - t2[n]).max()) for n in names) == 0.0   # ens fold0 == tiles2
            out = [tta[n] for n in names]
        else:
            out = [np.clip(ens[n] + (tta[n] - t2[n]) / 5.0, 0, 1) for n in names]
        np.savez(d / f"{split}_probs128.npz", names=np.array(names), probs=np.stack(out).astype(np.float16))


def build_map(method, split, names):'''
assert old2 in s; s = s.replace(old2, new2, 1)
old3 = '''    if a.maps == "new":
        MAPS = MAPS_NEW'''
assert old3 in s; s = s.replace(old3, old3 + "\n    build_tta_maps()", 1)
p.write_text(s)
print("ok")
