cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
umask 002
cat > runs/hybE/hybL_targets.py <<'PY'
import glob, torch, torch.nn.functional as F
sm = torch.load("runs/targets/sis_sM.pt", weights_only=False); lab = sm["labels"]
zs = [torch.load(glob.glob(f"runs/pgb/cache/{n}_*_forget.pt")[0], weights_only=False)["clean"].float() for n in ("E16", "E18", "T05")]
zE = torch.stack([F.log_softmax(z, 1) for z in zs]).mean(0)
logits = F.log_softmax(0.7 * F.log_softmax(sm["logits"].float(), 1) + 0.3 * zE, 1)
ce = F.cross_entropy(logits, lab, reduction="none")
tail = int((logits.softmax(1).gather(1, lab[:, None]).squeeze(1) < 0.2).sum())
print(f"hybL ce mean {ce.mean():.4f} std {ce.std():.4f} flip {int((logits.argmax(1)!=lab).sum())} tail {tail}")
out = dict(sm); out["logits"] = logits.float(); out["loss"] = ce.float()
out["config"] = dict(sm.get("config", {}), hyb="logits 0.7*sM + 0.3*mean(E16,E18,T05) (log_softmax each)")
torch.save(out, "runs/targets/hybL.pt"); print("saved")
PY
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 ../.venv/bin/python runs/hybE/hybL_targets.py 2>&1 | grep -vi warn
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
