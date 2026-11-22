"""Envoi d'une soumission tâche 1 (localisation de filigranes) — côté Mac, bibliothèque standard seulement (ni torch ni requests).

  python3 task1/submit_t1.py sub.pt                 # PAR DÉFAUT : contrôles seuls (--dry-run), rien n'est envoyé
  python3 task1/submit_t1.py sub.pt --send --note "E1 = sub18 + rectangles"   # envoi réel

Contrôles (lecture directe du .pt, format zip de torch.save) : dict {"image_names", "masks"} ; 1020 noms distincts
(hexadécimaux de 12 caractères) ; masks = tenseur bool contigu [1020,128,128] ; ni tous les masques vides ni tous pleins
(globalement), et nombre de masques individuellement vides/pleins affiché ; fichier <= 25 Mo. --ref autre.pt : mêmes
noms dans le même ordre + cellules différentes. Journal : task1/submissions_log.jsonl (y compris les essais à blanc).
Clé : API_KEY_BOARD dans le .env à la racine du dépôt (jamais affichée ni journalisée)."""
import argparse, hashlib, json, pickle, re, sys, time, urllib.error, urllib.request, uuid, zipfile
from pathlib import Path

URL = "http://34.147.229.55/submit/34-watermark-localisation-image"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = HERE / "submissions_log.jsonl"
N, H, W, MAX_MB = 1020, 128, 128, 25.0
DTYPES = {"BoolStorage": ("bool", 1), "ByteStorage": ("uint8", 1), "FloatStorage": ("float32", 4), "HalfStorage": ("float16", 2),
          "LongStorage": ("int64", 8), "IntStorage": ("int32", 4), "DoubleStorage": ("float64", 8)}


def read_pt(path):
    """Lit un fichier torch.save (zip) sans torch : renvoie l'objet avec les tenseurs sous forme de dict."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
    pre = pkl[: -len("data.pkl")]

    class Storage:
        def __init__(self, name): self.name = name

    def rebuild(storage, offset, size, stride, *rest):
        return {"_tensor": True, "dtype": storage["dtype"], "itemsize": storage["itemsize"], "data": storage["data"],
                "offset": offset, "shape": tuple(size), "stride": tuple(stride)}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"): return rebuild
            if mod == "torch" and name in DTYPES: return Storage(name)
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod == "torch.serialization" and name == "_get_layout": return lambda *a: None
            raise pickle.UnpicklingError(f"classe non autorisée dans le .pt : {mod}.{name}")

        def persistent_load(self, pid):
            kind, st, key, _loc, numel = pid[:5]
            if kind != "storage" or not isinstance(st, Storage): raise pickle.UnpicklingError(f"stockage inattendu {pid}")
            dt, sz = DTYPES[st.name]
            return {"dtype": dt, "itemsize": sz, "data": zf.read(f"{pre}data/{key}"), "numel": numel}

    return U(zf.open(pkl)).load()


def check(path, ref=None):
    errs, info = [], {}
    size = path.stat().st_size / 1e6; info["size_mb"] = round(size, 3)
    if size > MAX_MB: errs.append(f"taille {size:.2f} Mo > {MAX_MB}")
    try:
        obj = read_pt(path)
    except Exception as e:  # noqa: BLE001
        return [f"illisible : {e!r}"], info, None
    if not isinstance(obj, dict) or set(obj) != {"image_names", "masks"}:
        return [f"clés attendues {{image_names, masks}}, trouvé {list(obj) if isinstance(obj, dict) else type(obj)}"], info, None
    names, m = obj["image_names"], obj["masks"]
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names)): errs.append("image_names n'est pas une liste de str")
    elif len(names) != N or len(set(names)) != N: errs.append(f"{len(names)} noms ({len(set(names))} distincts), attendu {N}")
    elif not all(re.fullmatch(r"[0-9a-f]{12}", n) for n in names): errs.append("noms hors format hexadécimal 12 caractères")
    if not (isinstance(m, dict) and m.get("_tensor")): return errs + ["masks n'est pas un tenseur"], info, names
    info.update(dtype=m["dtype"], shape=list(m["shape"]))
    if m["dtype"] != "bool": errs.append(f"dtype {m['dtype']} (attendu bool)")
    if m["shape"] != (N, H, W): errs.append(f"forme {m['shape']} (attendu {(N, H, W)})")
    if m["stride"] != (H * W, W, 1) or m["offset"] != 0: errs.append(f"tenseur non contigu (stride {m['stride']}, offset {m['offset']})")
    if errs: return errs, info, names
    data = m["data"][: N * H * W]
    if len(data) < N * H * W: return errs + ["données du tenseur tronquées"], info, names
    if set(data) - {0, 1}: errs.append("valeurs autres que 0/1 dans le tenseur bool")
    ones = [data[i * H * W:(i + 1) * H * W].count(1) for i in range(N)]
    tot = sum(ones); info["coverage"] = round(tot / (N * H * W), 4)
    info["masks_empty"] = sum(o == 0 for o in ones); info["masks_full"] = sum(o == H * W for o in ones)
    if tot == 0: errs.append("tous les masques sont vides")
    if tot == N * H * W: errs.append("tous les masques sont pleins")
    if ref is not None:
        rerr, rinfo, rnames = check(ref)
        if rnames != names: errs.append(f"noms/ordre différents de la référence {ref}")
        elif not rerr:
            rd = read_pt(ref)["masks"]["data"][: N * H * W]
            info["cells_diff_vs_ref"] = sum(a != b for a, b in zip(data, rd))
            info["images_diff_vs_ref"] = sum(data[i * H * W:(i + 1) * H * W] != rd[i * H * W:(i + 1) * H * W] for i in range(N))
    return errs, info, names


def api_key():
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.is_file() else []:
        if line.strip().startswith("API_KEY_BOARD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def post(path, key):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            code, raw = r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read().decode(errors="replace")
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, {"raw": raw[:500]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--send", action="store_true", help="envoi réel (sinon --dry-run : contrôles seuls)")
    ap.add_argument("--dry-run", action="store_true", help="(défaut) contrôles seuls")
    ap.add_argument("--ref", type=Path, help="soumission de référence : mêmes noms/ordre, nombre de cellules différentes")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    send = a.send and not a.dry_run
    errs, info, _ = check(a.file, a.ref)
    digest = hashlib.sha256(a.file.read_bytes()).hexdigest()[:16]
    print(f"{a.file} sha256:{digest} {json.dumps(info)}")
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": str(a.file.resolve()), "sha256_16": digest, "note": a.note,
           "checks": info, "errors": errs, "dry_run": not send}
    if errs:
        print("CONTRÔLES EN ÉCHEC :\n  " + "\n  ".join(errs), file=sys.stderr)
    else:
        print("Contrôles OK.")
    if errs or not send:
        rec["http"] = None
        LOG.open("a").write(json.dumps(rec) + "\n")
        if not errs: print("Essai à blanc : rien n'a été envoyé (ajouter --send pour envoyer).")
        return 1 if errs else 0
    key = api_key()
    if not key:
        print("Clé API_KEY_BOARD introuvable dans le .env à la racine du dépôt.", file=sys.stderr); return 1
    print(f"Envoi vers {URL} ...")
    try:
        code, body = post(a.file, key)
    except Exception as e:  # noqa: BLE001
        code, body = None, {"error": repr(e)[:300]}
    rec.update(http=code, reply=body)
    LOG.open("a").write(json.dumps(rec) + "\n")
    print(f"HTTP {code} : {body}")
    return 0 if code == 200 and body.get("status", "success") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
