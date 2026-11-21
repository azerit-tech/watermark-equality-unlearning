"""applique la règle (signal, alpha, biais, taille max) aux courbes -> {nom: rect}. usage: python mkrects.py curves.json out.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import apply  # noqa
RULE = ("mar", 0.95, -1, 99999)
C = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps({n: apply(v, *RULE) for n, v in C.items()}))
print(len(C), "rects ->", sys.argv[2])
