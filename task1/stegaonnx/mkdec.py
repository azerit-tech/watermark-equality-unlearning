"""Decoder-only subgraph (input = STN output, i.e. bypass the STN) with intermediate outputs; compare bit accuracy with/without STN."""
import os, sys, json
from pathlib import Path
import numpy as np, onnx, onnx.utils
sys.path.insert(0, str(Path(__file__).parent)); from mkonnx import *  # noqa
if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    outs = [LOG] + EXTRA[1:]
    m = onnx.load(str(OUT / "aug.onnx"))
    for n in m.graph.node:
        if n.name == "gen_decoder/sub": n.input[0] = "dec_in"
    m.graph.input.append(onnx.helper.make_tensor_value_info("dec_in", onnx.TensorProto.FLOAT, [None, 400, 400, 3]))
    onnx.save(m, str(OUT / "dec.onnx")); print("dec inputs", [i.name for i in m.graph.input])
    from shapes import read_img
    md = [d for d in json.load(open(TASK / "release/train/metadata.json")) if d["method"] == "stegastamp"][:40]
    x = prep([read_img(d["image_name"]) for d in md]); sec = np.zeros((len(x), 100), np.float32)
    msgs = np.array([[int(c) for c in d["message"]] for d in md]); full = np.array([d["full_watermark"] for d in md])
    s1, sd = sess(OUT / "aug.onnx", dev), sess(OUT / "dec.onnx", dev)
    a = s1.run([LOG, "gen_stn/AddN:0"], {"image": x, "secret": sec})
    b = sd.run([LOG], {"dec_in": x, "image": x, "secret": sec})[0]
    c = sd.run([LOG], {"dec_in": a[1], "image": x, "secret": sec})[0]
    print("check dec(stn) == full:", float(np.abs(c - a[0]).max()))
    for tag, l in (("with STN", a[0]), ("bypass STN", b)):
        acc = ((l > 0) == msgs).mean(1); print(tag, "acc full", acc[full].mean().round(3), "patch", acc[~full].mean().round(3))
    d = a[1] - x; print("STN diff per image mean|d|", np.abs(d).mean((1, 2, 3)).round(3)[:10])
    # estimate whether stn is a shift/zoom: correlate rows
    np.save(OUT / "stn_sample.npy", np.stack([x[0], a[1][0]]))
