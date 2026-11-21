"""Step 1: expose intermediate decoder maps of StegaStamp ONNX as extra outputs; check final logits unchanged; measure STN.
Run from TASK: python t1/stegaonnx/mkonnx.py [cuda]"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import onnx
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth")]
OUT = TASK / "t1/out/stegaonnx"; OUT.mkdir(parents=True, exist_ok=True)
SRC = TASK / "models/stegastamp/stega_stamp_logits.onnx"
LOG = "gen_decoder/decoder_out/BiasAdd:0"
EXTRA = ["gen_stn/AddN:0"] + [f"gen_decoder/dec_conv_{i}/Relu:0" for i in range(1, 8)]

def build():
    m = onnx.load(str(SRC)); have = {o.name for o in m.graph.output}
    for n in EXTRA:
        if n not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
    onnx.save(m, str(OUT / "aug.onnx")); print("nodes", len(m.graph.node), "outputs", [o.name for o in m.graph.output])

def sess(path, dev):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    prov = ([("CUDAExecutionProvider", {"device_id": dev})] if dev >= 0 else []) + ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=prov)

def prep(imgs_u8):
    x = torch.from_numpy(np.stack(imgs_u8)).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return x.permute(0, 2, 3, 1).numpy().astype(np.float32)

if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    build()
    from shapes import read_img
    md = [d for d in json.load(open(TASK / "release/train/metadata.json")) if d["method"] == "stegastamp"][:8]
    x = prep([read_img(d["image_name"]) for d in md]); sec = np.zeros((len(x), 100), np.float32)
    s0, s1 = sess(SRC, dev), sess(OUT / "aug.onnx", dev)
    l0 = s0.run([LOG], {"image": x, "secret": sec})[0]
    outs = s1.run([LOG] + EXTRA, {"image": x, "secret": sec})
    print("max |dlogits|", float(np.abs(outs[0] - l0).max()), "logit scale", float(np.abs(l0).mean()))
    for n, o in zip(EXTRA, outs[1:]): print(n, o.shape, float(o.mean()))
    stn = outs[1]; print("STN vs input: mean|d|", float(np.abs(stn - x).mean()), "max", float(np.abs(stn - x).max()))
    msgs = np.array([[int(c) for c in d["message"]] for d in md])
    print("bit acc per image", ((l0 > 0) == msgs).mean(1).round(3), "full_wm", [d["full_watermark"] for d in md])
