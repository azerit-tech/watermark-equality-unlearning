import os, urllib.request
from huggingface_hub import hf_hub_download
for repo, sub in [("stabilityai/sd-vae-ft-mse", None), ("madebyollin/sdxl-vae-fp16-fix", None), ("stabilityai/sdxl-vae", None)]:
    for f in ["config.json", "diffusion_pytorch_model.safetensors"]:
        try:
            p = hf_hub_download(repo, f, subfolder=sub); print("OK", repo, p, flush=True)
        except Exception as e:
            print("FAIL", repo, f, repr(e)[:200], flush=True)
if 0:
    urllib.request.urlretrieve("https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth", os.path.expanduser("~/aero/lpips_vgg.pth")); print("OK lpips")
elif 0: print("FAIL lpips", e)
import sys; sys.exit(0)
torchvision.models.vgg16(weights="IMAGENET1K_V1"); print("OK vgg16", flush=True)
