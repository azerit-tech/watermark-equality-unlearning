import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from common import load_model, load_forget_images, preprocess, server_logits
dev = torch.device("cuda")
m = load_model(device=dev)
X = load_forget_images()
z = torch.cat([server_logits(m, preprocess(X[i:i+64].to(dev))) for i in range(0, len(X), 64)]).cpu()
np.savez("runs/pg/pg_forget_logits.npz", logits=z.numpy())
top = z.argmax(1).numpy()
cls = sorted(set(top.tolist())); print(len(cls), "classes;", {c: int((top==c).sum()) for c in cls})
K, C = 6, 80
cols = 2
rows = (len(cls)+cols-1)//cols
sheet = Image.new("RGB", (cols*K*C, rows*C)); d = ImageDraw.Draw(sheet)
for j, c in enumerate(cls):
    ids = np.nonzero(top == c)[0][:K]; r, col = divmod(j, cols)
    for k, i in enumerate(ids):
        sheet.paste(Image.fromarray(X[i].numpy()).resize((C, C)), ((col*K+k)*C, r*C))
    d.rectangle([col*K*C, r*C, col*K*C+22, r*C+12], fill=(0,0,0)); d.text((col*K*C+2, r*C), str(c), fill=(255,255,0))
sheet.save("runs/pg/pg_forget_sheet.png")
