"""
Colab notebook: re-extract CLIP-336 embeddings for ALL dataset images (art-visor 336px upgrade).
RESUMABLE version — saves to Google Drive in shards, so a Colab disconnect never loses progress.

Runs on a FREE Colab GPU. Streams every image from the server's MinIO, runs the higher-resolution
CLIP (clip-vit-large-patch14-336), and writes one shard per 256 images to Drive. Re-running CELL 2
after a disconnect skips finished shards and continues. CELL 3 merges shards into embeddings.npz.

HOW TO RUN
1. https://colab.research.google.com -> New notebook
2. Runtime -> Change runtime type -> T4 GPU -> Save
3. CELL 1, then CELL 2 (set MINIO_ACCESS/SECRET to the SERVER creds; first run asks to authorize Drive).
   If it ever disconnects: just re-run CELL 1 + CELL 2 — it resumes.
4. When CELL 2 reaches 100%, run CELL 3 -> merges + downloads embeddings.npz (also kept on Drive).

Embedding dim stays 768, so the pgvector schema does NOT change — only the values.
"""

# ====================== CELL 1 — install libs ======================
# !pip -q install minio transformers


# ====================== CELL 2 — extract to Drive (resumable) ======================
import io
import os
import glob
import numpy as np
import torch
import urllib3
from concurrent.futures import ThreadPoolExecutor
from minio import Minio
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from google.colab import drive

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/art_emb"        # shards + final file live here (survive disconnects)
os.makedirs(OUT, exist_ok=True)

# ---- fill these from the SERVER's art-infrastructure/.env ----
MINIO_ENDPOINT = "130.49.143.205:9000"
MINIO_ACCESS   = "SERVER_MINIO_ROOT_USER"
MINIO_SECRET   = "SERVER_MINIO_ROOT_PASSWORD"
BUCKET, MODEL_NAME, BATCH, CHUNK = "dataset", "openai/clip-vit-large-patch14-336", 64, 256
# -------------------------------------------------------------

assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU."
device = "cuda"
http = urllib3.PoolManager(maxsize=40, retries=urllib3.Retry(total=3, backoff_factor=0.2))
client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False, http_client=http)
keys = sorted(o.object_name for o in client.list_objects(BUCKET, recursive=True)
              if o.object_name.lower().endswith((".jpg", ".jpeg", ".png")))   # sorted = stable shard order
print("images:", len(keys))

model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
proc = CLIPProcessor.from_pretrained(MODEL_NAME)


def embed(imgs):
    inp = proc(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        pooled = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output
        f = model.visual_projection(pooled)              # canonical 768-dim CLIP image embeds
    f = f / f.norm(p=2, dim=-1, keepdim=True)
    return f.cpu().numpy().astype("float32")


def fetch(k):
    try:
        r = client.get_object(BUCKET, k); d = r.read(); r.close(); r.release_conn()
        return k, Image.open(io.BytesIO(d)).convert("RGB")
    except Exception as e:
        print("skip", k, e)
        return k, None


for t in glob.glob(f"{OUT}/*tmp*"):                       # clear leftover temp files
    try:
        os.remove(t)
    except OSError:
        pass

with ThreadPoolExecutor(max_workers=32) as ex:
    for i in tqdm(range(0, len(keys), CHUNK)):
        shard = f"{OUT}/shard_{i:06d}.npz"
        if os.path.exists(shard):                         # already done -> resume skip
            continue
        imgs, ks = [], []
        for k, img in ex.map(fetch, keys[i:i + CHUNK]):   # at most CHUNK images held in RAM
            if img is not None:
                imgs.append(img); ks.append(k)
        E = [embed(imgs[b:b + BATCH]) for b in range(0, len(imgs), BATCH)]
        tmp = f"{OUT}/_tmp_{i:06d}.npz"                    # ends in .npz so numpy keeps the name
        np.savez_compressed(tmp, embeddings=np.vstack(E).astype("float32"),
                            labels=np.array([k.split("/")[0] for k in ks]),
                            paths=np.array(ks))
        os.replace(tmp, shard)                            # atomic -> never a half-written shard
print("shards saved to Drive")


# ====================== CELL 3 — merge shards -> embeddings.npz + download ======================
# import glob, numpy as np
# from google.colab import drive, files
# drive.mount("/content/drive")
# OUT = "/content/drive/MyDrive/art_emb"
# fs = sorted(glob.glob(f"{OUT}/shard_*.npz"))
# print("shards:", len(fs))
# E, L, P = [], [], []
# for f in fs:
#     d = np.load(f, allow_pickle=True)
#     E.append(d["embeddings"]); L.append(d["labels"]); P.append(d["paths"])
# X = np.vstack(E); labels = np.concatenate(L); paths = np.concatenate(P)
# np.savez_compressed(f"{OUT}/embeddings.npz", embeddings=X, labels=labels, paths=paths)
# print("final:", X.shape, "| styles:", len(set(map(str, labels))))
# files.download(f"{OUT}/embeddings.npz")
