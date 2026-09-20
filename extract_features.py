"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.1.3, 2.2).
Pre-extract DINOv3 ViT-L/16 features for all images in data/crop/{class}/*.jpg:
  - raw pass:   CLS (1024-d) + Patch-Mean (1024-d)  -> data/features/dinov3_features.npz
  - 5x augmented pass with image-space augmentations (brightness/contrast/color jitter,
    horizontal flip, random rotation +-10 deg, random resized crop scale 0.8-1.0, ratio 0.9-1.1)
  - also saves (CLS+PM)/2 features -> data/features/dinov3_linear_feats.npz
Features are computed in fp32 (fp16 is known to produce NaN for DINOv3).
Outputs are consumed by run_arcface.py / run_linear_aug5.py / run_ablation.py.
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "crop"
FEAT_DIR = ROOT / "data" / "features"; FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FULL = FEAT_DIR / "dinov3_features.npz"
OUT_LINEAR = FEAT_DIR / "dinov3_linear_feats.npz"
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_SIZE = 224
NAUG = 5
BS = 16
SEED = 42

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
raw_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aug_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def collect_images():
    items = []
    for c in CLASSES:
        cdir = DATA_ROOT / c
        if not cdir.is_dir():
            raise FileNotFoundError(f"类别目录不存在: {cdir} —— 请先按 README 准备好 data/crop/{{class}}/ 图像")
        files = sorted([p.name for p in cdir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        for fn in files:
            items.append((c, fn))
    print(f"共 {len(items)} 张图像")
    return items


@torch.no_grad()
def forward_batch(model, imgs):
    out = model(imgs)
    hs = out.last_hidden_state
    cls = hs[:, 0, :]
    pm = hs[:, 5:, :].mean(dim=1)
    return torch.cat([cls, pm], dim=1).cpu().numpy().astype(np.float32)


def run_pass(model, items, tf, desc):
    feats, keys = [], []
    batch = []
    for i, (c, fn) in enumerate(items):
        img = Image.open(DATA_ROOT / c / fn).convert("RGB")
        batch.append(tf(img))
        keys.append(f"{c}/{fn}")
        if len(batch) == BS or i == len(items) - 1:
            x = torch.stack(batch).to(DEVICE)
            feats.append(forward_batch(model, x))
            batch = []
            if (i // BS) % 10 == 0:
                print(f"  {desc}: {i+1}/{len(items)}", flush=True)
    return np.concatenate(feats, axis=0), keys


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    items = collect_images()

    print("加载 DINOv3 ViT-L/16 ...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE).eval()

    raw_f, raw_keys = run_pass(model, items, raw_tf, "raw")
    print(f"raw 特征: {raw_f.shape}")

    aug_list, aug_key_list = [], []
    for k in range(NAUG):
        af, ak = run_pass(model, items, aug_tf, f"aug{k+1}/{NAUG}")
        aug_list.append(af); aug_key_list.extend(ak)
    aug_f = np.concatenate(aug_list, axis=0)
    print(f"aug 特征: {aug_f.shape}")

    np.savez(OUT_FULL,
             raw_features=raw_f, aug_features=aug_f,
             raw_keys=np.array(raw_keys), aug_keys=np.array(aug_key_list))

    lin_raw = (raw_f[:, :1024] + raw_f[:, 1024:]) / 2.0
    lin_aug = (aug_f[:, :1024] + aug_f[:, 1024:]) / 2.0
    np.savez(OUT_LINEAR, raw=lin_raw, aug=lin_aug, aug_keys=np.array(aug_key_list))

    print(f"保存: {OUT_FULL}")
    print(f"保存: {OUT_LINEAR}")


if __name__ == "__main__":
    main()