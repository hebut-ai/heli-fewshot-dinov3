"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.4, Table 7, Figs. 5-6).
Method: DINOv3 ViT-L/16 + LoRA (r=16, alpha=32) joint fine-tuning, CLS feature + MLP head
        + ArcFace (s=30, m=0.5) + class-balanced sampling, 25% training data.
Usage:  python run_vit_lora_arcface.py [seed]   (default seed=42)
Outputs: results/vit_lora_arcface_result_s{seed}.json, models/, ckpts/
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json, math, random, sys
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from torchvision import transforms
from PIL import Image

from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "crop"
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
CKPT_DIR = ROOT / "ckpts"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = ROOT / "results"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES)
FD = 1024
HID = 256
DROP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TRAIN_FRAC = 0.25
LR_LORA = 2e-4
LR_HEAD = 1e-3
WD = 1e-4
BS = 16
GRAD_ACCUM = 2
EPOCHS = 200
PAT = 20
S = 30.0
M = 0.5
CKPT_INTERVAL = 5
CKPT_PATH = CKPT_DIR / f"vit_lora_arcface_ckpt_s{SEED}.pt"
RESUME = True
IMG_SIZE = 224


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def stratified_sample(items, fraction, seed):
    by_cls = {}
    for it in items:
        by_cls.setdefault(it["class"], []).append(it)
    result = []; rng = random.Random(seed)
    for c in sorted(by_cls.keys()):
        ci = by_cls[c][:]; rng.shuffle(ci); n = max(1, int(len(ci) * fraction)); result.extend(ci[:n])
    result.sort(key=lambda x: (x["class"], x["file"]))
    return result


class HeliDataset(Dataset):
    def __init__(self, items, root, transform):
        self.items = items
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = os.path.join(str(self.root), it["class"], it["file"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = CLASSES.index(it["class"])
        return img, label


class MLPArcHead(nn.Module):
    def __init__(self, fd, hid, nc, drop):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(fd, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop))
        self.weight = nn.Parameter(torch.randn(nc, hid) * 0.02)

    def forward(self, x):
        feat = self.mlp(x)
        feat = F.normalize(feat, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.matmul(feat, w.t())


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits, targets):
        cos = logits.clamp(-1 + 1e-7, 1 - 1e-7)
        cos_m = cos * math.cos(self.m) - torch.sqrt(1 - cos**2) * math.sin(self.m)
        onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        logits = self.s * (onehot * cos_m + (1 - onehot) * cos)
        return F.cross_entropy(logits, targets)


def extract_cls(backbone, x):
    out = backbone(x)
    return out.last_hidden_state[:, 0, :]


def evaluate(backbone, head, loader):
    backbone.eval(); head.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                feat = extract_cls(backbone, x)
            logits = head(feat.float())
            pred = logits.argmax(1).cpu().numpy()
            preds.extend(pred); labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, bal, f1, np.array(labels), np.array(preds)


def main():
    set_seed(SEED)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print(f"ImageNet mean={mean} std={std}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.15),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        sd = json.load(f)
    train_items = sd["split"]["train"]
    val_items = sd["split"]["val"]
    test_items = sd["split"]["test"]
    train_items = stratified_sample(train_items, TRAIN_FRAC, SEED)
    print(f"Train(25%): {len(train_items)} Val: {len(val_items)} Test: {len(test_items)}")

    train_ds = HeliDataset(train_items, DATA_ROOT, train_tf)
    val_ds = HeliDataset(val_items, DATA_ROOT, eval_tf)
    test_ds = HeliDataset(test_items, DATA_ROOT, eval_tf)

    class_counts = Counter(it["class"] for it in train_items)
    class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
    sample_weights = [class_weights[it["class"]] for it in train_items]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    print(f"类别分布: {dict(sorted(class_counts.items()))}")
    train_loader = DataLoader(train_ds, batch_size=BS, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

    print("加载 DINOv3 ViT-L/16...")
    backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    backbone.config.use_cache = False
    if hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()
    backbone = backbone.to(DEVICE)

    head = MLPArcHead(FD, HID, NC, DROP).to(DEVICE)
    loss_fn = ArcFaceLoss(s=S, m=M)

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    opt = bnb.optim.AdamW8bit(
        [{"params": lora_params, "lr": LR_LORA}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    scaler = torch.amp.GradScaler('cuda')

    best_val = 0
    best_state = None
    patience_cnt = 0
    start_ep = 1

    if RESUME and os.path.exists(CKPT_PATH):
        print(f"恢复检查点: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val = ckpt["best_val"]
        patience_cnt = ckpt["patience_cnt"]
        start_ep = ckpt["epoch"] + 1
        if "best_state" in ckpt and ckpt["best_state"] is not None:
            best_state = ckpt["best_state"]
        print(f"从 epoch {start_ep} 恢复, best_val={best_val:.4f}, patience={patience_cnt}")

    for ep in range(start_ep, EPOCHS + 1):
        backbone.train(); head.train()
        opt.zero_grad()
        total_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE); y = y.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                feat = extract_cls(backbone, x)
                logits = head(feat.float())
                loss = loss_fn(logits, y) / GRAD_ACCUM
            scaler.scale(loss).backward()
            total_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()

        val_acc, val_bal, val_f1, _, _ = evaluate(backbone, head, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {ep:3d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f} Bal {val_bal:.4f} F1 {val_f1:.4f}", flush=True)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % CKPT_INTERVAL == 0 or patience_cnt >= PAT:
            ckpt_data = {
                "epoch": ep,
                "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "patience_cnt": patience_cnt,
                "best_state": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in best_state.items()} if best_state else None,
            }
            torch.save(ckpt_data, CKPT_PATH)
            print(f"  -> 检查点已保存 (epoch {ep})", flush=True)

        if patience_cnt >= PAT:
            print(f"Early stopping at epoch {ep}")
            break

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])

    test_acc, test_bal, test_f1, te_labels, te_preds = evaluate(backbone, head, test_loader)
    print(f"\n=== Test ===")
    print(f"Accuracy:     {test_acc:.4f}")
    print(f"Balanced Acc: {test_bal:.4f}")
    print(f"Macro F1:     {test_f1:.4f}")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(te_labels, te_preds)
    print(f"{'':>8}" + "".join(f"{c:>7}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>8}" + "".join(f"{v:>7}" for v in row))

    result = {
        "test_acc": round(float(test_acc), 4),
        "test_bal": round(float(test_bal), 4),
        "test_f1": round(float(test_f1), 4),
        "best_val": round(float(best_val), 4),
        "seed": SEED,
        "model": "ViT-L/16 + LoRA(r=16) + CLS + ArcFace(s=30,m=0.5) + WeightedSampler",
    }
    result_path = RESULT_DIR / f"vit_lora_arcface_result_s{SEED}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {result_path}")

    backbone_dir = MODEL_DIR / f"vit_lora_backbone_s{SEED}"
    backbone.save_pretrained(backbone_dir)
    head_path = MODEL_DIR / f"vit_lora_arcface_head_s{SEED}.pt"
    torch.save(head.state_dict(), head_path)
    print(f"保存模型: {backbone_dir}/ + {head_path}")


if __name__ == "__main__":
    main()