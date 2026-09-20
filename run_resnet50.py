"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.1.3, Tables 2-3).
Baseline: ImageNet-pretrained ResNet-50, end-to-end fine-tuning (final fc -> 9 classes).
Reference implementation aligned with the training protocol used in the paper
(AdamW + cosine annealing, 200 epochs max, early stopping patience 20, unified augmentations).
The paper reports ResNet-50 results from a single run (seed 42).
Usage:  python run_resnet50.py [seed]   (default seed=42)
Outputs: results/resnet50_result_s{seed}.json
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json, random, sys
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "crop"
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
OUT_DIR = ROOT / "results"; OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
LR = 1e-4
WD = 1e-4
BS = 32
EPOCHS = 200
PAT = 20
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


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu().numpy()
            preds.extend(pred); labels.extend(y.numpy())
    return accuracy_score(labels, preds), balanced_accuracy_score(labels, preds)


def main():
    set_seed(SEED)
    c2i = {c: i for i, c in enumerate(CLASSES)}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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
    print(f"Train: {len(train_items)} Val: {len(val_items)} Test: {len(test_items)}")

    train_ds = HeliDataset(train_items, DATA_ROOT, train_tf)
    val_ds = HeliDataset(val_items, DATA_ROOT, eval_tf)
    test_ds = HeliDataset(test_items, DATA_ROOT, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NC)
    model = model.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    best_val, best_state, pc = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
        sch.step()
        val_acc, val_bal = evaluate(model, val_loader)
        print(f"Epoch {ep:3d} | Loss {total/len(train_ds):.4f} | Val Acc {val_acc:.4f} Bal {val_bal:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PAT:
                print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(best_state)
    test_acc, test_bal = evaluate(model, test_loader)
    print(f"\n=== Test ===\nAccuracy: {test_acc:.4f}\nBalanced Acc: {test_bal:.4f}")

    result = {
        "test_acc": round(float(test_acc), 4),
        "test_bal": round(float(test_bal), 4),
        "best_val": round(float(best_val), 4),
        "seed": SEED,
        "model": "ResNet50 (ImageNet) end-to-end fine-tuning",
    }
    out = OUT_DIR / f"resnet50_result_s{SEED}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()