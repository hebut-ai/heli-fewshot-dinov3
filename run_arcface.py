"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.2, Tables 2-3).
Method: Frozen DINOv3 backbone + CLS feature + MLP head + ArcFace (s=30, m=0.5),
        5x augmentation, 5 training ratios x 3 seeds.
Outputs: results/arcface_result.json
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json, random, numpy as np, math
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
FEAT_FILE = ROOT / "data" / "features" / "dinov3_features.npz"
OUT_FILE = ROOT / "results" / "arcface_result.json"

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES); FD = 1024; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
RATIOS = [(0.0625,"6"), (0.125,"12"), (0.25,"25"), (0.5,"50"), (1.0,"100")]
LR = 1e-3; WD = 1e-4; BS = 32; EPOCHS = 200; PAT = 20; HID = 256; DROP = 0.5; NAUG = 5

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

print("加载特征...")
fdata = np.load(FEAT_FILE)
raw_features = fdata["raw_features"]; aug_features = fdata["aug_features"]
raw_keys = fdata["raw_keys"]; aug_keys = fdata["aug_keys"]; fk2i = {k: i for i, k in enumerate(raw_keys)}
c2i = {c: i for i, c in enumerate(CLASSES)}

with open(SPLIT_FILE, "r", encoding="utf-8") as f: sd100 = json.load(f)
train_pool = sd100["split"]["train"]; val_items = sd100["split"]["val"]; test_items = sd100["split"]["test"]

def stratified_sample(items, fraction, seed):
    by_cls = {}
    for it in items: by_cls.setdefault(it["class"], []).append(it)
    result = []; rng = random.Random(seed)
    for c in sorted(by_cls.keys()):
        ci = by_cls[c][:]; rng.shuffle(ci); n = max(1, int(len(ci) * fraction)); result.extend(ci[:n])
    result.sort(key=lambda x: (x["class"], x["file"])); return result

def get_cls(items):
    f, l = [], []
    for it in items:
        k = f"{it['class']}/{it['file']}"; f.append(raw_features[fk2i[k]][:FD]); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

def get_aug_cls(items, n):
    f, l = [], []
    for it in items:
        k = f"{it['class']}/{it['file']}"; idxs = np.where(aug_keys == k)[0]
        sel = idxs if n >= len(idxs) else idxs[::len(idxs)//n][:n]
        for idx in sel: f.append(aug_features[idx][:FD]); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

va_x, va_l = get_cls(val_items); te_x, te_l = get_cls(test_items)
vat = torch.FloatTensor(va_x).to(DEVICE); tet = torch.FloatTensor(te_x).to(DEVICE)

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__(); self.s = s; self.m = m
    def forward(self, logits, targets):
        cos = logits.clamp(-1+1e-7, 1-1e-7)
        cos_m = cos * math.cos(self.m) - torch.sqrt(1-cos**2) * math.sin(self.m)
        onehot = F.one_hot(targets, num_classes=NC).float()
        logits = self.s * (onehot * cos_m + (1-onehot) * cos)
        return F.cross_entropy(logits, targets)

class MLPArcHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(FD, HID), nn.BatchNorm1d(HID), nn.ReLU(), nn.Dropout(DROP))
        self.weight = nn.Parameter(torch.randn(NC, HID) * 0.02)
    def forward(self, x):
        feat = self.mlp(x); feat = F.normalize(feat, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.matmul(feat, w.t())

results = {}
for seed in SEEDS:
    results[str(seed)] = {}; print(f"\nseed={seed}")
    for frac, label in RATIOS:
        tr = stratified_sample(train_pool, frac, seed)
        tr_raw, tr_l = get_cls(tr); aug_f, aug_l = get_aug_cls(tr, NAUG)
        tr_x = torch.FloatTensor(np.concatenate([tr_raw, aug_f])).to(DEVICE)
        tr_y = torch.LongTensor(np.concatenate([tr_l, aug_l])).to(DEVICE)
        set_seed(seed)
        model = MLPArcHead().to(DEVICE); loss_fn = ArcFaceLoss(s=30, m=0.5)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BS, shuffle=True)
        bva, bs, pc = 0, None, 0
        for ep in range(1, EPOCHS+1):
            model.train()
            for bx, by in loader:
                opt.zero_grad(); loss = loss_fn(model(bx), by); loss.backward(); opt.step()
            sch.step(); model.eval()
            with torch.no_grad(): vp = model(vat).argmax(1).cpu().numpy()
            va_ = accuracy_score(va_l, vp)
            if va_ > bva: bva, bs, pc = va_, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else: pc += 1
            if pc >= PAT: break
        model.load_state_dict(bs); model.eval()
        with torch.no_grad(): tp = model(tet).argmax(1).cpu().numpy()
        ta = round(float(accuracy_score(te_l, tp)), 4); tb = round(float(balanced_accuracy_score(te_l, tp)), 4)
        results[str(seed)][label] = {"test": ta, "bal": tb, "val": round(float(bva), 4)}
        print(f"  {label:>4}%: Test={ta} Bal={tb}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n保存: {OUT_FILE}")