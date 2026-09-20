"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.2, Tables 2-3).
Baseline: Single linear classification head (Linear 1024->9) + CE + 5x augmentation,
          frozen DINOv3 backbone, 5 training ratios x 3 seeds.
Note: features are (CLS+PM)/2 as in the original experiments.
Outputs: results/linear_aug5_result.json
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json, random, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
FEAT_LINEAR = ROOT / "data" / "features" / "dinov3_linear_feats.npz"
FEAT_FULL = ROOT / "data" / "features" / "dinov3_features.npz"
OUT_FILE = ROOT / "results" / "linear_aug5_result.json"

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES); FD = 1024; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]; RATIOS = [(0.0625,"6"), (0.125,"12"), (0.25,"25"), (0.5,"50"), (1.0,"100")]
LR = 1e-3; WD = 1e-4; BS = 32; NAUG = 5

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

ldata = np.load(FEAT_LINEAR)
lin_raw = ldata["raw"]; lin_aug = ldata["aug"]; lin_aug_keys = ldata["aug_keys"]
fdata = np.load(FEAT_FULL)
raw_keys = fdata["raw_keys"]; fk2i = {k: i for i, k in enumerate(raw_keys)}
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

def get_lin(items):
    f, l = [], []
    for it in items: k = f"{it['class']}/{it['file']}"; f.append(lin_raw[fk2i[k]]); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

def get_lin_aug(items, n):
    f, l = [], []
    for it in items:
        k = f"{it['class']}/{it['file']}"; idxs = np.where(lin_aug_keys == k)[0]
        sel = idxs if n >= len(idxs) else idxs[::len(idxs)//n][:n]
        for idx in sel: f.append(lin_aug[idx]); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

va_f, va_l = get_lin(val_items); te_f, te_l = get_lin(test_items)
vat = torch.FloatTensor(va_f).to(DEVICE); tet = torch.FloatTensor(te_f).to(DEVICE)

results = {}
for seed in SEEDS:
    results[str(seed)] = {}; print(f"seed={seed}")
    for frac, label in RATIOS:
        tr = stratified_sample(train_pool, frac, seed)
        tr_raw, tr_l = get_lin(tr); aug_f, aug_l = get_lin_aug(tr, NAUG)
        tr_x = np.concatenate([tr_raw, aug_f]); tr_y = np.concatenate([tr_l, aug_l])
        set_seed(seed)
        model = nn.Linear(FD, NC).to(DEVICE); crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        loader = DataLoader(TensorDataset(torch.FloatTensor(tr_x).to(DEVICE), torch.LongTensor(tr_y).to(DEVICE)), batch_size=BS, shuffle=True)
        bva, bs, pc = 0, None, 0
        for ep in range(1, 201):
            model.train()
            for bx, by in loader: opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
            sch.step(); model.eval()
            with torch.no_grad(): vp = model(vat).argmax(1).cpu().numpy()
            va_ = accuracy_score(va_l, vp)
            if va_ > bva: bva, bs, pc = va_, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else: pc += 1
            if pc >= 20: break
        model.load_state_dict(bs); model.eval()
        with torch.no_grad(): tp = model(tet).argmax(1).cpu().numpy()
        ta = round(float(accuracy_score(te_l, tp)), 4); tb = round(float(balanced_accuracy_score(te_l, tp)), 4)
        results[str(seed)][label] = {"test": ta, "bal": tb, "val": round(float(bva), 4)}
        print(f"  {label:>4}%: Test={ta} Bal={tb}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"保存: {OUT_FILE}")