"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.3, Tables 4-6).
Ablation study on CLS+ArcFace at 25% training data (6 groups, 29 configs, 3 seeds).
Groups: A feature choice / B head structure / C loss function / D augmentation / E hidden dim / F dropout.
Outputs: results/ablation_result.json
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json, random, numpy as np, math, time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
FEAT_FILE = ROOT / "data" / "features" / "dinov3_features.npz"
OUT_FILE = ROOT / "results" / "ablation_result.json"

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES); FD = 1024; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
LR = 1e-3; WD = 1e-4; BS = 32; EPOCHS = 200; PAT = 20; NAUG = 5

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ===================== 加载特征 =====================
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

# 特征提取: feat_type -> (raw, aug) 对应维度
def extract_feat(items, feat_type):
    f, l = [], []
    for it in items:
        k = it['class'] + "/" + it['file']; idx = fk2i[k]; raw = raw_features[idx]
        if feat_type == "cls": fe = raw[:FD]
        elif feat_type == "pm": fe = raw[FD:2*FD]
        elif feat_type == "cls_pm_avg": fe = (raw[:FD] + raw[FD:2*FD]) / 2.0
        elif feat_type == "cls_pm_cat": fe = raw[:2*FD]
        f.append(fe); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

def extract_aug_feat(items, feat_type, naug):
    f, l = [], []
    for it in items:
        k = it['class'] + "/" + it['file']; idxs = np.where(aug_keys == k)[0]
        if naug <= len(idxs): sel = idxs[::len(idxs)//naug][:naug]
        else:
            sel = list(idxs)
            rng = random.Random(hash(k) % 2**31)
            while len(sel) < naug: sel.append(rng.choice(idxs))
        for idx in sel:
            aug = aug_features[idx]
            if feat_type == "cls": fe = aug[:FD]
            elif feat_type == "pm": fe = aug[FD:2*FD]
            elif feat_type == "cls_pm_avg": fe = (aug[:FD] + aug[FD:2*FD]) / 2.0
            elif feat_type == "cls_pm_cat": fe = aug[:2*FD]
            f.append(fe); l.append(c2i[it["class"]])
    return np.array(f, dtype=np.float32), np.array(l, dtype=np.int64)

def feat_dim(feat_type):
    return 2*FD if feat_type == "cls_pm_cat" else FD

# ===================== 模型定义 =====================
class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__(); self.s = s; self.m = m
    def forward(self, logits, targets):
        cos = logits.clamp(-1+1e-7, 1-1e-7)
        cos_m = cos * math.cos(self.m) - torch.sqrt(1-cos**2) * math.sin(self.m)
        onehot = F.one_hot(targets, num_classes=NC).float()
        return F.cross_entropy(self.s * (onehot*cos_m + (1-onehot)*cos), targets)

class CosFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.35):
        super().__init__(); self.s = s; self.m = m
    def forward(self, logits, targets):
        onehot = F.one_hot(targets, num_classes=NC).float()
        return F.cross_entropy(self.s * (logits - onehot*self.m), targets)

class FlexHead(nn.Module):
    """灵活分类头: 可选MLP/Linear, 可选BN/Dropout, 可选归一化(ArcFace)"""
    def __init__(self, in_dim, hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True):
        super().__init__()
        if use_mlp:
            layers = [nn.Linear(in_dim, hid)]
            if use_bn: layers.append(nn.BatchNorm1d(hid))
            layers.append(nn.ReLU())
            if use_drop: layers.append(nn.Dropout(drop))
            self.encoder = nn.Sequential(*layers); out_dim = hid
        else:
            self.encoder = nn.Identity(); out_dim = in_dim
        self.normalize = normalize
        self.weight = nn.Parameter(torch.randn(NC, out_dim) * 0.02)
    def forward(self, x):
        feat = self.encoder(x)
        if self.normalize:
            feat = F.normalize(feat, dim=-1); w = F.normalize(self.weight, dim=-1)
            return torch.matmul(feat, w.t())
        else:
            return torch.matmul(feat, self.weight.t()) + self.weight.new_zeros(NC)

def run_one(feat_type, head_kwargs, loss_name, loss_kwargs, naug, seed):
    """运行单个实验配置"""
    set_seed(seed)
    dim = feat_dim(feat_type)
    tr = stratified_sample(train_pool, 0.25, seed)
    tr_raw, tr_l = extract_feat(tr, feat_type)
    if naug > 0:
        aug_f, aug_l = extract_aug_feat(tr, feat_type, naug)
        tr_x = np.concatenate([tr_raw, aug_f]); tr_y = np.concatenate([tr_l, aug_l])
    else:
        tr_x, tr_y = tr_raw, tr_l
    va_x, va_l = extract_feat(val_items, feat_type)
    te_x, te_l = extract_feat(test_items, feat_type)
    tr_xt = torch.FloatTensor(tr_x).to(DEVICE); tr_yt = torch.LongTensor(tr_y).to(DEVICE)
    vat = torch.FloatTensor(va_x).to(DEVICE); tet = torch.FloatTensor(te_x).to(DEVICE)

    model = FlexHead(dim, **head_kwargs).to(DEVICE)
    if loss_name == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_name == "arcface":
        loss_fn = ArcFaceLoss(**loss_kwargs)
    elif loss_name == "cosface":
        loss_fn = CosFaceLoss(**loss_kwargs)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = DataLoader(TensorDataset(tr_xt, tr_yt), batch_size=BS, shuffle=True)
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
    return round(float(accuracy_score(te_l, tp)), 4), round(float(balanced_accuracy_score(te_l, tp)), 4)

# ===================== 基线配置 =====================
BASE_HEAD = dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True)
BASE_FEAT = "cls"; BASE_LOSS = "arcface"; BASE_LOSS_KW = dict(s=30, m=0.5); BASE_NAUG = 5

def run_group(group_name, configs):
    """运行一组消融实验"""
    print(f"\n{'='*80}")
    print(f"消融组: {group_name}")
    print(f"{'='*80}")
    results = {}
    for cfg_name, kwargs in configs:
        print(f"\n  --- {cfg_name} ---")
        results[cfg_name] = {}
        for seed in SEEDS:
            ta, tb = run_one(**kwargs, seed=seed)
            results[cfg_name][seed] = {"test": ta, "bal": tb}
            print(f"    seed={seed}: Test={ta} Bal={tb}")
    return results

# ===================== A. 特征消融 =====================
feat_configs = [
    ("CLS(1024)", dict(feat_type="cls", head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("PM(1024)", dict(feat_type="pm", head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("(CLS+PM)/2(1024)", dict(feat_type="cls_pm_avg", head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("CLS⊕PM(2048)", dict(feat_type="cls_pm_cat", head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
]
R_A = run_group("A.特征选择", feat_configs)

# ===================== B. 分类头消融 =====================
head_configs = [
    ("MLP完整(BN+Drop)", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("MLP无BN", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=False, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("MLP无Dropout", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=False, drop=0.0, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("MLP无BN无Drop", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=False, use_drop=False, drop=0.0, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("Linear", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=False, use_bn=False, use_drop=False, drop=0.0, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
]
R_B = run_group("B.分类头结构", head_configs)

# ===================== C. 损失函数消融 =====================
loss_configs = [
    ("CE", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=False), loss_name="ce", loss_kwargs={}, naug=BASE_NAUG)),
    ("ArcFace m=0.2", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name="arcface", loss_kwargs=dict(s=30, m=0.2), naug=BASE_NAUG)),
    ("ArcFace m=0.5(基线)", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name="arcface", loss_kwargs=dict(s=30, m=0.5), naug=BASE_NAUG)),
    ("ArcFace m=1.0", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name="arcface", loss_kwargs=dict(s=30, m=1.0), naug=BASE_NAUG)),
    ("CosFace m=0.35", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name="cosface", loss_kwargs=dict(s=30, m=0.35), naug=BASE_NAUG)),
    ("ArcFace s=50,m=0.5", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name="arcface", loss_kwargs=dict(s=50, m=0.5), naug=BASE_NAUG)),
]
R_C = run_group("C.损失函数", loss_configs)

# ===================== D. 数据增强消融 =====================
aug_configs = [
    ("0x(无增强)", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=0)),
    ("1x", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=1)),
    ("3x", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=3)),
    ("5x(基线)", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=5)),
    ("10x", dict(feat_type=BASE_FEAT, head_kwargs=BASE_HEAD, loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=10)),
]
R_D = run_group("D.数据增强", aug_configs)

# ===================== E. 隐藏维度消融 =====================
hid_configs = [
    ("hid=64", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=64, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("hid=128", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=128, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("hid=256(基线)", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("hid=512", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=512, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
]
R_E = run_group("E.隐藏维度", hid_configs)

# ===================== F. Dropout消融 =====================
drop_configs = [
    ("drop=0.0", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.0, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("drop=0.3", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.3, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("drop=0.5(基线)", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.5, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
    ("drop=0.7", dict(feat_type=BASE_FEAT, head_kwargs=dict(hid=256, use_mlp=True, use_bn=True, use_drop=True, drop=0.7, normalize=True), loss_name=BASE_LOSS, loss_kwargs=BASE_LOSS_KW, naug=BASE_NAUG)),
]
R_F = run_group("F.Dropout率", drop_configs)

# ===================== 保存 =====================
all_results = {"A_feature": R_A, "B_head": R_B, "C_loss": R_C, "D_aug": R_D, "E_hidden": R_E, "F_dropout": R_F}
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n保存: {OUT_FILE}")

# ===================== 汇总 =====================
print("\n" + "="*80)
print("消融实验汇总 (25%, 3 seeds Bal Acc 均值±标准差)")
print("="*80)
for group_name, group_data in [("A.特征选择", R_A), ("B.分类头结构", R_B), ("C.损失函数", R_C),
                               ("D.数据增强", R_D), ("E.隐藏维度", R_E), ("F.Dropout率", R_F)]:
    print(f"\n  {group_name}")
    print(f"  {'配置':>25} {'Bal Acc':>18} {'Test Acc':>18}")
    for cfg_name, data in group_data.items():
        bals = [data[s]["bal"] for s in SEEDS]; tests = [data[s]["test"] for s in SEEDS]
        print(f"  {cfg_name:>25} {np.mean(bals):.4f}±{np.std(bals):.4f}   {np.mean(tests):.4f}±{np.std(tests):.4f}")