"""
Paper: Few-shot helicopter image classification with DINOv3 (Section 3.4.1, Fig. 5).
Evaluate the LoRA fine-tuned model on the test set and print the confusion matrix.
Usage:  python eval_confusion.py [seed]   (default seed=42; expects models/vit_lora_backbone_s{seed})
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModel
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "crop"
SPLIT_FILE = ROOT / "data" / "splits" / "split_311.json"
MODEL_DIR = ROOT / "models"
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_SIZE = 224

CLASSES = ["AH64","CH47","CH53","Ka27","Ka52","Mi24","Mi26","Mi28","NH90"]
NC = len(CLASSES); FD = 1024; HID = 256; DROP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42


class HeliDataset(Dataset):
    def __init__(self, items, root, transform):
        self.items = items; self.root = root; self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        path = os.path.join(str(self.root), it["class"], it["file"])
        img = Image.open(path).convert("RGB")
        return self.transform(img), CLASSES.index(it["class"])


class MLPArcHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(FD, HID), nn.BatchNorm1d(HID), nn.ReLU(), nn.Dropout(DROP))
        self.weight = nn.Parameter(torch.randn(NC, HID) * 0.02)
    def forward(self, x):
        feat = F.normalize(self.mlp(x), dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return torch.matmul(feat, w.t())


mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    sd = json.load(f)
test_items = sd["split"]["test"]
test_ds = HeliDataset(test_items, DATA_ROOT, eval_tf)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

def main():
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    backbone = PeftModel.from_pretrained(base, MODEL_DIR / f"vit_lora_backbone_s{SEED}")
    backbone = backbone.to(DEVICE).eval()

    head = MLPArcHead().to(DEVICE)
    head.load_state_dict(torch.load(MODEL_DIR / f"vit_lora_arcface_head_s{SEED}.pt", map_location=DEVICE))
    head.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                feat = backbone(x).last_hidden_state[:, 0, :]
            logits = head(feat.float())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    cm = confusion_matrix(labels, preds)
    print("=== 混淆矩阵 ===")
    print(f"{'':>8}" + "".join(f"{c:>7}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>8}" + "".join(f"{v:>7}" for v in row))

    print("\n=== 分类报告 ===")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

if __name__ == "__main__":
    main()