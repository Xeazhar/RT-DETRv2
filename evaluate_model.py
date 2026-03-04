import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

import sys
import os

ROOT = r"C:\Users\jazzb\rt-deterv2\RT-DETRv2"
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
os.chdir(ROOT)

# ================================
# 2. Import RT-DETRv2
# ================================
from core import YAMLConfig
from solver import TASKS
import zoo.rtdetr.rtdetr
from core.utils import build_dataloader

# ================================
# 3. Config & Checkpoint
# ================================
CONFIG_PATH = os.path.join(ROOT, "configs", "rtdetrv2", "rtdetrv2_r101vd_6x_coco.yml")
MODEL_PATH = os.path.join(ROOT, "output", "best.pth")

cfg = YAMLConfig(CONFIG_PATH, resume=MODEL_PATH)

# ================================
# 4. Create Solver and Model
# ================================
solver = TASKS[cfg.yaml_cfg["task"]](cfg)
solver.eval()  # forces model creation and loading checkpoint
model = solver.model
model.eval()
device = solver.device

# ================================
# 5. Prepare Dataloaders
# ================================
val_loader = solver.val_dataloader  # built automatically
# Build custom test loader from YAML
test_loader = build_dataloader(cfg.yaml_cfg['test_dataloader'])

# Class names
CLASS_NAMES = ["Alligator Cracks", "Cracks", "Potholes", "Ravelling"]

# ================================
# 6. Helper functions
# ================================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def evaluate_loader(loader, split_name="VAL", threshold=0.7, export_csv=True):
    all_preds, all_gts, iou_list = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)

            outputs = model(images, orig_sizes)
            probs = outputs["pred_logits"].sigmoid()
            scores, labels = probs.max(dim=-1)
            boxes = outputs["pred_boxes"]

            for i in range(len(targets)):
                gt_lbl = targets[i]["labels"].cpu().numpy()
                gt_bx = targets[i]["boxes"].cpu().numpy()

                pred_scores = scores[i].cpu().numpy()
                pred_lbl = labels[i].cpu().numpy()
                pred_bx = boxes[i].cpu().numpy()

                valid = pred_scores > threshold
                pred_lbl = pred_lbl[valid]
                pred_bx = pred_bx[valid]

                if len(gt_lbl) > 0 and len(pred_lbl) > 0:
                    all_gts.append(gt_lbl[0])
                    all_preds.append(pred_lbl[0])

                for pb in pred_bx:
                    best = 0
                    for gb in gt_bx:
                        best = max(best, compute_iou(pb, gb))
                    iou_list.append(best)

    # Precision / Recall / F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gts, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    print(f"\n=== {split_name} Detection Metrics ===")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{name}: P={precision[idx]:.3f}, R={recall[idx]:.3f}, F1={f1[idx]:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9,7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"{split_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # IoU stats
    iou_arr = np.array(iou_list)
    print(f"{split_name} IoU avg: {iou_arr.mean():.3f}, min: {iou_arr.min():.3f}, max: {iou_arr.max():.3f}")

    # Export CSV
    if export_csv:
        df = pd.DataFrame({"GT": all_gts, "Pred": all_preds, "IoU": iou_list})
        csv_name = f"{split_name.lower()}_results.csv"
        df.to_csv(csv_name, index=False)
        print(f"CSV exported: {csv_name}")

# ================================
# 7. Run Evaluations
# ================================
if __name__ == "__main__":
    print("\n>>> Evaluating VAL set")
    evaluate_loader(val_loader, split_name="VAL", threshold=0.7)

    print("\n>>> Evaluating CUSTOM TEST set")
    evaluate_loader(test_loader, split_name="TEST", threshold=0.7)