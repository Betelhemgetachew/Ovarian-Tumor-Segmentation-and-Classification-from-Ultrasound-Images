"""Evaluate the trained MMOTU models on the official OTU_2d validation split.

Classification: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
Segmentation:   mean Dice and IoU per image at a 0.5 threshold (256 x 256, as in training).

Usage:
    python evaluate.py --data archive/MMOTU/OTU_2d --models Models
"""
import argparse
import json
import os

import albumentations as A
import cv2
import matplotlib
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Label IDs as defined by the MMOTU dataset (train_cls.txt / val_cls.txt)
CLASS_NAMES = [
    "Chocolate Cyst", "Serous Cystadenoma", "Teratoma", "Theca Cell Tumor",
    "Simple Cyst", "Normal Ovary", "Mucinous Cystadenoma", "High-Grade Serous Carcinoma",
]

TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def evaluate_classifier(data_dir, model_path, out_dir, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with open(os.path.join(data_dir, "val_cls.txt")) as f:
        samples = [line.split() for line in f if line.strip()]
    with torch.no_grad():
        for name, label in samples:
            img = TRANSFORM(image=load_rgb(os.path.join(data_dir, "images", name)))["image"]
            y_pred.append(int(model(img.unsqueeze(0).to(device)).argmax(1)))
            y_true.append(int(label))

    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(labels, CLASS_NAMES)
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Classifier confusion matrix (validation set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)

    return {
        "n_images": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
    }


def evaluate_segmentation(data_dir, model_path, device, threshold=0.5):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
                     classes=1, activation="sigmoid")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    with open(os.path.join(data_dir, "val.txt")) as f:
        ids = [line.strip() for line in f if line.strip()]
    dices, ious = [], []
    with torch.no_grad():
        for img_id in ids:
            image = load_rgb(os.path.join(data_dir, "images", f"{img_id}.JPG"))
            mask = (cv2.imread(os.path.join(data_dir, "annotations", f"{img_id}_binary.PNG"), 0) > 0)
            out = TRANSFORM(image=image, mask=mask.astype("float32"))
            pred = model(out["image"].unsqueeze(0).to(device))[0, 0].cpu().numpy() > threshold
            gt = out["mask"].numpy() > 0.5
            inter = np.logical_and(pred, gt).sum()
            total = pred.sum() + gt.sum()
            dices.append((2 * inter + 1e-7) / (total + 1e-7))
            ious.append((inter + 1e-7) / (total - inter + 1e-7))
    return {"n_images": len(ids), "threshold": threshold,
            "mean_dice": float(np.mean(dices)), "mean_iou": float(np.mean(ious))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="archive/MMOTU/OTU_2d")
    parser.add_argument("--models", default="Models")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "classification": evaluate_classifier(
            args.data, os.path.join(args.models, "mmotu_classifier_fullimage_best.pth"), args.out, device),
        "segmentation": evaluate_segmentation(
            args.data, os.path.join(args.models, "mmotu_segmentation_best_model.pth"), device),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    cls, seg = results["classification"], results["segmentation"]
    print(f"Classification: accuracy {cls['accuracy']:.3f}, macro-F1 {cls['macro_f1']:.3f}")
    for name, m in cls["per_class"].items():
        print(f"  {name:30s} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1-score']:.2f}  n={m['support']}")
    print(f"Segmentation: mean Dice {seg['mean_dice']:.3f}, mean IoU {seg['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
