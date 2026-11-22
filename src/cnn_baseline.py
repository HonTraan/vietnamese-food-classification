"""Train and evaluate a CNN baseline on the same deterministic split as EfficientNet."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from .dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, eval_transform, train_transform
except ImportError:
    from dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, eval_transform, train_transform

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR, OUTPUTS_DIR = ROOT / "models", ROOT / "outputs"


class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, images): return self.classifier(self.features(images))


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval(); total_loss = correct = total = 0; actual = []; predicted = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images); total_loss += criterion(logits, labels).item() * labels.size(0)
        probabilities = torch.softmax(logits, 1); predictions = probabilities.argmax(1)
        correct += (predictions == labels).sum().item(); total += labels.size(0)
        actual.extend(labels.cpu().tolist()); predicted.extend(predictions.cpu().tolist())
    return total_loss / total, correct / total, actual, predicted


@torch.inference_mode()
def evaluate_with_top3(model, loader, criterion, device):
    model.eval(); total_loss = correct = top3_correct = total = 0; actual = []; predicted = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device); logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0); probabilities = torch.softmax(logits, 1)
        predictions = probabilities.argmax(1); top3 = probabilities.topk(min(3, probabilities.size(1)), 1).indices
        correct += (predictions == labels).sum().item(); top3_correct += top3.eq(labels[:, None]).any(1).sum().item()
        total += labels.size(0); actual.extend(labels.cpu().tolist()); predicted.extend(predictions.cpu().tolist())
    return total_loss / total, correct / total, top3_correct / total, actual, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1: raise SystemExit("epochs, batch-size, and patience must be positive.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest_path = OUTPUTS_DIR / "clean_split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        folder_classes = manifest["classes"]
        train_dataset = ManifestImageFolder(manifest["train"], train_transform())
        val_dataset = ManifestImageFolder(manifest["validation"], eval_transform())
        test_dataset = ManifestImageFolder(manifest["test"], eval_transform())
        test_paths = manifest["test"]
    else:
        folder = ImageFolder(RAW_DIR); labels = [label for _, label in folder.samples]; indices = np.arange(len(labels))
        train_ids, held_out = train_test_split(indices, test_size=.30, stratify=labels, random_state=args.seed)
        val_ids, test_ids = train_test_split(held_out, test_size=.50, stratify=[labels[i] for i in held_out], random_state=args.seed)
        folder_classes = folder.classes
        train_dataset = IndexedImageFolder(train_ids, train_transform())
        val_dataset = IndexedImageFolder(val_ids, eval_transform())
        test_dataset = IndexedImageFolder(test_ids, eval_transform())
        test_paths = None
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    val_loader = DataLoader(val_dataset, **options)
    test_loader = DataLoader(test_dataset, **options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); MODELS_DIR.mkdir(exist_ok=True); OUTPUTS_DIR.mkdir(exist_ok=True)
    model = CNNBaseline(len(folder_classes)).to(device); criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=1)
    best, stale, history = -1.0, 0, []
    print(f"Device={device}; samples train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = correct = total = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, targets); loss.backward(); optimizer.step()
            train_loss += loss.item() * targets.size(0); correct += (logits.argmax(1) == targets).sum().item(); total += targets.size(0)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device); scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": train_loss / total, "train_accuracy": correct / total,
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        print(f"Epoch {epoch:02d}: train_acc={correct/total:.3f}, val_acc={val_acc:.3f}")
        if val_acc > best:
            best, stale = val_acc, 0
            checkpoint_payload = {"model_state": model.state_dict(), "classes": folder_classes,
                                 "seed": args.seed, "best_val_accuracy": val_acc}
            if test_paths is not None:
                checkpoint_payload["test_paths"] = test_paths
            else:
                checkpoint_payload["test_indices"] = test_ids.tolist()
            torch.save(checkpoint_payload, MODELS_DIR / "cnn_baseline.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation accuracy={best:.3f}"); break
    checkpoint = torch.load(MODELS_DIR / "cnn_baseline.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, top3_acc, actual, predicted = evaluate_with_top3(model, test_loader, criterion, device)
    report = classification_report(actual, predicted, target_names=folder_classes, output_dict=True, zero_division=0)
    metrics = {"test_loss": test_loss, "test_accuracy": test_acc, "top3_accuracy": top3_acc,
               "best_validation_accuracy": checkpoint["best_val_accuracy"], "report": report,
               "confusion_matrix": confusion_matrix(actual, predicted).tolist()}
    (OUTPUTS_DIR / "cnn_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "cnn_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (OUTPUTS_DIR / "cnn_history.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    with (OUTPUTS_DIR / "cnn_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["metric"] + folder_classes
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in ("precision", "recall", "f1-score", "support"):
            row = {"metric": metric_name}
            for class_name in folder_classes:
                value = report[class_name][metric_name]
                row[class_name] = int(value) if metric_name == "support" else f"{value:.6f}"
            writer.writerow(row)
    with (OUTPUTS_DIR / "cnn_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        matrix = confusion_matrix(actual, predicted)
        writer.writerow(["true_label"] + folder_classes)
        for true_label, row in zip(folder_classes, matrix):
            writer.writerow([true_label] + row.tolist())
    figures = OUTPUTS_DIR / "figures"; figures.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 4)); epochs = [item["epoch"] for item in history]
    plt.plot(epochs, [item["train_accuracy"] for item in history], label="Train accuracy")
    plt.plot(epochs, [item["validation_accuracy"] for item in history], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("CNN Baseline Learning Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "cnn_learning_curves.png", dpi=180); plt.close()
    plt.figure(figsize=(15, 12)); sns.heatmap(confusion_matrix(actual, predicted), cmap="Blues", square=True,
        xticklabels=folder_classes, yticklabels=folder_classes)
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("CNN Baseline Confusion Matrix"); plt.tight_layout()
    plt.savefig(figures / "cnn_confusion_matrix.png", dpi=180); plt.close()
    print(f"Test accuracy={test_acc:.3f}; top-3 accuracy={top3_acc:.3f}; saved CNN artifacts.")


if __name__ == "__main__": main()
