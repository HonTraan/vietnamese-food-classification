"""Train EfficientNet-B0 for Vietnamese food classification."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:  # Works both as `python src/train.py` and as package import from app.py.
    from .dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, train_transform, eval_transform
except ImportError:
    from dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, train_transform, eval_transform

ROOT = Path(__file__).resolve().parents[1]; MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, num_classes))
    return model


@torch.inference_mode()
def evaluate_loader(model, loader, criterion, device):
    model.eval(); loss = correct = total = 0; y_true = []; y_pred = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device); logits = model(images)
        loss += criterion(logits, labels).item() * labels.size(0); pred = logits.argmax(1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
        y_true.extend(labels.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
    return loss / total, correct / total, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Train the full EfficientNet backbone too; slower and best suited to a GPU.")
    parser.add_argument("--resume", type=Path,
                        help="Resume from a compatible checkpoint and preserve it unless validation accuracy improves.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing in [0, 1); try 0.05 when fine-tuning.")
    parser.add_argument("--patience", type=int, default=4,
                        help="Stop after this many epochs without validation improvement.")
    parser.add_argument("--class-weighted", action="store_true",
                        help="Use inverse-frequency class weights in cross-entropy.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or not 0 <= args.label_smoothing < 1:
        raise SystemExit("epochs, batch-size, and patience must be positive; --label-smoothing must be in [0, 1).")
    if not RAW_DIR.exists():
        raise SystemExit("Dataset missing. Run: python src/dataset.py --max-per-class 600")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    manifest_path = OUTPUTS_DIR / "clean_split_manifest.json"
    test_paths = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = manifest["classes"]
        train_ds = ManifestImageFolder(manifest["train"], train_transform())
        val_ds = ManifestImageFolder(manifest["validation"], eval_transform())
        test_paths = manifest["test"]
        train_ids = np.arange(len(train_ds))
        val_ids = np.arange(len(val_ds))
    else:
        try:
            folder = ImageFolder(RAW_DIR)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Invalid dataset at {RAW_DIR}: {error}") from error
        labels = [label for _, label in folder.samples]
        if len(folder.classes) < 2 or not labels:
            raise SystemExit("At least two non-empty class folders are required in data/raw/.")
        per_class = {name: labels.count(index) for index, name in enumerate(folder.classes)}
        too_small = {name: count for name, count in per_class.items() if count < 7}
        if too_small:
            raise SystemExit(f"Each class needs at least 7 images for a stratified split. Found: {too_small}")
        indices = np.arange(len(labels))
        train_ids, held_out = train_test_split(indices, test_size=.30, stratify=labels, random_state=args.seed)
        val_ids, test_ids = train_test_split(held_out, test_size=.50, stratify=[labels[i] for i in held_out], random_state=args.seed)
        classes = folder.classes
        train_ds, val_ds = IndexedImageFolder(train_ids, train_transform()), IndexedImageFolder(val_ids, eval_transform())
        test_paths = None
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **options); val_loader = DataLoader(val_ds, **options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_backbone = not args.fine_tune
    model = build_model(len(classes), pretrained=not args.no_pretrained and args.resume is None,
                        freeze_backbone=freeze_backbone).to(device)
    best = -1.0
    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("classes") != classes:
            raise SystemExit("Checkpoint classes do not match data/raw/. Use a checkpoint trained on this dataset.")
        model.load_state_dict(checkpoint["model_state"])
        best = checkpoint.get("best_val_accuracy", -1.0)
        print(f"Resumed {args.resume}; validation baseline={best:.3f}")
    class_weights = None
    if args.class_weighted:
        if test_paths is not None:
            counts = {name: 0 for name in classes}
            for relative_path in manifest["train"]:
                class_name = Path(relative_path).parts[1]
                counts[class_name] += 1
        else:
            counts = {name: len(train_ds.base.samples) for name in []}
        if test_paths is not None:
            weights = [len(classes) / counts[name] for name in classes]
            weight_sum = sum(weights)
            class_weights = torch.tensor(
                [weight * len(classes) / weight_sum for weight in weights],
                dtype=torch.float32,
                device=device,
            )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                      lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    mode = "classifier head only" if freeze_backbone else "full fine-tuning"
    test_count = len(test_paths) if test_paths is not None else 0
    print(f"Device={device}; mode={mode}; samples train/val/test={len(train_ids)}/{len(val_ids)}/{test_count}")
    improved = False; stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if freeze_backbone:
            # Keep pretrained BatchNorm statistics fixed while only the classifier learns.
            model.features.eval()
        total = correct = 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels_batch); loss.backward(); optimizer.step()
            total += labels_batch.size(0); correct += (logits.argmax(1) == labels_batch).sum().item()
        _, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, device); scheduler.step()
        print(f"Epoch {epoch:02d}: train_acc={correct/total:.3f}, val_acc={val_acc:.3f}")
        history.append({"epoch": epoch, "train_accuracy": correct / total, "validation_accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_acc > best:
            best = val_acc; stale_epochs = 0
            improved = True
            checkpoint_payload = {"model_state": model.state_dict(), "classes": classes,
                "seed": args.seed, "best_val_accuracy": val_acc,
                "frozen_backbone": freeze_backbone}
            if test_paths is not None:
                checkpoint_payload["test_paths"] = test_paths
            else:
                checkpoint_payload["test_indices"] = test_ids.tolist()
            torch.save(checkpoint_payload, MODELS_DIR / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation accuracy={best:.3f}")
                break
    (OUTPUTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (OUTPUTS_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["epoch", "train_accuracy", "validation_accuracy", "learning_rate"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    if improved:
        print(f"Training complete. Best validation accuracy={best:.3f}; saved {MODELS_DIR / 'best_model.pt'}")
    else:
        print(f"Training complete. No validation improvement beyond {best:.3f}; existing checkpoint was kept.")


if __name__ == "__main__": main()
