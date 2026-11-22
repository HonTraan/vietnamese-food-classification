"""Evaluate the best checkpoint and save metrics to outputs/."""
from __future__ import annotations

import json
import csv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
try:
    from .train import build_model, evaluate_loader
    from .dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, eval_transform
except ImportError:
    from train import build_model, evaluate_loader
    from dataset import RAW_DIR, IndexedImageFolder, ManifestImageFolder, eval_transform

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def predict_test_set(model, loader, criterion, device, temperature=1.0):
    model.eval(); actual = []; predicted = []; confidences = []; probabilities = []; total_loss = total = top3_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images) / temperature
        total_loss += criterion(logits, labels).item() * labels.size(0); total += labels.size(0)
        batch_probabilities = torch.softmax(logits, dim=1).cpu()
        actual.extend(labels.tolist()); predicted.extend(batch_probabilities.argmax(1).tolist())
        top3_correct += batch_probabilities.topk(min(3, batch_probabilities.size(1)), dim=1).indices.eq(labels.cpu()[:, None]).any(1).sum().item()
        confidences.extend(batch_probabilities.max(1).values.tolist()); probabilities.extend(batch_probabilities.tolist())
    return total_loss / total, sum(a == b for a, b in zip(actual, predicted)) / total, top3_correct / total, actual, predicted, confidences, probabilities

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT / "models" / "best_model.pt"
    if not model_path.is_file():
        raise SystemExit(f"Checkpoint not found: {model_path}. Run: python src/train.py --epochs 12 --batch-size 32")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = {"model_state", "classes"}
    missing = required.difference(checkpoint)
    if missing:
        raise SystemExit(f"Invalid checkpoint; missing keys: {sorted(missing)}. Train the model again.")
    if not ("test_indices" in checkpoint or "test_paths" in checkpoint):
        raise SystemExit("Invalid checkpoint; missing test split. Train the model again.")
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Dataset not found at {RAW_DIR}. Evaluation needs the same data used for training.")
    current_classes = sorted(path.name for path in RAW_DIR.iterdir() if path.is_dir())
    if current_classes != checkpoint["classes"]:
        raise SystemExit("Class folders changed since training. Restore the original data/raw/ or train a new checkpoint.")
    if checkpoint.get("test_paths"):
        test_dataset = ManifestImageFolder(checkpoint["test_paths"], eval_transform())
    else:
        if not checkpoint["test_indices"]:
            raise SystemExit("Checkpoint contains an empty test split. Train the model again.")
        test_dataset = IndexedImageFolder(checkpoint["test_indices"], eval_transform())
    model = build_model(len(checkpoint["classes"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(test_dataset, batch_size=32)
    temperature = float(checkpoint.get("temperature", 1.0))
    loss, accuracy, top3_accuracy, actual, predicted, confidences, probabilities = predict_test_set(
        model, loader, nn.CrossEntropyLoss(), device, temperature)
    report = classification_report(actual, predicted, target_names=checkpoint["classes"], output_dict=True, zero_division=0)
    output = {"test_loss": loss, "test_accuracy": accuracy, "top3_accuracy": top3_accuracy,
              "temperature": temperature, "report": report,
              "confusion_matrix": confusion_matrix(actual, predicted).tolist()}
    output_dir = ROOT / "outputs"; output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["metric"] + checkpoint["classes"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in ("precision", "recall", "f1-score", "support"):
            row = {"metric": metric_name}
            for class_name in checkpoint["classes"]:
                value = report[class_name][metric_name]
                row[class_name] = int(value) if metric_name == "support" else f"{value:.6f}"
            writer.writerow(row)
    predictions_dir = output_dir / "predictions"; predictions_dir.mkdir(exist_ok=True)
    rows = []
    for dataset_index, true_id, predicted_id, confidence, probability in zip(
            range(len(test_dataset)), actual, predicted, confidences, probabilities):
        if hasattr(test_dataset, "paths"):
            image_path = test_dataset.paths[dataset_index]
        else:
            source_index = test_dataset.indices[dataset_index]
            image_path, _ = test_dataset.base.samples[source_index]
        top3_ids = sorted(range(len(probability)), key=probability.__getitem__, reverse=True)[:3]
        rows.append({"image_path": image_path, "true_label": checkpoint["classes"][true_id],
                     "predicted_label": checkpoint["classes"][predicted_id], "confidence": confidence,
                     "correct": true_id == predicted_id,
                     "top3": "; ".join(f"{checkpoint['classes'][i]} ({probability[i]:.3f})" for i in top3_ids)})
    with (predictions_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    mistakes = sorted((row for row in rows if not row["correct"]), key=lambda row: row["confidence"], reverse=True)[:15]
    with (predictions_dir / "misclassified_top15.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(mistakes)
    figures_dir = output_dir / "figures"; figures_dir.mkdir(exist_ok=True)
    matrix = confusion_matrix(actual, predicted)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label"] + checkpoint["classes"])
        for true_label, row in zip(checkpoint["classes"], matrix):
            writer.writerow([true_label] + row.tolist())
    plt.figure(figsize=(15, 12)); sns.heatmap(matrix, annot=False, cmap="Blues", square=True,
        xticklabels=checkpoint["classes"], yticklabels=checkpoint["classes"])
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("EfficientNet B0 Confusion Matrix")
    plt.tight_layout(); plt.savefig(figures_dir / "confusion_matrix.png", dpi=180); plt.close()
    print(f"Test accuracy: {accuracy:.3f}; top-3 accuracy: {top3_accuracy:.3f}; saved outputs/metrics.json")
    print("Saved predictions, top-15 errors, and outputs/figures/confusion_matrix.png")

if __name__ == "__main__": main()
