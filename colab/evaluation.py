"""Test-set metrics, inference speed and per-run report files."""
from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.inference_mode()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).float().cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)


def evaluate_model(model, loader, device, classes: list[str], temperature: float = 1.0) -> dict:
    logits, labels = collect_logits(model, loader, device)
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=1)
    predicted = probabilities.argmax(1)
    top3 = probabilities.topk(min(3, len(classes)), dim=1)
    actual_list, predicted_list = labels.tolist(), predicted.tolist()
    report = classification_report(actual_list, predicted_list, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    rows = []
    for path, true_id, pred_id, prob, top_ids, top_scores in zip(
            loader.dataset.paths, actual_list, predicted_list, probabilities.tolist(),
            top3.indices.tolist(), top3.values.tolist()):
        rows.append({"image_path": path, "true_label": classes[true_id], "predicted_label": classes[pred_id],
                     "confidence": prob[pred_id], "correct": true_id == pred_id,
                     "top3": "; ".join(f"{classes[i]} ({s:.3f})" for i, s in zip(top_ids, top_scores))})
    return {
        "test_loss": torch.nn.functional.cross_entropy(logits, labels).item(),
        "test_accuracy": predicted.eq(labels).float().mean().item(),
        "macro_f1": report["macro avg"]["f1-score"],
        "top3_accuracy": top3.indices.eq(labels[:, None]).any(1).float().mean().item(),
        "report": report,
        "confusion_matrix": confusion_matrix(actual_list, predicted_list, labels=list(range(len(classes)))).tolist(),
        "rows": rows,
    }


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def measure_speed(model, device) -> dict:
    """gpu_images_per_sec is measured on the accelerator (cuda or mps); see metrics["device"] for which."""
    device = torch.device(device)
    gpu_rate = None
    if device.type in ("cuda", "mps"):
        model.eval()
        batch = torch.randn(32, 3, 224, 224, device=device)
        for _ in range(5):
            model(batch)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(20):
            model(batch)
        synchronize(device)
        gpu_rate = 20 * 32 / (time.perf_counter() - start)
    cpu_model = copy.deepcopy(model).cpu().eval()
    image = torch.randn(1, 3, 224, 224)
    for _ in range(5):
        cpu_model(image)
    start = time.perf_counter()
    for _ in range(20):
        cpu_model(image)
    return {"gpu_images_per_sec": gpu_rate, "cpu_ms_per_image": (time.perf_counter() - start) / 20 * 1000}


def write_artifacts(run_dir: Path, metrics: dict, history: list[dict], classes: list[str]) -> None:
    run_dir = Path(run_dir)
    rows = metrics["rows"]
    summary = {key: value for key, value in metrics.items() if key != "rows"}
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    left.plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    left.set(title="Accuracy", xlabel="Epoch")
    right.plot(epochs, [row["train_loss"] for row in history], label="Train")
    right.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    right.set(title="Loss", xlabel="Epoch")
    left.legend(); right.legend()
    figure.suptitle(run_dir.name); figure.tight_layout()
    figure.savefig(run_dir / "learning_curves.png", dpi=150); plt.close(figure)

    plt.figure(figsize=(15, 12))
    sns.heatmap(metrics["confusion_matrix"], cmap="Blues", square=True, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title(f"{run_dir.name} — confusion matrix")
    plt.tight_layout(); plt.savefig(run_dir / "confusion_matrix.png", dpi=150); plt.close()
