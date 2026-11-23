"""Post-training analysis of the chosen run: temperature scaling and top misclassified images."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from colab.evaluation import collect_logits
from colab.models_zoo import build
from colab.train_arch import ManifestDataset
from src.calibrate import calibration_metrics
from src.dataset import eval_transform


def _logits(model, paths, classes, root, device):
    loader = DataLoader(ManifestDataset(paths, classes, root, eval_transform()), batch_size=64, num_workers=2)
    return collect_logits(model, loader, device)


def calibrate_run(run_dir: Path, manifest: dict, root: Path, device) -> dict:
    """Fit temperature on validation only (as src/calibrate.py does) and store it in best.pt."""
    run_dir, device = Path(run_dir), torch.device(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = build(checkpoint["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    val_logits, val_labels = _logits(model, manifest["validation"], classes, root, device)
    test_logits, test_labels = _logits(model, manifest["test"], classes, root, device)

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def objective():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(objective)
    value = float(temperature.detach().clamp(min=0.05).item())
    report = {
        "run_id": run_dir.name, "fitted_on": "validation", "temperature": value,
        "validation_before": calibration_metrics(val_logits, val_labels, 1.0),
        "validation_after": calibration_metrics(val_logits, val_labels, value),
        "test_before": calibration_metrics(test_logits, test_labels, 1.0),
        "test_after": calibration_metrics(test_logits, test_labels, value),
    }
    (run_dir / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint["temperature"] = value
    partial = run_dir / "best.pt.partial"
    torch.save(checkpoint, partial)
    partial.replace(run_dir / "best.pt")
    return report


def misclassified_gallery(run_dir: Path, root: Path, top_n: int = 15) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    run_dir = Path(run_dir)
    with (run_dir / "test_predictions.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    wrong = sorted((row for row in rows if row["correct"] == "False"), key=lambda row: float(row["confidence"]),
                   reverse=True)[:top_n]
    with (run_dir / "misclassified_top15.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(wrong)
    columns = 5
    figure, axes = plt.subplots(max(1, -(-len(wrong) // columns)), columns, figsize=(18, 11), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, wrong):
        with Image.open(Path(root) / row["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"Đúng: {row['true_label']}\nĐoán: {row['predicted_label']} ({float(row['confidence']):.0%})",
                       fontsize=9)
    figure.tight_layout()
    output = run_dir / "misclassified_top15.png"
    figure.savefig(output, dpi=120); plt.close(figure)
    return output
