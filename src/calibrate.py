"""Fit temperature scaling on the validation split and measure test calibration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import ManifestImageFolder, eval_transform
    from .train import build_model
except ImportError:
    from dataset import ManifestImageFolder, eval_transform
    from train import build_model

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "best_model.pt"


def collect_logits(model, paths: list[str], device):
    dataset = ManifestImageFolder(paths, eval_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    logits_list = []
    labels_list = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits_list.append(model(images.to(device)).cpu())
            labels_list.append(labels)
    return torch.cat(logits_list), torch.cat(labels_list)


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float,
                       bins: int = 15) -> dict:
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels)
    ece = 0.0
    mce = 0.0
    reliability = []
    bin_edges = torch.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = confidences.gt(left) & (confidences.le(right) if right == 1.0 else confidences.lt(right))
        count = mask.sum().item()
        if count:
            accuracy = correctness[mask].float().mean().item()
            confidence = confidences[mask].mean().item()
            gap = abs(accuracy - confidence)
            ece += (count / labels.numel()) * gap
            mce = max(mce, gap)
            reliability.append({
                "bin_left": left.item(),
                "bin_right": right.item(),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
            })
    brier = ((probabilities - torch.nn.functional.one_hot(labels, probabilities.size(1)).float()) ** 2).sum(dim=1).mean().item()
    nll = torch.nn.functional.cross_entropy(logits / temperature, labels).item()
    return {
        "temperature": temperature,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "nll": nll,
        "accuracy": correctness.float().mean().item(),
        "reliability": reliability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Checkpoint missing: {MODEL_PATH}")

    manifest_path = OUTPUTS_DIR / "clean_split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing clean_split_manifest.json. Run src/data_cleaning.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = build_model(len(checkpoint["classes"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    validation_logits, validation_labels = collect_logits(model, manifest["validation"], device)
    test_logits, test_labels = collect_logits(model, manifest["test"], device)

    temperature = nn.Parameter(torch.ones(1, device="cpu"))
    optimizer = torch.optim.LBFGS([temperature], lr=0.5, max_iter=args.max_iterations,
                                  line_search_fn="strong_wolfe")
    def objective():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(validation_logits / temperature, validation_labels)
        loss.backward()
        return loss

    optimizer.step(objective)
    temperature_value = float(temperature.detach().clamp(min=0.05).item())

    before_validation = calibration_metrics(validation_logits, validation_labels, 1.0)
    after_validation = calibration_metrics(validation_logits, validation_labels, temperature_value)
    before_test = calibration_metrics(test_logits, test_labels, 1.0)
    after_test = calibration_metrics(test_logits, test_labels, temperature_value)

    report = {
        "fitted_on": "validation",
        "temperature": temperature_value,
        "validation_before": before_validation,
        "validation_after": after_validation,
        "test_before": before_test,
        "test_after": after_test,
    }
    report_path = OUTPUTS_DIR / "calibration.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUTPUTS_DIR / "calibration_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["metric", "validation_before", "validation_after", "test_before", "test_after"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in ("temperature", "ece", "mce", "brier", "nll", "accuracy"):
            writer.writerow({
                "metric": metric_name,
                "validation_before": before_validation[metric_name],
                "validation_after": after_validation[metric_name],
                "test_before": before_test[metric_name],
                "test_after": after_test[metric_name],
            })

    with (OUTPUTS_DIR / "calibration_reliability.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["split", "phase", "bin_left", "bin_right", "count", "accuracy", "confidence"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, before, after in (
            ("validation", before_validation, after_validation),
            ("test", before_test, after_test),
        ):
            for phase_name, result in (("before", before), ("after", after)):
                for bin_result in result["reliability"]:
                    writer.writerow({
                        "split": split_name,
                        "phase": phase_name,
                        **bin_result,
                    })

    checkpoint["temperature"] = temperature_value
    torch.save(checkpoint, MODEL_PATH)
    print(f"Temperature={temperature_value:.4f}")
    print(
        "Validation ECE/NLL "
        f"{before_validation['ece']:.4f}/{before_validation['nll']:.4f} "
        f"-> {after_validation['ece']:.4f}/{after_validation['nll']:.4f}"
    )
    print(
        "Test ECE/NLL "
        f"{before_test['ece']:.4f}/{before_test['nll']:.4f} "
        f"-> {after_test['ece']:.4f}/{after_test['nll']:.4f}"
    )
    print(f"Saved calibration report: {report_path}")
    print("Added temperature to models/best_model.pt")


if __name__ == "__main__":
    main()
