"""Create the required CNN-versus-EfficientNet metrics comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def load_metrics(name: str) -> dict:
    path = OUTPUTS_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Train and evaluate the required model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cnn, efficientnet = load_metrics("cnn_metrics.json"), load_metrics("metrics.json")
    rows = []
    for name, metrics in (("CNN baseline", cnn), ("EfficientNet B0", efficientnet)):
        report = metrics["report"]
        rows.append({"model": name, "test_accuracy": metrics["test_accuracy"],
                     "macro_f1": report["macro avg"]["f1-score"], "top3_accuracy": metrics["top3_accuracy"]})
    with (OUTPUTS_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    figure_dir = OUTPUTS_DIR / "figures"; figure_dir.mkdir(exist_ok=True)
    names = [row["model"] for row in rows]; labels = ["Test accuracy", "Macro F1", "Top-3 accuracy"]
    values = [[row["test_accuracy"], row["macro_f1"], row["top3_accuracy"]] for row in rows]
    x = range(len(labels)); width = .34
    plt.figure(figsize=(9, 5))
    for index, (name, model_values) in enumerate(zip(names, values)):
        plt.bar([item + (index - .5) * width for item in x], model_values, width=width, label=name)
    plt.xticks(list(x), labels); plt.ylim(0, 1); plt.ylabel("Score"); plt.title("CNN Baseline vs EfficientNet B0")
    plt.legend(); plt.tight_layout(); plt.savefig(figure_dir / "model_comparison.png", dpi=180); plt.close()
    print("Saved outputs/model_comparison.csv and outputs/figures/model_comparison.png")


if __name__ == "__main__": main()
