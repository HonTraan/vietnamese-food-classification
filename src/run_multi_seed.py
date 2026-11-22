"""Run EfficientNet across fixed data splits and summarize seed variance."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--class-weighted", action="store_true")
    args = parser.parse_args()

    manifest_path = OUTPUTS_DIR / "clean_split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing clean_split_manifest.json. Run src/data_cleaning.py first.")

    summaries = []
    for seed in args.seeds:
        command = [
            sys.executable,
            str(ROOT / "src" / "train.py"),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
        ]
        if args.fine_tune:
            command.append("--fine-tune")
        if args.class_weighted:
            command.append("--class-weighted")
        subprocess.run(command, cwd=ROOT, check=True)

        subprocess.run(
            [sys.executable, str(ROOT / "src" / "evaluate.py")],
            cwd=ROOT,
            check=True,
        )
        metrics = json.loads((OUTPUTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        seed_checkpoint = MODELS_DIR / f"best_model_seed_{seed}.pt"
        seed_metrics = OUTPUTS_DIR / f"metrics_seed_{seed}.json"
        shutil.copy2(MODELS_DIR / "best_model.pt", seed_checkpoint)
        shutil.copy2(OUTPUTS_DIR / "metrics.json", seed_metrics)
        summaries.append({
            "seed": seed,
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["report"]["macro avg"]["f1-score"],
            "top3_accuracy": metrics["top3_accuracy"],
        })

    accuracies = [row["test_accuracy"] for row in summaries]
    macro_f1 = [row["macro_f1"] for row in summaries]
    top3 = [row["top3_accuracy"] for row in summaries]
    statistics = {
        "seeds": summaries,
        "test_accuracy_mean": sum(accuracies) / len(accuracies),
        "test_accuracy_std": (sum((value - sum(accuracies) / len(accuracies)) ** 2 for value in accuracies) / len(accuracies)) ** 0.5,
        "macro_f1_mean": sum(macro_f1) / len(macro_f1),
        "macro_f1_std": (sum((value - sum(macro_f1) / len(macro_f1)) ** 2 for value in macro_f1) / len(macro_f1)) ** 0.5,
        "top3_mean": sum(top3) / len(top3),
        "top3_std": (sum((value - sum(top3) / len(top3)) ** 2 for value in top3) / len(top3)) ** 0.5,
    }
    report_path = OUTPUTS_DIR / "multi_seed_results.json"
    report_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_columns = [str(row["seed"]) for row in summaries]
    with (OUTPUTS_DIR / "multi_seed_results.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["metric"] + seed_columns + ["mean", "std"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, result_key, mean_key, std_key in (
            ("test_accuracy", "test_accuracy", "test_accuracy_mean", "test_accuracy_std"),
            ("macro_f1", "macro_f1", "macro_f1_mean", "macro_f1_std"),
            ("top3_accuracy", "top3_accuracy", "top3_mean", "top3_std"),
        ):
            writer.writerow({
                "metric": metric_name,
                **{str(row["seed"]): f"{row[result_key]:.6f}" for row in summaries},
                "mean": f"{statistics[mean_key]:.6f}",
                "std": f"{statistics[std_key]:.6f}",
            })
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
