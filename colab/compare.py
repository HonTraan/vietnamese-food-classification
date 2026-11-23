"""Collect finished runs into one comparison table and the charts used in the slides.

Model choice is always made on validation accuracy; test numbers are only reported.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from colab.experiments import EXPERIMENTS, PRIORITY, RunConfig

FIELDS = ["run_id", "exp", "arch", "pretrained", "freeze_backbone", "augment", "lr", "label_smoothing", "seed",
          "best_val_accuracy", "test_accuracy", "macro_f1", "top3_accuracy", "params", "train_seconds",
          "gpu_images_per_sec", "cpu_ms_per_image"]
REQUIRED_FOR_TUNING = ("E1", "E2", "E3", "E4")


def _order(row: dict) -> tuple:
    exp = row["exp"]
    return (PRIORITY.index(exp) if exp in PRIORITY else len(PRIORITY), row["run_id"])


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()) if Path(runs_dir).is_dir() else []:
        if not (run_dir / "DONE").is_file():
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        row = {"run_id": run_dir.name, "exp": run_dir.name.split("_")[0]}
        row.update({key: config.get(key) for key in FIELDS[2:9]})
        row.update({key: metrics.get(key) for key in FIELDS[9:]})
        row["run_dir"] = str(run_dir)
        rows.append(row)
    return sorted(rows, key=_order)


def pick_best(rows: list[dict], prefix: str | None = None) -> dict:
    candidates = [row for row in rows if prefix is None or row["exp"].startswith(prefix)]
    if not candidates:
        raise ValueError(f"No finished runs with prefix {prefix!r}")
    best_val = max(row["best_val_accuracy"] for row in candidates)
    tied = [row for row in candidates if row["best_val_accuracy"] == best_val]
    best_f1 = max(row["macro_f1"] for row in tied)
    return min((row for row in tied if row["macro_f1"] == best_f1), key=lambda row: row["run_id"])


def pin_tuning_base(runs_dir: Path, rows: list[dict]) -> RunConfig:
    """Choose the base architecture for T* runs once (best validation among E*), then keep it fixed."""
    pin = Path(runs_dir) / "tuning_base.json"
    if pin.is_file():
        return RunConfig(**json.loads(pin.read_text(encoding="utf-8"))["config"])
    done = {row["exp"] for row in rows}
    missing = [exp for exp in REQUIRED_FOR_TUNING if exp not in done]
    if missing:
        raise RuntimeError(f"Cần chạy xong {', '.join(missing)} trước khi chọn base cho tinh chỉnh.")
    best = pick_best(rows, prefix="E")
    base = EXPERIMENTS[best["exp"]]
    pin.write_text(json.dumps({"base_exp": best["exp"], "run_id": best["run_id"], "config": base.to_dict(),
                               "best_val_accuracy": best["best_val_accuracy"]}, indent=2), encoding="utf-8")
    return base


def write_comparison(rows: list[dict], out_dir: Path, plots: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        _plot(rows, out_dir)


def _plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch_rows = [row for row in rows if row["exp"].startswith("E")]
    if arch_rows:
        names = [row["arch"] for row in arch_rows]
        positions = range(len(arch_rows))
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.bar([p - 0.2 for p in positions], [row["test_accuracy"] for row in arch_rows], 0.4, label="Test accuracy")
        axis.bar([p + 0.2 for p in positions], [row["macro_f1"] for row in arch_rows], 0.4, label="Macro F1")
        axis.set_xticks(list(positions), names, rotation=15); axis.set_ylim(0, 1); axis.legend()
        axis.set_title("So sánh kiến trúc (cùng split sạch)")
        figure.tight_layout(); figure.savefig(out_dir / "comparison_arch.png", dpi=150); plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        for row in arch_rows:
            axis.scatter(row["params"] / 1e6, row["test_accuracy"])
            axis.annotate(row["arch"], (row["params"] / 1e6, row["test_accuracy"]))
        axis.set(xlabel="Số tham số (triệu)", ylabel="Test accuracy", title="Accuracy vs kích thước mô hình")
        figure.tight_layout(); figure.savefig(out_dir / "accuracy_vs_params.png", dpi=150); plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for row in arch_rows:
            with (Path(row["run_dir"]) / "history.csv").open(encoding="utf-8") as file:
                history = list(csv.DictReader(file))
            axis.plot([int(h["epoch"]) for h in history], [float(h["val_accuracy"]) for h in history], label=row["arch"])
        axis.set(xlabel="Epoch", ylabel="Validation accuracy", title="Learning curves (validation)"); axis.legend()
        figure.tight_layout(); figure.savefig(out_dir / "learning_curves_all.png", dpi=150); plt.close(figure)

    pin = Path(rows[0]["run_dir"]).parent / "tuning_base.json" if rows else None
    tuning_rows = [row for row in rows if row["exp"].startswith("T")]
    if tuning_rows and pin and pin.is_file():
        base_id = json.loads(pin.read_text(encoding="utf-8"))["run_id"]
        shown = [row for row in rows if row["run_id"] == base_id] + tuning_rows
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.barh([row["run_id"] for row in shown], [row["test_accuracy"] for row in shown])
        axis.set(xlabel="Test accuracy", title="Tinh chỉnh so với cấu hình chuẩn"); axis.set_xlim(0, 1)
        figure.tight_layout(); figure.savefig(out_dir / "tuning.png", dpi=150); plt.close(figure)
