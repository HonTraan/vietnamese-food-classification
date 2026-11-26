"""Collect finished runs into one comparison table and the charts used in the slides.

Model choice is always made on validation accuracy; test numbers are only reported.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from colab.experiments import EXPERIMENTS, PRIORITY, RunConfig

FIELDS = ["run_id", "exp", "arch", "pretrained", "freeze_backbone", "augment", "lr", "label_smoothing", "seed",
          "device", "best_val_accuracy", "test_accuracy", "macro_f1", "top3_accuracy", "params", "train_seconds",
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
        row.update({key: config.get(key) for key in FIELDS[2:10]})
        row.update({key: metrics.get(key) for key in FIELDS[10:]})
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


def tuning_reference(rows: list[dict], runs_dir: Path) -> dict | None:
    """The bar T* runs are compared with: the C1 control (same machine) once done, else the pinned base."""
    control = [row for row in rows if row["exp"] == "C1"]
    if control:
        return control[0]
    pin = Path(runs_dir) / "tuning_base.json"
    if not pin.is_file():
        return None
    base_id = json.loads(pin.read_text(encoding="utf-8"))["run_id"]
    return next((row for row in rows if row["run_id"] == base_id), None)


def _same_device_reference(row: dict, rows: list[dict], runs_dir: Path) -> dict | None:
    if str(row.get("device") or "").startswith("mps"):
        control = [other for other in rows if other["exp"] == "C1"]
        if control:
            return control[0]
    pin = Path(runs_dir) / "tuning_base.json"
    if not pin.is_file():
        return None
    base_id = json.loads(pin.read_text(encoding="utf-8"))["run_id"]
    return next((other for other in rows if other["run_id"] == base_id), None)


def tuning_deltas(rows: list[dict], runs_dir: Path) -> list[dict]:
    """Each T* run minus its reference on the same hardware (C1 for mps runs, the pinned base for T4 runs)."""
    deltas = []
    for row in rows:
        if not row["exp"].startswith("T"):
            continue
        reference = _same_device_reference(row, rows, runs_dir)
        if reference is None:
            continue
        deltas.append({"run_id": row["run_id"], "device": row.get("device"), "reference": reference["run_id"],
                       "delta_val": row["best_val_accuracy"] - reference["best_val_accuracy"],
                       "delta_test": row["test_accuracy"] - reference["test_accuracy"]})
    return deltas


def hardware_effect(rows: list[dict], runs_dir: Path) -> dict | None:
    """Pinned base (Colab T4) minus C1 (same recipe on the Mac): the cost of switching hardware."""
    control = next((row for row in rows if row["exp"] == "C1"), None)
    pin = Path(runs_dir) / "tuning_base.json"
    if control is None or not pin.is_file():
        return None
    base_id = json.loads(pin.read_text(encoding="utf-8"))["run_id"]
    base = next((row for row in rows if row["run_id"] == base_id), None)
    if base is None:
        return None
    return {"base": base_id, "base_device": base.get("device"), "control": control["run_id"],
            "control_device": control.get("device"),
            "delta_val": base["best_val_accuracy"] - control["best_val_accuracy"],
            "delta_test": base["test_accuracy"] - control["test_accuracy"]}


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

    deltas = tuning_deltas(rows, Path(rows[0]["run_dir"]).parent) if rows else []
    if deltas:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        labels = [f"{d['run_id']}  (vs {d['reference']}, {d['device']})" for d in deltas]
        values = [d["delta_test"] * 100 for d in deltas]
        axis.barh(labels, values, color=["tab:green" if v >= 0 else "tab:red" for v in values])
        axis.axvline(0, color="black", linewidth=0.8)
        pad = max(0.5, max(abs(v) for v in values) * 0.2)
        axis.set_xlim(min(0, min(values)) - pad, max(0, max(values)) + pad)
        for y, value in enumerate(values):
            axis.annotate(f"{value:+.2f}", (value, y), xytext=(3 if value >= 0 else -3, 0),
                          textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8)
        axis.set(xlabel="Chênh lệch test accuracy so với mốc cùng phần cứng (điểm %)",
                 title="Tinh chỉnh: mỗi run chỉ đổi một yếu tố so với mốc")
        figure.tight_layout(); figure.savefig(out_dir / "tuning.png", dpi=150); plt.close(figure)
