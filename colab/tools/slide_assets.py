"""Tables and figures for the slides, computed from the real models and the finished runs.

Run: python -m colab.tools.slide_assets [--runs colab/results] [--out colab/results/_slides]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn

from colab import compare
from colab.models_zoo import ARCHS, build, count_params, head

IMAGE = (1, 3, 224, 224)


def _shape(tensor: torch.Tensor) -> str:
    return "×".join(map(str, tensor.shape[1:]))


def _first_tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _trace(model: nn.Module, modules: dict[str, nn.Module]) -> dict[str, tuple[str, str, int]]:
    """Run one 224×224 image through the model; return name -> (in_shape, out_shape, macs) for the given modules."""
    seen: dict[str, tuple[str, str, int]] = {}

    def hook(name):
        def record(module, inputs, output):
            out = _first_tensor(output)
            macs = 0
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                macs = out[0].numel() * (module.in_channels // module.groups) * kh * kw
            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
            seen.setdefault(name, (_shape(_first_tensor(inputs)), _shape(out), macs))
        return record

    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(IMAGE))
    for handle in handles:
        handle.remove()
    return seen


def _leaves(model: nn.Module) -> dict[str, nn.Module]:
    return {name: module for name, module in model.named_modules() if name and not list(module.children())}


def _pair(value) -> str:
    if value is None:
        return ""
    return "×".join(map(str, value)) if isinstance(value, tuple) else str(value)


def layer_table(arch: str, num_classes: int = 30) -> list[dict]:
    model = build(arch, num_classes, pretrained=False)
    modules = _leaves(model)
    if arch == "resnet50":
        modules.update({f"layer{i}": getattr(model, f"layer{i}") for i in range(1, 5)})
    traced = _trace(model, modules)
    rows = []
    for name, module in modules.items():
        in_shape, out_shape, _ = traced.get(name, ("", "", 0))
        rows.append({"name": name, "type": type(module).__name__, "in_shape": in_shape, "out_shape": out_shape,
                     "kernel": _pair(getattr(module, "kernel_size", None)), "stride": _pair(getattr(module, "stride", None)),
                     "padding": _pair(getattr(module, "padding", None)), "params": count_params(module),
                     "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)})
    return rows


def resnet_stage_summary(num_classes: int = 30) -> list[dict]:
    model = build("resnet50", num_classes, pretrained=False)
    stages = {"conv1": model.conv1, "maxpool": model.maxpool, **{f"layer{i}": getattr(model, f"layer{i}") for i in range(1, 5)},
              "avgpool": model.avgpool, "fc": model.fc}
    traced = _trace(model, stages)
    return [{"stage": name, "blocks": len(module) if isinstance(module, nn.Sequential) else 1,
             "in_shape": traced[name][0], "out_shape": traced[name][1],
             # the conv1 stage is conv1 + bn1 + relu, so its BatchNorm counts here
             "params": count_params(module) + (count_params(model.bn1) if name == "conv1" else 0)}
            for name, module in stages.items()]


def weight_layer_count(arch: str) -> int:
    model = build(arch, 30, pretrained=False)
    return sum(1 for name, module in model.named_modules()
               if isinstance(module, (nn.Conv2d, nn.Linear)) and "downsample" not in name)


def model_summary(arch: str, num_classes: int = 30) -> dict:
    model = build(arch, num_classes, pretrained=False)
    leaves = _leaves(model)
    traced = _trace(model, {**leaves, "_features": model.layer4 if arch == "resnet50" else model.features})
    head_module = model.classifier if arch == "cnn" else head(model, arch)
    frozen = None
    if arch != "cnn":
        frozen_model = build(arch, num_classes, pretrained=False, freeze_backbone=True)
        frozen = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    return {"arch": arch, "total_params": count_params(model), "head_params": count_params(head_module),
            "frozen_trainable_params": frozen, "macs": sum(traced[name][2] for name in leaves if name in traced),
            "final_feature_shape": traced["_features"][1]}


# ---------------------------------------------------------------- figures and tables from finished runs

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _history(run_dir: Path) -> list[dict]:
    with (Path(run_dir) / "history.csv").open(encoding="utf-8") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def _metrics(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def grid_run_ids(rows: list[dict]) -> list[str]:
    """Runs shown in the learning-curve grid: every finished run, in compare's order."""
    return [row["run_id"] for row in rows]


def learning_curves_grid(rows: list[dict], out: Path, metric: str = "accuracy") -> Path:
    """Landscape grid (4 columns) of train vs validation `metric` per run; the dashed line marks the best val epoch."""
    plt = _pyplot()
    columns = 4
    lines = -(-len(rows) // columns)
    figure, axes = plt.subplots(lines, columns, figsize=(16, 3.3 * lines + 0.8), squeeze=False)
    for axis, row in zip(axes.flat, rows):
        history = _history(row["run_dir"])
        epochs = [h["epoch"] for h in history]
        axis.plot(epochs, [h[f"train_{metric}"] for h in history], label="Train")
        axis.plot(epochs, [h[f"val_{metric}"] for h in history], label="Validation")
        best = _metrics(row["run_dir"]).get("best_epoch")
        if best:
            axis.axvline(best, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(row["run_id"], fontsize=10)
        axis.set_xlabel("Epoch", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(rows):]:
        axis.axis("off")
    name = "Accuracy" if metric == "accuracy" else "Loss"
    figure.suptitle(f"{name}: train vs validation của mọi run (vạch đứt: epoch có val tốt nhất)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96)); figure.savefig(out, dpi=150); plt.close(figure)
    return Path(out)


def train_val_gap(rows: list[dict], out: Path) -> Path:
    plt = _pyplot()
    gaps = []
    for row in rows:
        history = _history(row["run_dir"])
        best = max(history, key=lambda h: h["val_accuracy"])
        gaps.append((row["run_id"], (best["train_accuracy"] - best["val_accuracy"]) * 100))
    figure, axis = plt.subplots(figsize=(8, 0.45 * len(gaps) + 1.2))
    axis.barh([g[0] for g in gaps], [g[1] for g in gaps], color="tab:orange")
    axis.axvline(0, color="black", linewidth=0.8)
    for y, (_, gap) in enumerate(gaps):
        axis.annotate(f"{gap:+.1f}", (gap, y), xytext=(3 if gap >= 0 else -3, 0), textcoords="offset points",
                      ha="left" if gap >= 0 else "right", va="center", fontsize=8)
    values = [g[1] for g in gaps]
    pad = max(1.0, max(abs(v) for v in values) * 0.15)
    axis.set_xlim(min(0, min(values)) - pad, max(0, max(values)) + pad)
    axis.invert_yaxis()
    axis.set(xlabel="Train accuracy − val accuracy tại epoch val tốt nhất (điểm %)", title="Mức overfit của từng run")
    figure.tight_layout(); figure.savefig(out, dpi=150); plt.close(figure)
    return Path(out)


def _class_names(report: dict) -> list[str]:
    return [name for name, value in report.items() if isinstance(value, dict) and name not in ("macro avg", "weighted avg")]


def per_class_f1(run_dir: Path, out: Path) -> Path:
    plt = _pyplot()
    report = _metrics(run_dir)["report"]
    scores = sorted(((name, report[name]["f1-score"]) for name in _class_names(report)), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(8, 0.3 * len(scores) + 1.2))
    axis.barh([s[0] for s in scores], [s[1] for s in scores], color="tab:blue")
    for y, (_, score) in enumerate(scores):
        axis.annotate(f"{score:.2f}", (score, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    axis.set_xlim(0, 1.05)
    axis.set(xlabel="F1 trên tập test", title=f"F1 theo từng lớp — {Path(run_dir).name}")
    figure.tight_layout(); figure.savefig(out, dpi=150); plt.close(figure)
    return Path(out)


def row_normalized(matrix: list[list[int]]) -> list[list[float]]:
    return [[value / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]


def confusion_grid(rows: list[dict], out: Path) -> Path:
    import seaborn as sns
    plt = _pyplot()
    arch_rows = [row for row in rows if row["exp"] in ("E1", "E2", "E3", "E4")]
    arch_rows.sort(key=lambda row: row["exp"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 11))
    for axis, row in zip(axes.flat, arch_rows):
        metrics = _metrics(row["run_dir"])
        sns.heatmap(row_normalized(metrics["confusion_matrix"]), ax=axis, cmap="Blues", vmin=0, vmax=1, square=True,
                    xticklabels=False, yticklabels=False, cbar=False)
        axis.set_title(f"{row['run_id']} (test acc {metrics['test_accuracy']:.1%})", fontsize=10)
        axis.set(xlabel="Dự đoán", ylabel="Nhãn thật")
    for axis in list(axes.flat)[len(arch_rows):]:
        axis.axis("off")
    figure.suptitle("Confusion matrix chuẩn hoá theo hàng (đường chéo = recall từng lớp)")
    figure.tight_layout(); figure.savefig(out, dpi=150); plt.close(figure)
    return Path(out)


HYPERPARAM_FIELDS = ["run_id", "arch", "pretrained", "freeze_backbone", "augment", "lr", "label_smoothing", "epochs",
                     "batch_size", "patience", "seed", "optimizer", "weight_decay", "scheduler", "device"]


def hyperparams_table(rows: list[dict]) -> list[dict]:
    table = []
    for row in rows:
        config = json.loads((Path(row["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        scheduler = ("ReduceLROnPlateau(factor=0.5, patience=1)" if config["arch"] == "cnn"
                     else "CosineAnnealingLR(T_max=epochs)")
        table.append({**{key: config.get(key) for key in HYPERPARAM_FIELDS}, "run_id": row["run_id"],
                      "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": scheduler})
    return table


def _write_csv(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _final_run_dir(runs_dir: Path, rows: list[dict]) -> Path:
    final = Path(runs_dir) / "_compare" / "final.json"
    if final.is_file():
        return Path(runs_dir) / json.loads(final.read_text(encoding="utf-8"))["run_id"]
    return Path(compare.pick_best(rows)["run_dir"])


def main(runs_dir: Path, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compare.collect(runs_dir)
    archs = [arch for arch in ARCHS if arch != "convnext_tiny"]
    for arch in archs:
        _write_csv(out_dir / f"layers_{arch}.csv", layer_table(arch))
    _write_csv(out_dir / "resnet50_stages.csv", resnet_stage_summary())
    _write_csv(out_dir / "model_summary.csv", [model_summary(arch) for arch in archs])
    _write_csv(out_dir / "hyperparams.csv", hyperparams_table(rows))
    final_dir = _final_run_dir(runs_dir, rows)
    learning_curves_grid(rows, out_dir / "learning_curves_grid.png", "accuracy")
    learning_curves_grid(rows, out_dir / "learning_curves_grid_loss.png", "loss")
    train_val_gap(rows, out_dir / "train_val_gap.png")
    per_class_f1(final_dir, out_dir / f"per_class_f1_{final_dir.name}.png")
    confusion_grid(rows, out_dir / "confusion_grid.png")
    print(f"Runs: {', '.join(grid_run_ids(rows))}")
    for path in sorted(out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("colab/results"))
    parser.add_argument("--out", type=Path, default=Path("colab/results/_slides"))
    args = parser.parse_args()
    main(args.runs, args.out)
