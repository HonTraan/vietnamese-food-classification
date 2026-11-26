"""Train one Colab experiment (any architecture), resumable after a Colab disconnect.

Usage: python -m colab.train_arch --exp E4 --runs-dir /content/drive/MyDrive/DL/vnfood_colab/runs
       python -m colab.train_arch --tune T1 --runs-dir ...    (base from runs/tuning_base.json)
       python -m colab.train_arch --control C1 --runs-dir ... (base re-run on this machine)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from colab import evaluation, session
from colab.experiments import EXPERIMENTS, RunConfig, control, run_state, seed_runs, tuning
from colab.models_zoo import build, count_params, head
from src.dataset import eval_transform, train_transform

REPO = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "epoch_seconds"]


class ManifestDataset(Dataset):
    """Images listed in a split manifest; the label is the parent folder name."""

    def __init__(self, paths: list[str], classes: list[str], root: Path, transform):
        self.paths = list(paths)
        self.root = Path(root)
        self.transform = transform
        self.class_to_id = {name: index for index, name in enumerate(classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(self.root / path) as image:
            return self.transform(image.convert("RGB")), self.class_to_id[Path(path).parts[-2]]


def _after_epoch(epoch: int) -> None:
    """Hook called right after checkpoint_last.pt is written (tests use it to simulate a disconnect)."""


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _loader(paths, classes, root, transform, batch_size, shuffle, device) -> DataLoader:
    workers = {"cuda": 2, "mps": int(os.environ.get("VNFOOD_WORKERS", 4))}.get(device.type, 0)
    return DataLoader(ManifestDataset(paths, classes, root, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def _write_history(run_dir: Path, history: list[dict]) -> None:
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _save_atomic(payload: dict, path: Path) -> None:
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    partial.replace(path)


@torch.inference_mode()
def _validate(model, loader, criterion, device, use_amp, max_batches):
    model.eval()
    loss = correct = total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss += criterion(logits.float(), labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
    return loss / total, correct / total


def _device(device: str | None) -> torch.device:
    return torch.device(device or session.choose_device(torch.cuda.is_available(), torch.backends.mps.is_available()))


def train(cfg: RunConfig, runs_dir: Path, manifest: dict, root: Path, device: str | None = None,
          max_batches: int | None = None) -> Path:
    run_dir = Path(runs_dir) / cfg.run_id
    state = run_state(run_dir)
    if state == "done":
        print(f"SKIP {cfg.run_id} (DONE)")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device)
    use_amp = device.type == "cuda"
    _seed_everything(cfg.seed)

    classes = manifest["classes"]
    train_loader = _loader(manifest["train"], classes, root, train_transform() if cfg.augment else eval_transform(),
                           cfg.batch_size, True, device)
    val_loader = _loader(manifest["validation"], classes, root, eval_transform(), cfg.batch_size, False, device)

    model = build(cfg.arch, len(classes), pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=1e-4)
    if cfg.arch == "cnn":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, stale, history, train_seconds = 1, -1.0, 0, [], 0.0
    if state == "resume":
        last = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=True)
        model.load_state_dict(last["model_state"]); optimizer.load_state_dict(last["optimizer_state"])
        scheduler.load_state_dict(last["scheduler_state"]); scaler.load_state_dict(last["scaler_state"])
        start_epoch, best, stale = last["epoch"] + 1, last["best_val_accuracy"], last["stale_epochs"]
        history, train_seconds = last["history"], last["train_seconds"]
        print(f"RESUME {cfg.run_id} from epoch {start_epoch}; best val={best:.4f}")
    else:
        version_file = REPO / "VERSION.txt"
        (run_dir / "config.json").write_text(json.dumps({
            **cfg.to_dict(),
            "version": version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else None,
            "device": session.device_label(device),
            "amp": use_amp,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_size": len(manifest["train"]), "validation_size": len(manifest["validation"]),
            "test_size": len(manifest["test"]),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"START {cfg.run_id} on {device} (amp={use_amp})")

    for epoch in range(start_epoch, cfg.epochs + 1):
        if stale >= cfg.patience:
            break
        started = time.perf_counter()
        model.train()
        if cfg.freeze_backbone:
            model.eval(); head(model, cfg.arch).train()  # keep pretrained BatchNorm statistics fixed
        loss_sum = correct = total = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
        val_loss, val_acc = _validate(model, val_loader, criterion, device, use_amp, max_batches)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()
        seconds = time.perf_counter() - started
        train_seconds += seconds
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
                        "val_loss": val_loss, "val_accuracy": val_acc, "lr": lr, "epoch_seconds": seconds})
        print(f"Epoch {epoch:02d}: train_acc={correct / total:.4f} val_acc={val_acc:.4f} ({seconds:.0f}s)")
        if val_acc > best:
            best, stale = val_acc, 0
            _save_atomic({"model_state": model.state_dict(), "classes": classes, "arch": cfg.arch,
                          "config": cfg.to_dict(), "best_val_accuracy": val_acc, "epoch": epoch,
                          "test_paths": manifest["test"]}, run_dir / "best.pt")
        else:
            stale += 1
        _save_atomic({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                      "epoch": epoch, "best_val_accuracy": best, "stale_epochs": stale, "history": history,
                      "train_seconds": train_seconds}, run_dir / "checkpoint_last.pt")
        _write_history(run_dir, history)
        _after_epoch(epoch)
        if stale >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; best val={best:.4f}")
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = _loader(manifest["test"], classes, root, eval_transform(), cfg.batch_size, False, device)
    metrics = evaluation.evaluate_model(model, test_loader, device, classes)
    metrics.update({"run_id": cfg.run_id, "params": count_params(model), "train_seconds": train_seconds,
                    "best_val_accuracy": best, "best_epoch": checkpoint["epoch"], "epochs_run": len(history),
                    "device": session.device_label(device),
                    **evaluation.measure_speed(model, device)})
    evaluation.write_artifacts(run_dir, metrics, history, classes)
    (run_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"), encoding="utf-8")
    (run_dir / "checkpoint_last.pt").unlink(missing_ok=True)
    print(f"DONE {cfg.run_id}: test_acc={metrics['test_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"top3={metrics['top3_accuracy']:.4f}")
    return run_dir


def benchmark(cfg: RunConfig, manifest: dict, root: Path, device: str | None = None, batches: int = 30) -> dict:
    """Time a few training steps on this machine and extrapolate to one run; writes nothing to runs/."""
    device = _device(device)
    use_amp = device.type == "cuda"
    _seed_everything(cfg.seed)
    loader = _loader(manifest["train"], manifest["classes"], root,
                     train_transform() if cfg.augment else eval_transform(), cfg.batch_size, True, device)
    model = build(cfg.arch, len(manifest["classes"]), pretrained=cfg.pretrained,
                  freeze_backbone=cfg.freeze_backbone).to(device).train()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup, timed, started = 3, 0, None
    while timed < batches:
        for images, labels in loader:
            if timed == 0 and warmup == 0:
                evaluation.synchronize(device); started = time.perf_counter()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if warmup:
                warmup -= 1
                continue
            timed += 1
            if timed >= batches:
                break
    evaluation.synchronize(device)
    per_batch = (time.perf_counter() - started) / batches
    steps = len(manifest["train"]) / cfg.batch_size + 0.4 * len(manifest["validation"]) / cfg.batch_size
    epoch_minutes = per_batch * steps / 60
    return {"device": session.device_label(device), "seconds_per_batch": per_batch,
            "est_epoch_minutes": epoch_minutes, "est_run_hours": epoch_minutes * cfg.epochs / 60}


def _base_from_pin(runs_dir: Path) -> RunConfig:
    pin = runs_dir / "tuning_base.json"
    if not pin.is_file():
        raise SystemExit("Chưa chốt mô hình base: chạy cell 'pin_tuning_base' trong notebook 01 trước.")
    return RunConfig(**json.loads(pin.read_text(encoding="utf-8"))["config"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--exp", choices=list(EXPERIMENTS))
    which.add_argument("--tune", choices=["T1", "T2", "T3", "T4a", "T4b", "T5"])
    which.add_argument("--seed-run", choices=["S123", "S2024"])
    which.add_argument("--control", choices=["C1"])
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path, default=REPO / "outputs" / "clean_split_manifest.json")
    args = parser.parse_args()
    if args.exp:
        cfg = EXPERIMENTS[args.exp]
    elif args.tune:
        cfg = tuning(_base_from_pin(args.runs_dir))[args.tune]
    elif args.control:
        cfg = control(_base_from_pin(args.runs_dir))[args.control]
    else:
        cfg = seed_runs(_base_from_pin(args.runs_dir))[args.seed_run]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train(cfg, args.runs_dir, manifest, args.root)


if __name__ == "__main__":
    main()
