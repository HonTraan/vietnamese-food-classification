import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from colab import train_arch
from colab.experiments import EXPERIMENTS


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / "repo"
    manifest = {"classes": ["A", "B"], "train": [], "validation": [], "test": []}
    rng = np.random.default_rng(0)
    for name in manifest["classes"]:
        folder = root / "data" / "raw" / name
        folder.mkdir(parents=True)
        for index in range(6):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(folder / f"{index}.jpg")
            split = ("train", "train", "validation", "validation", "test", "test")[index]
            manifest[split].append(f"data/raw/{name}/{index}.jpg")
    return root, manifest, tmp_path / "runs"


def _epochs(run_dir):
    with (run_dir / "history.csv").open() as file:
        return [int(row["epoch"]) for row in csv.DictReader(file)]


def test_train_cnn_creates_artifacts(tiny_data):
    root, manifest, runs = tiny_data
    cfg = replace(EXPERIMENTS["E1"], epochs=1, batch_size=2)
    run_dir = train_arch.train(cfg, runs, manifest, root, device="cpu")
    assert run_dir == runs / "E1_cnn"
    for name in ("DONE", "best.pt", "config.json", "metrics.json", "test_predictions.csv",
                 "learning_curves.png", "confusion_matrix.png"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "checkpoint_last.pt").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("test_accuracy", "macro_f1", "top3_accuracy", "params", "train_seconds",
                "best_val_accuracy", "cpu_ms_per_image"):
        assert key in metrics, key
    assert _epochs(run_dir) == [1]


def test_resume_continues_from_last_epoch(tiny_data, monkeypatch):
    root, manifest, runs = tiny_data
    cfg = replace(EXPERIMENTS["E1"], epochs=2, batch_size=2)

    def interrupt(epoch):
        if epoch == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(train_arch, "_after_epoch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_arch.train(cfg, runs, manifest, root, device="cpu", max_batches=1)
    run_dir = runs / cfg.run_id
    assert (run_dir / "checkpoint_last.pt").is_file()
    assert not (run_dir / "DONE").exists()

    monkeypatch.setattr(train_arch, "_after_epoch", lambda epoch: None)
    train_arch.train(cfg, runs, manifest, root, device="cpu", max_batches=1)
    assert _epochs(run_dir) == [1, 2]
    assert (run_dir / "DONE").is_file()


def test_skip_when_done(tiny_data):
    root, manifest, runs = tiny_data
    cfg = replace(EXPERIMENTS["E1"], epochs=1, batch_size=2)
    run_dir = train_arch.train(cfg, runs, manifest, root, device="cpu", max_batches=1)
    before = (run_dir / "metrics.json").stat().st_mtime_ns
    train_arch.train(cfg, runs, manifest, root, device="cpu", max_batches=1)
    assert (run_dir / "metrics.json").stat().st_mtime_ns == before


def test_manifest_dataset_labels(tiny_data):
    root, manifest, _ = tiny_data
    from src.dataset import eval_transform
    dataset = train_arch.ManifestDataset(manifest["test"], manifest["classes"], root, eval_transform())
    labels = [dataset[i][1] for i in range(len(dataset))]
    assert labels == [0, 0, 1, 1]


def test_calibrate_run_writes_temperature(tiny_data):
    import torch
    from colab import analyze
    root, manifest, runs = tiny_data
    run_dir = train_arch.train(replace(EXPERIMENTS["E1"], epochs=1, batch_size=2), runs, manifest, root, device="cpu")
    report = analyze.calibrate_run(run_dir, manifest, root, "cpu")
    assert report["fitted_on"] == "validation" and report["temperature"] >= 0.05
    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    assert {"model_state", "classes", "arch", "temperature"} <= set(checkpoint)
    assert analyze.misclassified_gallery(run_dir, root).is_file()
