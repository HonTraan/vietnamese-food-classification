import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_clean_split_is_disjoint():
    manifest_path = ROOT / "outputs" / "clean_split_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("clean split manifest not generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = set(manifest["train"])
    validation = set(manifest["validation"])
    test = set(manifest["test"])
    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)
    assert len(manifest["classes"]) == 30


def test_all_split_paths_exist():
    manifest_path = ROOT / "outputs" / "clean_split_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("clean split manifest not generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [
        path for split in ("train", "validation", "test")
        for path in manifest[split]
        if not (ROOT / path).is_file()
    ]
    assert not missing


def test_calibration_temperature_is_valid():
    report_path = ROOT / "outputs" / "calibration.json"
    if not report_path.is_file():
        pytest.skip("calibration report not generated")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    temperature = report["temperature"]
    assert temperature >= 0.05
    assert report["test_after"]["ece"] < report["test_before"]["ece"]
    assert report["test_after"]["nll"] < report["test_before"]["nll"]


def test_checkpoint_exists_and_has_classes():
    checkpoint_path = ROOT / "models" / "best_model.pt"
    if not checkpoint_path.is_file():
        pytest.skip("checkpoint not generated")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state" in checkpoint
    assert len(checkpoint["classes"]) == 30


def test_manual_visual_review_has_no_pending_rows():
    review_path = ROOT / "outputs" / "weak_class_manual_review.csv"
    if not review_path.is_file():
        pytest.skip("manual review manifest not generated")
    import csv
    with review_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert all(row["manual_label_status"] != "needs_visual_review" for row in rows)


def test_horizontal_metric_csv_files_exist():
    for relative_path in (
        "outputs/class_metrics.csv",
        "outputs/confusion_matrix.csv",
        "outputs/cnn_class_metrics.csv",
        "outputs/cnn_confusion_matrix.csv",
        "outputs/calibration_summary.csv",
        "outputs/calibration_reliability.csv",
        "outputs/training_history.csv",
        "outputs/cnn_history.csv",
        "outputs/multi_seed_results.csv",
        "outputs/eda_summary.csv",
    ):
        assert (ROOT / relative_path).is_file(), relative_path
