import csv
import json

from colab import compare
from colab.experiments import EXPERIMENTS


def _run(runs, run_id, val, test, macro=0.5, done=True, arch=None, device=None):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    exp = run_id.split("_")[0]
    config = EXPERIMENTS[exp].to_dict() if exp in EXPERIMENTS else {**EXPERIMENTS["E4"].to_dict(), "run_id": run_id}
    if arch:
        config["arch"] = arch
    if device:
        config["device"] = device
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_val_accuracy": val, "test_accuracy": test, "macro_f1": macro, "top3_accuracy": 0.9,
        "params": 1000, "train_seconds": 60.0, "gpu_images_per_sec": None, "cpu_ms_per_image": 10.0}))
    if done:
        (run_dir / "DONE").write_text("")


def test_collect_skips_unfinished(tmp_path):
    _run(tmp_path, "E1_cnn", 0.6, 0.6)
    _run(tmp_path, "E4_efficientnet_b0", 0.9, 0.9, done=False)
    rows = compare.collect(tmp_path)
    assert [row["run_id"] for row in rows] == ["E1_cnn"]
    assert rows[0]["exp"] == "E1" and rows[0]["arch"] == "cnn"


def test_pick_best_uses_validation_not_test(tmp_path):
    _run(tmp_path, "E2_resnet50", 0.90, 0.80)
    _run(tmp_path, "E3_mobilenet_v3_large", 0.85, 0.95)
    assert compare.pick_best(compare.collect(tmp_path))["run_id"] == "E2_resnet50"


def test_pick_best_prefix(tmp_path):
    _run(tmp_path, "E4_efficientnet_b0", 0.88, 0.88)
    _run(tmp_path, "T1_efficientnet_b0_frozen", 0.95, 0.95)
    rows = compare.collect(tmp_path)
    assert compare.pick_best(rows)["run_id"] == "T1_efficientnet_b0_frozen"
    assert compare.pick_best(rows, prefix="E")["run_id"] == "E4_efficientnet_b0"


def _four_done(runs, e4_val=0.9):
    _run(runs, "E1_cnn", 0.6, 0.6)
    _run(runs, "E2_resnet50", 0.85, 0.85)
    _run(runs, "E3_mobilenet_v3_large", 0.86, 0.86)
    _run(runs, "E4_efficientnet_b0", e4_val, 0.9)


def test_tuning_base_is_pinned(tmp_path):
    _four_done(tmp_path)
    base = compare.pin_tuning_base(tmp_path, compare.collect(tmp_path))
    assert base.run_id == "E4_efficientnet_b0"
    _run(tmp_path, "E5_convnext_tiny", 0.99, 0.99)
    again = compare.pin_tuning_base(tmp_path, compare.collect(tmp_path))
    assert again == base
    assert json.loads((tmp_path / "tuning_base.json").read_text())["base_exp"] == "E4"


def test_pin_requires_E1_to_E4_done(tmp_path):
    _run(tmp_path, "E1_cnn", 0.6, 0.6)
    try:
        compare.pin_tuning_base(tmp_path, compare.collect(tmp_path))
    except RuntimeError as error:
        assert "E4" in str(error) and "E2" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
    assert not (tmp_path / "tuning_base.json").exists()


def test_write_comparison_creates_csv(tmp_path):
    _run(tmp_path / "runs", "E1_cnn", 0.6, 0.6)
    _run(tmp_path / "runs", "E4_efficientnet_b0", 0.9, 0.9)
    compare.write_comparison(compare.collect(tmp_path / "runs"), tmp_path / "out", plots=False)
    with (tmp_path / "out" / "comparison.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [row["run_id"] for row in rows] == ["E1_cnn", "E4_efficientnet_b0"]
    assert list(rows[0])[:3] == ["run_id", "exp", "arch"]


def test_collect_has_device(tmp_path):
    _run(tmp_path, "E1_cnn", 0.6, 0.6, device="Tesla T4")
    row = compare.collect(tmp_path)[0]
    assert row["device"] == "Tesla T4"
    assert compare.FIELDS.index("device") == compare.FIELDS.index("seed") + 1


def test_tuning_reference_prefers_control(tmp_path):
    _four_done(tmp_path, e4_val=0.8)
    compare.pin_tuning_base(tmp_path, compare.collect(tmp_path))
    _run(tmp_path, "C1_resnet50_m4", 0.84, 0.84, arch="resnet50", device="mps (Apple M4 Pro)")
    _run(tmp_path, "T1_resnet50_frozen", 0.7, 0.7, arch="resnet50")
    assert compare.tuning_reference(compare.collect(tmp_path), tmp_path)["run_id"] == "C1_resnet50_m4"


def test_tuning_reference_falls_back_to_base(tmp_path):
    _four_done(tmp_path, e4_val=0.8)
    rows = compare.collect(tmp_path)
    assert compare.tuning_reference(rows, tmp_path) is None
    base = compare.pin_tuning_base(tmp_path, rows)
    _run(tmp_path, "C1_resnet50_m4", 0.84, 0.84, arch="resnet50", done=False)
    assert compare.tuning_reference(compare.collect(tmp_path), tmp_path)["run_id"] == base.run_id


def _tuned(runs):
    _four_done(runs, e4_val=0.8)
    base = compare.pin_tuning_base(runs, compare.collect(runs))
    _run(runs, "C1_resnet50_m4", 0.84, 0.83, arch="resnet50", device="mps (Apple M4 Pro)")
    _run(runs, "T1_resnet50_frozen", 0.70, 0.71, arch="resnet50", device="mps (Apple M4 Pro)")
    _run(runs, "T3_resnet50_noaug", 0.80, 0.79, arch="resnet50", device="Tesla T4")
    return base


def test_tuning_deltas_use_same_device_reference(tmp_path):
    base = _tuned(tmp_path)
    deltas = {row["run_id"]: row for row in compare.tuning_deltas(compare.collect(tmp_path), tmp_path)}
    assert deltas["T1_resnet50_frozen"]["reference"] == "C1_resnet50_m4"
    assert abs(deltas["T1_resnet50_frozen"]["delta_test"] - (0.71 - 0.83)) < 1e-9
    assert deltas["T3_resnet50_noaug"]["reference"] == base.run_id


def test_hardware_effect(tmp_path):
    base = _tuned(tmp_path)
    effect = compare.hardware_effect(compare.collect(tmp_path), tmp_path)
    assert effect["base"] == base.run_id and effect["control"] == "C1_resnet50_m4"
    base_row = next(r for r in compare.collect(tmp_path) if r["run_id"] == base.run_id)
    assert abs(effect["delta_test"] - (base_row["test_accuracy"] - 0.83)) < 1e-9
