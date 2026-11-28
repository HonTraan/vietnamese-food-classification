from colab.tools import slide_assets as sa


def test_resnet50_param_counts():
    s = sa.model_summary("resnet50")
    assert s["total_params"] == 23_569_502
    assert s["head_params"] == 61_470
    assert s["frozen_trainable_params"] == 61_470
    assert s["final_feature_shape"] == "2048×7×7"


def test_resnet50_has_50_weight_layers():
    assert sa.weight_layer_count("resnet50") == 50


def test_resnet_stages():
    rows = {r["stage"]: r for r in sa.resnet_stage_summary()}
    assert [rows[f"layer{i}"]["blocks"] for i in range(1, 5)] == [3, 4, 6, 3]
    assert rows["conv1"]["out_shape"] == "64×112×112"
    assert rows["layer1"]["out_shape"] == "256×56×56"
    assert rows["layer4"]["out_shape"] == "2048×7×7"
    assert rows["fc"]["out_shape"] == "30"


def test_cnn_block_shapes():
    pools = [r for r in sa.layer_table("cnn") if r["type"] in ("MaxPool2d", "AdaptiveAvgPool2d")]
    assert [r["out_shape"] for r in pools] == ["32×112×112", "64×56×56", "128×28×28", "256×1×1"]
    assert sa.layer_table("cnn")[-1]["out_shape"] == "30"


def test_layer_table_params_sum_to_total():
    for arch in ("cnn", "mobilenet_v3_large", "efficientnet_b0"):
        leaf = [r for r in sa.layer_table(arch) if not r["name"].startswith("layer")]
        assert sum(r["params"] for r in leaf) == sa.model_summary(arch)["total_params"]


import csv
import json

from colab import compare
from colab.experiments import EXPERIMENTS

CLASSES = ["Pho", "Banh mi"]


def _fake_run(runs, run_id, arch, done=True):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    exp = run_id.split("_")[0]
    config = {**EXPERIMENTS.get(exp, EXPERIMENTS["E2"]).to_dict(), "run_id": run_id, "arch": arch, "device": "cpu"}
    (run_dir / "config.json").write_text(json.dumps(config))
    report = {name: {"precision": 0.8, "recall": 0.7, "f1-score": f1, "support": 10} for name, f1 in zip(CLASSES, (0.9, 0.6))}
    report.update({"accuracy": 0.8, "macro avg": {"f1-score": 0.75}, "weighted avg": {"f1-score": 0.75}})
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_val_accuracy": 0.8, "test_accuracy": 0.8, "macro_f1": 0.75, "top3_accuracy": 0.9, "params": 1000,
        "train_seconds": 60.0, "gpu_images_per_sec": None, "cpu_ms_per_image": 10.0, "best_epoch": 2,
        "report": report, "confusion_matrix": [[8, 2], [3, 7]]}))
    with (run_dir / "history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "epoch_seconds"])
        for epoch, (tr, va) in enumerate([(0.5, 0.5), (0.9, 0.8), (0.95, 0.78)], 1):
            writer.writerow([epoch, 1 - tr, tr, 1 - va, va, 3e-4, 10])
    if done:
        (run_dir / "DONE").write_text("")
    return run_dir


def _fake_runs(tmp_path):
    runs = tmp_path / "runs"
    for run_id, arch in [("E1_cnn", "cnn"), ("E2_resnet50", "resnet50"), ("E3_mobilenet_v3_large", "mobilenet_v3_large"),
                         ("E4_efficientnet_b0", "efficientnet_b0")]:
        _fake_run(runs, run_id, arch)
    return runs


def test_grid_skips_unfinished_run(tmp_path):
    runs = tmp_path / "runs"
    _fake_run(runs, "E2_resnet50", "resnet50")
    _fake_run(runs, "C1_resnet50_m4", "resnet50")
    _fake_run(runs, "T5_resnet50_ls0.1", "resnet50", done=False)
    for metric in ("accuracy", "loss"):
        assert sa.learning_curves_grid(compare.collect(runs), tmp_path / f"{metric}.png", metric).is_file()
    assert sa.grid_run_ids(compare.collect(runs)) == ["E2_resnet50", "C1_resnet50_m4"]


def test_row_normalized_confusion():
    assert sa.row_normalized([[2, 2], [0, 0]]) == [[0.5, 0.5], [0.0, 0.0]]


def test_hyperparams_scheduler(tmp_path):
    rows = {r["run_id"]: r for r in sa.hyperparams_table(compare.collect(_fake_runs(tmp_path)))}
    assert rows["E1_cnn"]["scheduler"] == "ReduceLROnPlateau(factor=0.5, patience=1)"
    assert rows["E2_resnet50"]["scheduler"] == "CosineAnnealingLR(T_max=epochs)"
    assert rows["E2_resnet50"]["optimizer"] == "AdamW" and rows["E2_resnet50"]["weight_decay"] == 1e-4
    assert rows["E1_cnn"]["epochs"] == 15 and rows["E2_resnet50"]["batch_size"] == 32


def test_main_writes_only_into_out_dir(tmp_path):
    runs = _fake_runs(tmp_path)
    before = sorted(p.relative_to(runs) for p in runs.rglob("*"))
    out = tmp_path / "out"
    sa.main(runs, out)
    first = (out / "model_summary.csv").read_text()
    sa.main(runs, out)
    assert sorted(p.relative_to(runs) for p in runs.rglob("*")) == before
    assert len(list(out.iterdir())) == 12
    assert (out / "learning_curves_grid_loss.png").is_file()
    assert (out / "model_summary.csv").read_text() == first


def test_resnet_stage_params_sum_to_total():
    assert sum(r["params"] for r in sa.resnet_stage_summary()) == 23_569_502
