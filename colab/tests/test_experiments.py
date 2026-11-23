from colab import experiments
from colab.experiments import EXPERIMENTS, run_state, seed_runs, tuning


def test_experiment_ids():
    assert list(EXPERIMENTS) == ["E1", "E2", "E3", "E4", "E5"]
    assert EXPERIMENTS["E1"].run_id == "E1_cnn"
    assert EXPERIMENTS["E4"].run_id == "E4_efficientnet_b0"
    cnn = EXPERIMENTS["E1"]
    assert (cnn.pretrained, cnn.lr, cnn.label_smoothing, cnn.epochs) == (False, 1e-3, 0.05, 15)
    assert EXPERIMENTS["E2"].lr == 3e-4 and EXPERIMENTS["E2"].epochs == 12


def test_tuning_differs_in_one_field():
    base = EXPERIMENTS["E4"]
    variants = tuning(base)
    expected = {
        "T1": ("freeze_backbone", True, "T1_efficientnet_b0_frozen"),
        "T2": ("pretrained", False, "T2_efficientnet_b0_scratch"),
        "T3": ("augment", False, "T3_efficientnet_b0_noaug"),
        "T4a": ("lr", 1e-3, "T4a_efficientnet_b0_lr1e-3"),
        "T4b": ("lr", 1e-4, "T4b_efficientnet_b0_lr1e-4"),
        "T5": ("label_smoothing", 0.1, "T5_efficientnet_b0_ls0.1"),
    }
    assert list(variants) == list(expected)
    base_fields = {k: v for k, v in base.to_dict().items() if k != "run_id"}
    for key, (field, value, run_id) in expected.items():
        fields = {k: v for k, v in variants[key].to_dict().items() if k != "run_id"}
        changed = {k for k in fields if fields[k] != base_fields[k]}
        assert changed == {field}, key
        assert fields[field] == value
        assert variants[key].run_id == run_id


def test_seed_runs():
    runs = seed_runs(EXPERIMENTS["E4"])
    assert list(runs) == ["S123", "S2024"]
    assert runs["S123"].seed == 123 and runs["S123"].run_id == "S123_efficientnet_b0"


def test_run_state(tmp_path):
    assert run_state(tmp_path / "missing") == "new"
    tmp_path.joinpath("checkpoint_last.pt").write_bytes(b"x")
    assert run_state(tmp_path) == "resume"
    tmp_path.joinpath("DONE").write_text("")
    assert run_state(tmp_path) == "done"


def test_priority_covers_registry():
    assert experiments.PRIORITY[:2] == ["E1", "E4"]
    assert set(experiments.PRIORITY) == set(EXPERIMENTS) | {"T1", "T2", "T3", "T4a", "T4b", "T5"}
