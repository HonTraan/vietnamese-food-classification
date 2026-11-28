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
