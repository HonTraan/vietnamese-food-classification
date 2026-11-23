import pytest
import torch

from colab import models_zoo

PRETRAINABLE = [arch for arch in models_zoo.ARCHS if arch != "cnn"]


@pytest.mark.parametrize("arch", models_zoo.ARCHS)
def test_output_shape(arch):
    model = models_zoo.build(arch, 30, pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 30)


@pytest.mark.parametrize("arch", PRETRAINABLE)
def test_freeze_only_head_trainable(arch):
    model = models_zoo.build(arch, 30, pretrained=False, freeze_backbone=True)
    head_ids = {id(p) for p in models_zoo.head(model, arch).parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert all(id(p) in head_ids for p in trainable)


def test_cnn_rejects_pretrained():
    with pytest.raises(ValueError):
        models_zoo.build("cnn", 30, pretrained=True)
    with pytest.raises(ValueError):
        models_zoo.build("cnn", 30, pretrained=False, freeze_backbone=True)


def test_unknown_arch():
    with pytest.raises(ValueError):
        models_zoo.build("vgg16", 30, pretrained=False)


def test_count_params_positive():
    assert models_zoo.count_params(models_zoo.build("cnn", 30, pretrained=False)) > 0


def test_efficientnet_checkpoint_loads_in_src_build_model():
    from src.train import build_model
    state = models_zoo.build("efficientnet_b0", 30, pretrained=False).state_dict()
    build_model(30, pretrained=False).load_state_dict(state)
