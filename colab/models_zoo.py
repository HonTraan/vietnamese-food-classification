"""Build the architectures compared on Colab with a fresh 30-class head."""
from __future__ import annotations

from torch import nn
from torchvision import models

ARCHS = ("cnn", "resnet50", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny")


def build(arch: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHS}")
    if arch == "cnn":
        if pretrained or freeze_backbone:
            raise ValueError("cnn is trained from scratch; pretrained/freeze_backbone are not supported")
        from src.cnn_baseline import CNNBaseline
        return CNNBaseline(num_classes)

    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        # Same head as src/train.build_model so checkpoints work with app.py.
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, num_classes))
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if freeze_backbone:
        head_ids = {id(p) for p in head(model, arch).parameters()}
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in head_ids
    return model


def head(model: nn.Module, arch: str) -> nn.Module:
    return model.fc if arch == "resnet50" else model.classifier


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
