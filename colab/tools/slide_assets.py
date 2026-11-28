"""Tables and figures for the slides, computed from the real models and the finished runs.

Run: python -m colab.tools.slide_assets [--runs colab/results] [--out colab/results/_slides]
"""
from __future__ import annotations

import torch
from torch import nn

from colab.models_zoo import build, count_params, head

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
             "in_shape": traced[name][0], "out_shape": traced[name][1], "params": count_params(module)}
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
