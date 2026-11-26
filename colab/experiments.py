"""Registry of the experiments (E* architectures, C1 local control, T* tuning, S* extra seeds)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    arch: str
    pretrained: bool = True
    freeze_backbone: bool = False
    augment: bool = True
    lr: float = 3e-4
    label_smoothing: float = 0.0
    epochs: int = 12
    batch_size: int = 32
    patience: int = 4
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def _exp(key: str, arch: str, **overrides) -> RunConfig:
    return RunConfig(run_id=f"{key}_{arch}", arch=arch, **overrides)


EXPERIMENTS: dict[str, RunConfig] = {
    "E1": _exp("E1", "cnn", pretrained=False, lr=1e-3, label_smoothing=0.05, epochs=15),
    "E2": _exp("E2", "resnet50"),
    "E3": _exp("E3", "mobilenet_v3_large"),
    "E4": _exp("E4", "efficientnet_b0"),
    "E5": _exp("E5", "convnext_tiny"),
}

PRIORITY = ["E1", "E4", "E2", "E3", "C1", "T1", "T3", "T2", "T4a", "T4b", "T5", "E5"]


def tuning(base: RunConfig) -> dict[str, RunConfig]:
    arch = base.arch
    return {
        "T1": replace(base, run_id=f"T1_{arch}_frozen", freeze_backbone=True),
        "T2": replace(base, run_id=f"T2_{arch}_scratch", pretrained=False),
        "T3": replace(base, run_id=f"T3_{arch}_noaug", augment=False),
        "T4a": replace(base, run_id=f"T4a_{arch}_lr1e-3", lr=1e-3),
        "T4b": replace(base, run_id=f"T4b_{arch}_lr1e-4", lr=1e-4),
        "T5": replace(base, run_id=f"T5_{arch}_ls0.1", label_smoothing=0.1),
    }


def control(base: RunConfig) -> dict[str, RunConfig]:
    """Base recipe re-run on the Mac (mps, fp32): T* run there are compared against it, not the T4 base."""
    return {"C1": replace(base, run_id=f"C1_{base.arch}_m4")}


def seed_runs(base: RunConfig, seeds: tuple[int, ...] = (123, 2024)) -> dict[str, RunConfig]:
    return {f"S{seed}": replace(base, run_id=f"S{seed}_{base.arch}", seed=seed) for seed in seeds}


def run_state(run_dir: Path) -> Literal["new", "resume", "done"]:
    run_dir = Path(run_dir)
    if (run_dir / "DONE").is_file():
        return "done"
    if (run_dir / "checkpoint_last.pt").is_file():
        return "resume"
    return "new"
