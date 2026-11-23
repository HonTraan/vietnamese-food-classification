"""Paths and per-session helpers shared by every Colab notebook."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/DL/vnfood_colab")
REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = DRIVE_ROOT / "runs"
DATA_DIR = DRIVE_ROOT / "data"
MANIFEST_NAME = "clean_split_manifest.json"
SPLITS = ("train", "validation", "test")


def read_version(path: Path) -> dict[str, str]:
    info = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            info[key.strip()] = value.strip()
    return info


CODE_PATTERNS = ("app.py", "src/**/*.py", "colab/**/*.py", "colab/**/*.sh")


def code_hash(root: Path) -> str:
    """sha256 over every code file (path + bytes); VERSION.txt alone cannot prove Drive finished syncing."""
    import hashlib
    root = Path(root)
    files = {path for pattern in CODE_PATTERNS for path in root.glob(pattern)
             if path.is_file() and "__pycache__" not in path.parts and "results" not in path.relative_to(root).parts[:2]}
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def compare_manifests(a: dict, b: dict) -> dict:
    """Compare two split manifests by the set of paths in each split."""
    only_in_a, only_in_b = {}, {}
    for split in SPLITS:
        left, right = set(a.get(split, [])), set(b.get(split, []))
        only_in_a[split] = len(left - right)
        only_in_b[split] = len(right - left)
    same = not any(only_in_a.values()) and not any(only_in_b.values()) and a.get("classes") == b.get("classes")
    return {
        "same": same,
        "counts_a": {split: len(a.get(split, [])) for split in SPLITS},
        "counts_b": {split: len(b.get(split, [])) for split in SPLITS},
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
    }


def check_gpu() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Chưa bật GPU: Runtime > Change runtime type > T4 GPU")
    return torch.cuda.get_device_name(0)


def save_data(repo: Path = REPO, data_dir: Path = DATA_DIR) -> None:
    """Store the cleaned data/raw tree and its manifest on Drive (one-time, notebook 00)."""
    raw = repo / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    partial = data_dir / "raw.zip.partial"
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(repo / "data").as_posix())
    partial.replace(data_dir / "raw.zip")
    shutil.copy2(repo / "outputs" / MANIFEST_NAME, data_dir / MANIFEST_NAME)


def restore_data(repo: Path = REPO, data_dir: Path = DATA_DIR) -> bool:
    """Unpack Drive data into the local repo copy; False when notebook 00 has not run yet."""
    archive_path = data_dir / "raw.zip"
    if not archive_path.is_file():
        return False
    target = repo / "data"
    if not (target / "raw").is_dir():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
    (repo / "outputs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / MANIFEST_NAME, repo / "outputs" / MANIFEST_NAME)
    return True
