"""Paths and per-session helpers shared by every Colab notebook."""
from __future__ import annotations

import os
import shutil
import zipfile
from collections.abc import Mapping
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


def drive_root(on_colab: bool, env: Mapping[str, str] = os.environ, home: Path = Path.home()) -> Path:
    """vnfood_colab on Drive: the Colab mount, VNFOOD_DRIVE_ROOT, or the single Drive for Desktop mirror."""
    if on_colab:
        return DRIVE_ROOT
    if env.get("VNFOOD_DRIVE_ROOT"):
        return Path(env["VNFOOD_DRIVE_ROOT"])
    found = sorted(Path(home).glob("Library/CloudStorage/GoogleDrive-*/My Drive/DL/vnfood_colab"))
    if len(found) != 1:
        raise RuntimeError(f"Cần đúng một thư mục vnfood_colab của Drive for Desktop (tìm thấy {len(found)}); "
                           "bật Drive for Desktop hoặc đặt VNFOOD_DRIVE_ROOT.")
    return found[0]


def choose_device(cuda: bool, mps: bool) -> str:
    return "cuda" if cuda else "mps" if mps else "cpu"


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


def device_label(device) -> str:
    """Human-readable accelerator name stored in config.json / metrics.json."""
    import subprocess
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        import torch
        return torch.cuda.get_device_name(0)
    if kind == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "mps"
        return f"mps ({chip})" if chip else "mps"
    return "cpu"


def check_gpu() -> str:
    import torch
    device = choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())
    if device == "cpu":
        raise RuntimeError("Chưa bật GPU: Runtime > Change runtime type > T4 GPU")
    return device_label(device)


def _torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())


def start(repo: Path, on_colab: bool) -> dict:
    """Second half of every notebook's cell 0 (the first half mounts Drive / finds the repo)."""
    import subprocess
    import sys
    repo = Path(repo)
    root = drive_root(on_colab)
    os.chdir(repo)
    if on_colab:
        print(Path(root / "code" / "VERSION.txt").read_text(encoding="utf-8"))
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "datasets", "streamlit", "pytest"], check=True)
        expected = read_version(root / "code" / "VERSION.txt")["content_hash"]
        actual = code_hash(repo)
        assert actual == expected, (f"Code chưa đồng bộ xong (hash {actual} != {expected}). Đợi 1-2 phút rồi chạy "
                                    "lại cell này; nếu vẫn lệch: Runtime > Restart session rồi chạy lại.")
        print("Code khớp Drive, hash:", actual)
        print("GPU:", check_gpu())
        manifest = repo / "outputs" / MANIFEST_NAME
    else:
        manifest = repo / "data" / MANIFEST_NAME
    device = _torch_device()
    print("Runs:", root / "runs")
    print("Thiết bị:", device_label(device) if device != "cpu" else "cpu (không có GPU!)")
    return {"on_colab": on_colab, "repo": repo, "runs": root / "runs", "data": root / "data",
            "manifest": manifest, "device": device}


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


def restore_data(repo: Path = REPO, data_dir: Path = DATA_DIR, manifest_dst: Path | None = None) -> bool:
    """Unpack Drive data into the repo copy; False when notebook 00 has not run yet.

    Extracts into data/.extract first so an interrupted unpack is never mistaken for a complete data/raw.
    """
    archive_path = data_dir / "raw.zip"
    if not archive_path.is_file():
        return False
    target = repo / "data"
    if not (target / "raw").is_dir():
        staging = target / ".extract"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging)
        (staging / "raw").replace(target / "raw")
        shutil.rmtree(staging)
    manifest_dst = Path(manifest_dst) if manifest_dst else repo / "outputs" / MANIFEST_NAME
    manifest_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / MANIFEST_NAME, manifest_dst)
    return True
