"""Download and load the five-class Vietnamese-food dataset."""
from __future__ import annotations

import re
import shutil
import unicodedata
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# A small five-class subset kept only for quick experiments. The project default
# is all 30 classes from the source dataset.
FIVE_CLASS_NAMES = ("Pho", "Bun bo Hue", "Banh mi", "Com tam", "Goi cuon")
IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def download_dataset(max_per_class: int = 0, force: bool = False,
                     dataset_name: str = "TuyenTrungLe/vietnamese_food_images",
                     all_classes: bool = True, output_dir: Path | None = None,
                     class_names: tuple[str, ...] | None = None) -> dict[str, int]:
    """Save every class, the five-class subset, or explicitly requested classes."""
    destination_root = output_dir or RAW_DIR
    if destination_root.exists() and any(destination_root.iterdir()):
        if not force:
            raise FileExistsError(f"{destination_root} is not empty; pass force=True to replace it.")
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    splits = load_dataset(dataset_name)
    merged = concatenate_datasets(list(splits.values()))
    label_names = merged.features["label"].names
    if class_names is not None:
        name_to_id = {_normalize(name): index for index, name in enumerate(label_names)}
        selected = {name_to_id[_normalize(name)]: name for name in class_names if _normalize(name) in name_to_id}
        missing = set(class_names) - set(selected.values())
        if missing:
            raise ValueError(f"Missing classes {sorted(missing)}. Available: {label_names}")
    elif all_classes:
        selected = dict(enumerate(label_names))
    else:
        name_to_id = {_normalize(name): index for index, name in enumerate(label_names)}
        selected = {name_to_id[_normalize(name)]: name for name in FIVE_CLASS_NAMES if _normalize(name) in name_to_id}
        missing = set(FIVE_CLASS_NAMES) - set(selected.values())
        if missing:
            raise ValueError(f"Missing classes {sorted(missing)}. Available: {label_names}")
    counts = {name: 0 for name in selected.values()}
    for row in tqdm(merged, desc="Saving selected images"):
        class_name = selected.get(row["label"])
        if class_name is None or (max_per_class and counts[class_name] >= max_per_class):
            continue
        folder = destination_root / class_name; folder.mkdir(exist_ok=True)
        row["image"].convert("RGB").save(folder / f"{counts[class_name]:05d}.jpg", quality=95)
        counts[class_name] += 1
    return counts


def train_transform():
    return transforms.Compose([transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def eval_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class IndexedImageFolder(Dataset):
    def __init__(self, indices, transform):
        self.base = ImageFolder(RAW_DIR); self.indices = list(indices); self.transform = transform
        self.classes = self.base.classes

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        path, label = self.base.samples[self.indices[index]]
        return self.transform(self.base.loader(path)), label


class ManifestImageFolder(Dataset):
    """ImageFolder-like dataset loaded from a cleaned split manifest."""

    def __init__(self, paths, transform):
        self.paths = [ROOT / path for path in paths]
        self.transform = transform
        manifest = json.loads((ROOT / "outputs" / "clean_split_manifest.json").read_text(encoding="utf-8"))
        self.classes = manifest["classes"]
        self.class_to_id = {name: index for index, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.class_to_id[path.relative_to(RAW_DIR).parts[0]]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download all 30 Vietnamese-food classes by default.")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="0 means all images; use 600 for a quick experiment.")
    parser.add_argument("--force", action="store_true", help="Replace existing data/raw.")
    class_mode = parser.add_mutually_exclusive_group()
    class_mode.add_argument("--all-classes", dest="all_classes", action="store_true",
                            help="Download all 30 source classes (default).")
    class_mode.add_argument("--five-classes", dest="all_classes", action="store_false",
                            help="Download only the five-class subset for a quick experiment.")
    parser.set_defaults(all_classes=True)
    parser.add_argument("--classes", nargs="+", metavar="CLASS",
                        help="Specific source labels to download. Quote multi-word labels; overrides class mode.")
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR,
                        help="Destination folder; default is data/raw.")
    options = parser.parse_args()
    print("Saved:", download_dataset(options.max_per_class, options.force,
                                      all_classes=options.all_classes, output_dir=options.output_dir,
                                      class_names=tuple(options.classes) if options.classes else None))
