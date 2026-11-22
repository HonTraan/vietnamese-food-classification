"""Create dataset summary, class-distribution and sample-image figures."""
from __future__ import annotations

import json
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder

try:
    from .dataset import RAW_DIR
except ImportError:
    from dataset import RAW_DIR

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def main() -> None:
    if not RAW_DIR.is_dir():
        raise SystemExit("Dataset missing. Run: python src/dataset.py --force")
    dataset = ImageFolder(RAW_DIR)
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, class_id in dataset.samples:
        counts[dataset.classes[class_id]] += 1
    OUTPUTS_DIR.mkdir(exist_ok=True); FIGURES_DIR.mkdir(exist_ok=True)
    summary = {"num_classes": len(dataset.classes), "num_images": len(dataset), "class_counts": counts}
    (OUTPUTS_DIR / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUTS_DIR / "eda_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["class_name", "image_count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {"class_name": class_name, "image_count": count}
            for class_name, count in counts.items()
        )

    plt.figure(figsize=(14, 8))
    plt.barh(list(counts), list(counts.values()), color="#2a6fbb")
    plt.xlabel("Number of images"); plt.ylabel("Food class"); plt.title("30VNFoods Class Distribution")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180); plt.close()

    random.seed(42)
    examples = []
    for class_name in dataset.classes:
        images = list((RAW_DIR / class_name).glob("*"))
        if images:
            examples.append((class_name, random.choice(images)))
    columns = 5; rows = (len(examples) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3))
    for axis, item in zip(axes.flat, examples):
        class_name, image_path = item
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(class_name, fontsize=9); axis.axis("off")
    for axis in axes.flat[len(examples):]: axis.axis("off")
    figure.suptitle("One Sample Image per Vietnamese Food Class", fontsize=16)
    figure.tight_layout(); figure.savefig(FIGURES_DIR / "sample_images.png", dpi=180); plt.close(figure)
    print(f"Created EDA for {len(dataset.classes)} classes and {len(dataset)} images.")


if __name__ == "__main__":
    main()
