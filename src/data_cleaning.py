"""Detect duplicate images and create a leak-safe train/validation/test manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
MANIFEST_PATH = OUTPUTS_DIR / "clean_split_manifest.json"
DUPLICATE_REPORT_PATH = OUTPUTS_DIR / "duplicate_report.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = [image.getpixel((x, y)) for y in range(size) for x in range(size + 1)]
    value = 0
    width = size + 1
    for y in range(size):
        for x in range(size):
            value = (value << 1) | int(pixels[y * width + x] < pixels[y * width + x + 1])
    return value


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, dict[int, "BKTree"]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node_value, children = self.root
        distance = (value ^ node_value).bit_count()
        if distance == 0:
            return
        if distance not in children:
            children[distance] = BKTree()
        children[distance].add(value)

    def search(self, value: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        node_value, children = self.root
        distance = (value ^ node_value).bit_count()
        matches = [node_value] if distance <= threshold else []
        for child_distance in range(max(1, distance - threshold), distance + threshold + 1):
            if child_distance in children:
                matches.extend(children[child_distance].search(value, threshold))
        return matches


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-distance", type=int, default=5)
    parser.add_argument("--apply", action="store_true", help="Move excluded duplicates to data/quarantine.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.near_distance < 0:
        raise SystemExit("--near-distance must be non-negative.")
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Dataset missing: {RAW_DIR}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    union = list(range(len(image_paths)))

    def find(index: int) -> int:
        while union[index] != index:
            union[index] = union[union[index]]
            index = union[index]
        return index

    def union_sets(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            union[right] = left

    sha_groups: dict[str, list[int]] = defaultdict(list)
    hashes: list[int | None] = [None] * len(image_paths)
    unreadable: list[tuple[int, str]] = []
    for index, path in enumerate(image_paths):
        try:
            sha_groups[file_sha256(path)].append(index)
            hashes[index] = dhash(path)
        except Exception as error:
            unreadable.append((index, str(error)))

    for indices in sha_groups.values():
        for index in indices[1:]:
            union_sets(indices[0], index)

    # Match perceptual neighbors in a BK-tree, but only verify the candidates returned.
    tree = BKTree()
    hash_owner: dict[int, int] = {}
    hash_seen: set[int] = set()
    for index, hash_value in enumerate(hashes):
        if hash_value is None:
            continue
        if hash_value not in hash_seen:
            candidates = tree.search(hash_value, args.near_distance)
            if not candidates:
                tree.add(hash_value)
            else:
                owner_index = hash_owner[candidates[0]]
                union_sets(index, owner_index)
            hash_seen.add(hash_value)
            hash_owner[hash_value] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(image_paths)):
        if index not in {item[0] for item in unreadable}:
            groups[find(index)].append(index)

    retained_indices: list[int] = []
    excluded_indices: set[int] = set()
    report_rows: list[dict[str, str | int]] = []
    for indices in groups.values():
        retained = min(indices)
        retained_indices.append(retained)
        for index in indices:
            if index == retained:
                continue
            excluded_indices.add(index)
            report_rows.append({
                "kept_path": relative(image_paths[retained]),
                "excluded_path": relative(image_paths[index]),
                "reason": "duplicate",
                "same_label": str(image_paths[retained].parent.name == image_paths[index].parent.name),
            })

    for index, error in unreadable:
        excluded_indices.add(index)
        report_rows.append({
            "kept_path": "",
            "excluded_path": relative(image_paths[index]),
            "reason": f"unreadable: {error}",
            "same_label": "",
        })

    retained_indices.sort(key=lambda index: relative(image_paths[index]))
    retained_paths = [image_paths[index] for index in retained_indices]
    classes = sorted({path.parent.name for path in retained_paths})
    class_to_id = {name: index for index, name in enumerate(classes)}

    indices_by_class: dict[str, list[int]] = defaultdict(list)
    for index, path in enumerate(retained_paths):
        indices_by_class[path.parent.name].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    too_small: dict[str, int] = {}
    for class_name in classes:
        class_indices = indices_by_class[class_name]
        if len(class_indices) < 7:
            too_small[class_name] = len(class_indices)
            continue
        train, held_out = train_test_split(
            class_indices, test_size=0.30, random_state=args.seed, shuffle=True
        )
        val, test = train_test_split(
            held_out, test_size=0.50, random_state=args.seed, shuffle=True
        )
        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)
    if too_small:
        raise SystemExit(f"Classes need at least 7 retained images for 70/15/15 split: {too_small}")

    manifest = {
        "seed": args.seed,
        "classes": classes,
        "train": [relative(retained_paths[index]) for index in train_indices],
        "validation": [relative(retained_paths[index]) for index in val_indices],
        "test": [relative(retained_paths[index]) for index in test_indices],
        "excluded_count": len(excluded_indices),
        "duplicate_count": len(report_rows) - len(unreadable),
        "unreadable_count": len(unreadable),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with DUPLICATE_REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["kept_path", "excluded_path", "reason", "same_label"])
        writer.writeheader()
        writer.writerows(report_rows)

    if args.apply:
        quarantine = ROOT / "data" / "quarantine"
        for index in excluded_indices:
            source = image_paths[index]
            destination = quarantine / source.relative_to(RAW_DIR)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    print(
        f"Images={len(image_paths)}; retained={len(retained_paths)}; "
        f"duplicates={manifest['duplicate_count']}; unreadable={manifest['unreadable_count']}"
    )
    print(
        f"Splits train/validation/test="
        f"{len(manifest['train'])}/{len(manifest['validation'])}/{len(manifest['test'])}"
    )
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Duplicate report: {DUPLICATE_REPORT_PATH}")
    if args.apply:
        print("Excluded files moved to data/quarantine/")


if __name__ == "__main__":
    main()
