"""Prepare contact sheets and verification summaries for manual label audit."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "outputs" / "weak_class_manual_review.csv"
OUTPUT_DIR = ROOT / "outputs" / "label_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> None:
    with REVIEW_CSV.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    pending = [row for row in rows if row["manual_label_status"] == "needs_visual_review"]

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pending:
        groups[(row["true_label"], row["predicted_label"])].append(row)

    pair_rows = []
    for pair, pair_items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        pair_items.sort(key=lambda row: float(row["confidence"]), reverse=True)
        safe_pair = "__".join(part.replace(" ", "_") for part in pair)
        sheet_path = OUTPUT_DIR / f"{safe_pair}.jpg"
        create_contact_sheet(pair_items, pair, sheet_path)
        pair_rows.append({
            "true_label": pair[0],
            "predicted_label": pair[1],
            "count": len(pair_items),
            "contact_sheet": str(sheet_path.relative_to(ROOT)).replace("\\", "/"),
            "status": "pending_visual_review",
        })

    pair_summary = OUTPUT_DIR / "pair_summary.csv"
    with pair_summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["true_label", "predicted_label", "count",
                                                   "contact_sheet", "status"])
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Pending visual rows: {len(pending)}")
    print(f"Unique confusion pairs: {len(groups)}")
    print(f"Contact sheets: {OUTPUT_DIR}")
    print(f"Pair summary: {pair_summary}")


def create_contact_sheet(items: list[dict[str, str]], pair: tuple[str, str], output: Path) -> None:
    columns = 4
    rows = math.ceil(len(items) / columns)
    cell_width = 310
    cell_height = 300
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 70), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(24)
    label_font = load_font(16)
    draw.text((20, 18), f"True: {pair[0]}  ->  Predicted: {pair[1]}",
              fill="black", font=title_font)

    for index, row in enumerate(items):
        column = index % columns
        row_index = index // columns
        left = column * cell_width + 5
        top = row_index * cell_height + 70
        try:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB").copy()
        except Exception:
            continue
        image.thumbnail((300, 250), Image.Resampling.LANCZOS)
        image_left = left + (300 - image.width) // 2
        sheet.paste(image, (image_left, top))
        text = f"{index + 1}. conf={float(row['confidence']):.1%}"
        draw.text((left + 8, top + 252), text, fill="black", font=label_font)
        draw.text((left + 8, top + 272), Path(row["image_path"]).name,
                  fill="black", font=label_font)

    sheet.save(output, quality=92)


if __name__ == "__main__":
    main()
