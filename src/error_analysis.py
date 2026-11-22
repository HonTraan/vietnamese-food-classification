"""Render the 15 high-confidence errors and create a report template for human analysis."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "predictions" / "misclassified_top15.csv"
FIGURE = ROOT / "outputs" / "figures" / "misclassified_top15.png"
REPORT = ROOT / "outputs" / "error_analysis.md"


def main() -> None:
    if not PREDICTIONS.is_file():
        raise SystemExit("Missing misclassified_top15.csv. Run: python src/evaluate.py")
    with PREDICTIONS.open(encoding="utf-8") as file: rows = list(csv.DictReader(file))
    figure, axes = plt.subplots(3, 5, figsize=(15, 10))
    for axis, row in zip(axes.flat, rows):
        with Image.open(row["image_path"]) as image: axis.imshow(image.convert("RGB"))
        axis.set_title(f"True: {row['true_label']}\nPred: {row['predicted_label']} ({float(row['confidence']):.1%})", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(rows):]: axis.axis("off")
    figure.suptitle("15 High-Confidence Misclassifications", fontsize=16)
    figure.tight_layout(); FIGURE.parent.mkdir(exist_ok=True); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    causes = {
        ("Banh can", "Banh khot"): "Hai món có hình dáng bánh rán tương tự; cần kiểm tra nhãn gốc.",
        ("Bun bo Hue", "Bun rieu"): "Nước dùng đỏ, sợi bún và rau tạo đặc trưng chung.",
        ("Banh tet", "Banh chung"): "Cùng là gạo nếp/bọc lá; góc crop làm hai món khó phân biệt.",
        ("Canh chua", "Ca kho to"): "Ảnh cá trong nồi và gia vị che đặc trưng canh chua.",
        ("Banh canh", "Chao long"): "Tô nước có thịt/màu tương tự; crop nên tập trung phần sợi.",
        ("Banh cuon", "Goi cuon"): "Cùng có bánh, rau và nước chấm; bố cục gần giống nhau.",
        ("Banh duc", "Nem chua"): "Bánh/rau ít đặc trưng; cần kiểm tra nhãn gốc và ánh sáng.",
        ("Hu tieu", "Bun rieu"): "Màu nước, thịt và sợi bún làm hai tô nước dễ nhầm.",
        ("Banh xeo", "Banh khot"): "Hai món là bột chiên, rau và topping; ảnh thiếu chi tiết đặc trưng.",
        ("Banh mi", "Goi cuon"): "Bố cục ảnh tập trung đĩa cuốn/rau thay vì ổ bánh mì.",
        ("Chao long", "Bun bo Hue"): "Mô hình nhầm đặc trưng chung của món bún nước.",
        ("Pho", "Bun rieu"): "Góc chụp và nước dùng/nhiều topping làm giảm khác biệt.",
        ("Pho", "Bun bo Hue"): "Nước dùng và sợi bún giống nhau; thiếu gia vị/thịt đặc trưng.",
    }
    lines = ["# Phân tích 15 ảnh dự đoán sai", "",
             "Các nhận xét dựa trên quan sát ảnh trong `figures/misclassified_top15.png`.", "",
             "| STT | Nhãn đúng | Nhãn dự đoán | Confidence | Nguyên nhân có thể |",
             "|---:|---|---|---:|---|"]
    for index, row in enumerate(rows, 1):
        key = (row["true_label"], row["predicted_label"])
        cause = causes.get(key, "Hai nhãn có đặc trưng hình ảnh gần nhau; cần xem lại nhãn và crop.")
        lines.append(
            f"| {index} | {row['true_label']} | {row['predicted_label']} | "
            f"{float(row['confidence']):.1%} | {cause} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {FIGURE} and {REPORT}")


if __name__ == "__main__": main()
