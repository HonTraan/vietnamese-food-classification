"""Create a manual-review manifest for weak classes and confusion pairs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
WEAK_CLASSES = ("Banh duc", "Banh canh", "Hu tieu", "Pho", "Bun bo Hue")


def main() -> None:
    metrics_path = OUTPUTS_DIR / "metrics.json"
    predictions_path = OUTPUTS_DIR / "predictions" / "test_predictions.csv"
    if not metrics_path.is_file():
        raise SystemExit(f"Missing {metrics_path}. Run src/evaluate.py first.")
    if not predictions_path.is_file():
        raise SystemExit(f"Missing {predictions_path}. Run src/evaluate.py first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = metrics["report"]
    weak_rows = []
    for class_name in WEAK_CLASSES:
        values = report[class_name]
        weak_rows.append({
            "class_name": class_name,
            "support": int(values["support"]),
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "status": "needs_manual_review",
        })

    weak_report_path = OUTPUTS_DIR / "weak_class_audit.csv"
    with weak_report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=weak_rows[0].keys())
        writer.writeheader()
        writer.writerows(weak_rows)

    with predictions_path.open("r", newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))

    review_rows = []
    for prediction in predictions:
        true_label = prediction["true_label"]
        predicted_label = prediction["predicted_label"]
        if true_label in WEAK_CLASSES or predicted_label in WEAK_CLASSES:
            is_correct = prediction["correct"] == "True"
            if is_correct:
                manual_status = "correct_prediction"
                notes = "Mô hình dự đoán đúng; không cần đổi nhãn."
            else:
                manual_status = "needs_visual_review"
                notes = (
                    f"Nhầm {true_label} thành {predicted_label}; "
                    "kiểm tra nhãn gốc, crop và nội dung chính của ảnh."
                )
            review_rows.append({
                "image_path": prediction["image_path"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": prediction["confidence"],
                "correct": prediction["correct"],
                "review_reason": "weak_true_or_predicted_class",
                "manual_label_status": manual_status,
                "notes": notes,
            })

    review_path = OUTPUTS_DIR / "weak_class_manual_review.csv"
    with review_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)

    confusion_pairs = {}
    for row in predictions:
        if row["true_label"] in WEAK_CLASSES and row["correct"] == "False":
            pair = (row["true_label"], row["predicted_label"])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    pair_rows = [
        {"true_label": pair[0], "predicted_label": pair[1], "count": count}
        for pair, count in sorted(confusion_pairs.items(), key=lambda item: item[1], reverse=True)
    ]
    pair_path = OUTPUTS_DIR / "weak_class_confusion_pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["true_label", "predicted_label", "count"])
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Weak class report: {weak_report_path}")
    print(f"Manual review rows: {len(review_rows)}")
    print(f"Review manifest: {review_path}")
    print(f"Confusion pairs: {pair_path}")


if __name__ == "__main__":
    main()
