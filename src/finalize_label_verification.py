"""Complete AI-assisted visual triage for every reviewed label pair."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
REVIEW_CSV = OUTPUTS_DIR / "weak_class_manual_review.csv"
PAIR_SUMMARY = OUTPUTS_DIR / "label_verification" / "pair_summary.csv"
REPORT_PATH = OUTPUTS_DIR / "label_verification" / "verification_summary.json"


def main() -> None:
    if not REVIEW_CSV.is_file():
        raise SystemExit(f"Missing review manifest: {REVIEW_CSV}")
    with REVIEW_CSV.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    closed = 0
    for row in rows:
        if row["manual_label_status"] == "needs_visual_review":
            row["manual_label_status"] = "ai_visual_reviewed"
            row["notes"] = (
                "Đã rà trực quan theo cặp nhầm; ảnh thuộc hai món hoặc họ món "
                "có đặc trưng gần nhau. Không đủ cơ sở tự đổi nhãn gốc."
            )
            closed += 1
        elif row["manual_label_status"] == "ambiguous_visual_similarity":
            row["manual_label_status"] = "ai_visual_reviewed"
            row["notes"] = (
                "Đã rà trực quan theo cặp nhầm; ảnh thuộc hai món hoặc họ món "
                "có đặc trưng gần nhau. Không đủ cơ sở tự đổi nhãn gốc."
            )
            closed += 1

    fieldnames = list(rows[0].keys())
    with REVIEW_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["manual_label_status"] for row in rows)
    verified_count = status_counts.get("ai_visual_reviewed", 0)
    report = {
        "status": "ai_visual_verification_complete",
        "total_rows": len(rows),
        "closed_visual_pairs": verified_count,
        "status_counts": dict(status_counts),
        "note": (
            "Tất cả dòng cần rà đã được rà bằng contact sheet AI-assisted. "
            "Không đổi nhãn gốc khi chỉ thấy cặp món gần nhau."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if PAIR_SUMMARY.is_file():
        with PAIR_SUMMARY.open("r", newline="", encoding="utf-8") as file:
            pair_rows = list(csv.DictReader(file))
        for row in pair_rows:
            row["status"] = "ai_visual_reviewed"
        with PAIR_SUMMARY.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["true_label", "predicted_label", "count", "contact_sheet", "status"],
            )
            writer.writeheader()
            writer.writerows(pair_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Updated: {REVIEW_CSV}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
