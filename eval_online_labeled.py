from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from train_independent_structured import LeNet5Structured50, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES


LABEL_RE = re.compile(r"^([A-Z0-9]+)_")


def plate_label(path: Path) -> str | None:
    match = LABEL_RE.match(path.stem.upper())
    return match.group(1) if match else None


def position_accuracy(expected: str, predicted: str) -> float:
    width = max(len(expected), len(predicted))
    if width == 0:
        return 0.0
    return sum(a == b for a, b in zip(expected, predicted)) / width


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the independent LeNet-5 plate OCR on labeled online images")
    parser.add_argument("images", type=Path, help="directory containing images named LABEL_*.jpg")
    parser.add_argument("--model", type=Path, default=Path("artifacts/independent_lenet5_structured50_train1.pt"))
    parser.add_argument("--limit", type=int, default=0, help="maximum number of images; 0 means all")
    parser.add_argument("--output", type=Path, default=Path("artifacts/online_test_metrics.json"))
    args = parser.parse_args()

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit > 0:
        files = files[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = LeNet5Structured50(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for path in files:
        expected = plate_label(path)
        if expected is None:
            reasons["missing_filename_label"] += 1
            continue
        image = cv2.imread(str(path))
        result: dict[str, object] = {"file": path.name, "expected": expected}
        if image is None:
            result.update(status="skipped", reason="image_read_failed")
            reasons["image_read_failed"] += 1
            rows.append(result)
            continue
        plate, corrected, angle = perspective_correct(image)
        segmented = segment_characters(plate, expected_count=None)
        if segmented is None:
            result.update(status="skipped", reason="segmentation_failed")
            reasons["segmentation_failed"] += 1
            rows.append(result)
            continue
        crops, row_count, char_count = segmented
        if char_count not in (8, 9, 10):
            result.update(status="skipped", reason="ambiguous_character_count", detected_characters=char_count, rows=row_count)
            reasons["ambiguous_character_count"] += 1
            rows.append(result)
            continue
        batch = []
        for crop in crops:
            pixels = crop.astype(np.float32) / 255.0
            batch.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
        with torch.no_grad():
            predictions = model(torch.stack(batch).to(device)).argmax(1).cpu().tolist()
        predicted = "".join(CLASS_NAMES[index] for index in predictions)
        exact = predicted == expected
        result.update(
            status="correct" if exact else "wrong",
            predicted=predicted,
            detected_characters=char_count,
            rows=row_count,
            perspective_corrected=corrected,
            angle_deg=round(float(angle), 3),
            position_accuracy=position_accuracy(expected, predicted),
            macs_this_plate=int(checkpoint["macs_per_character"] * char_count),
        )
        reasons["correct" if exact else "wrong"] += 1
        rows.append(result)

    attempted = [row for row in rows if row.get("status") in {"correct", "wrong"}]
    correct = sum(row.get("status") == "correct" for row in attempted)
    position_scores = [float(row["position_accuracy"]) for row in attempted]
    metrics = {
        "source_directory": str(args.images),
        "model": str(args.model),
        "total_images": len(files),
        "attempted_images": len(attempted),
        "correct_plates": correct,
        "wrong_plates": sum(row.get("status") == "wrong" for row in attempted),
        "skipped_images": len(rows) - len(attempted),
        "plate_exact_accuracy_attempted": correct / len(attempted) if attempted else 0.0,
        "plate_exact_accuracy_end_to_end": correct / len(rows) if rows else 0.0,
        "character_position_accuracy_attempted": sum(position_scores) / len(position_scores) if position_scores else 0.0,
        "skip_rate": (len(rows) - len(attempted)) / len(rows) if rows else 0.0,
        "reasons": dict(reasons),
        "macs_per_character": checkpoint["macs_per_character"],
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
