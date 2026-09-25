from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from recognize_independent import find_plate_candidate
from train_independent_structured import parse_source_name


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser(description="Measure classical plate-box proposals against train(1) validation box labels")
    parser.add_argument("--manifest", type=Path, default=Path("data/independent_chars_train1_enhanced/manifest.json"))
    parser.add_argument("--source-root", type=Path, default=Path("data/OCR/OCR/images/train(1)/detection"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/plate_localization_validation.json"))
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [r for r in records if r["split"] == "val"]
    groups = defaultdict(lambda: {"images": 0, "found": 0, "iou_sum": 0.0, "iou50": 0, "iou75": 0})
    details = []
    for index, record in enumerate(records, 1):
        parsed = parse_source_name(Path(record["source"]))
        group = groups[record["type"]]
        group["images"] += 1
        if parsed is None:
            details.append({"source": record["source"], "type": record["type"], "reason": "invalid_label"})
            continue
        _, x1, y1, x2, y2, _ = parsed
        image = cv2.imread(str(args.source_root / record["source"]))
        found = find_plate_candidate(image, return_box=True) if image is not None else None
        if found is None:
            details.append({"source": record["source"], "type": record["type"], "reason": "not_found", "iou": 0.0})
            continue
        _, _, _, predicted = found
        overlap = iou(predicted, (x1, y1, x2 - x1, y2 - y1))
        group["found"] += 1
        group["iou_sum"] += overlap
        group["iou50"] += int(overlap >= 0.5)
        group["iou75"] += int(overlap >= 0.75)
        details.append({"source": record["source"], "type": record["type"], "ground_truth_box": [x1, y1, x2-x1, y2-y1], "predicted_box": list(predicted), "iou": overlap})
        if index % 500 == 0:
            print(f"images={index}/{len(records)}", flush=True)

    summary = {}
    for key, value in groups.items():
        n = value["images"]
        summary[key] = {
            "images": n,
            "found": value["found"],
            "recall": value["found"] / max(1, n),
            "mean_iou_all": value["iou_sum"] / max(1, n),
            "iou_at_least_0_5": value["iou50"] / max(1, n),
            "iou_at_least_0_75": value["iou75"] / max(1, n),
        }
    result = {
        "validation_plates": len(records),
        "detector": "contour+morphology heuristic",
        "groups": summary,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
