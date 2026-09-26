from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from train_independent_structured import (
    LeNet5Structured50,
    enhance_plate_for_ocr,
    parse_source_name,
    perspective_correct,
    segment_characters,
)
from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes


SOURCE_ROOT = Path("data/OCR/OCR/images/train(1)/detection")
DATA_ROOT = Path("data/independent_chars_train1")


def load_model(model_path: Path, architecture: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, model_path)
    model = (LeNet5 if architecture == "dense" else LeNet5Structured50)(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def score_records(records, enhancement: str, model, device: torch.device, data_root: Path):
    tensors = []
    ownership = []
    image_results = []
    for index, record in enumerate(records):
        parsed = parse_source_name(Path(record["source"]))
        if parsed is None:
            image_results.append({"record": record, "reason": "invalid_source_label"})
            continue
        expected, x1, y1, x2, y2, _ = parsed
        source_image = cv2.imread(str(data_root / record["source"]))
        if source_image is None:
            image_results.append({"record": record, "reason": "image_read_failed"})
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(source_image.shape[1], x2), min(source_image.shape[0], y2)
        plate = source_image[y1:y2, x1:x2]
        if plate.size == 0:
            image_results.append({"record": record, "reason": "empty_roi"})
            continue
        corrected, _, _ = perspective_correct(plate)
        segmented = segment_characters(corrected, expected_count=None, enhancement=enhancement)
        if segmented is None:
            image_results.append({"record": record, "reason": "segmentation_failed", "expected": expected})
            continue
        crops, row_count, count = segmented
        if count not in (8, 9, 10):
            image_results.append({
                "record": record,
                "reason": "unsupported_character_count",
                "expected": expected,
                "detected_count": count,
                "row_count": row_count,
            })
            continue
        result = {
            "record": record,
            "expected": expected,
            "row_count": row_count,
            "detected_count": count,
            "start": len(tensors),
            "end": len(tensors) + count,
        }
        for crop in crops:
            pixels = crop.astype(np.float32) / 255.0
            tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
            ownership.append(index)
        image_results.append(result)

    predictions = []
    batch_size = 512
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            predictions.extend(model(batch).argmax(1).cpu().tolist())
    predicted_texts = [CLASS_NAMES[i] for i in predictions]

    aggregates = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct_characters": 0, "skipped": 0, "count_mismatch": 0})
    details = []
    for item in image_results:
        record = item["record"]
        keys = ("all", record["type"])
        if "reason" in item:
            details.append({"source": record["source"], "type": record["type"], "expected": item.get("expected", record["text"]), "predicted": "", "status": "skipped", "reason": item["reason"], "detected_count": item.get("detected_count")})
            for key in keys:
                aggregates[key]["plates"] += 1
                aggregates[key]["skipped"] += 1
            continue
        predicted = "".join(predicted_texts[item["start"] : item["end"]])
        expected = item["expected"]
        char_count_mismatch = len(predicted) != len(expected)
        correct = sum(a == b for a, b in zip(expected, predicted))
        exact = predicted == expected
        status = "correct" if exact else "wrong"
        details.append({
            "source": record["source"],
            "type": record["type"],
            "expected": expected,
            "predicted": predicted,
            "status": status,
            "detected_count": item["detected_count"],
            "count_mismatch": char_count_mismatch,
            "row_count": item["row_count"],
            "correct_characters": correct,
            "expected_characters": len(expected),
        })
        for key in keys:
            stats = aggregates[key]
            stats["plates"] += 1
            stats["exact"] += int(exact)
            stats["characters"] += len(expected)
            stats["correct_characters"] += correct
            stats["count_mismatch"] += int(char_count_mismatch)

    def finalized(stats):
        return {
            "plates": stats["plates"],
            "exact_plates": stats["exact"],
            "plate_exact_accuracy_end_to_end": stats["exact"] / max(1, stats["plates"]),
            "character_accuracy_position": stats["correct_characters"] / max(1, stats["characters"]),
            "skipped": stats["skipped"],
            "skip_rate": stats["skipped"] / max(1, stats["plates"]),
            "character_count_mismatch": stats["count_mismatch"],
        }

    return {"groups": {key: finalized(stats) for key, stats in sorted(aggregates.items())}, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Compare plate contrast/sharpening modes on the fixed train(1) validation split")
    parser.add_argument("--model", type=Path, default=Path("artifacts/lenet5_dense_baseline_train1.pt"))
    parser.add_argument("--architecture", choices=("dense", "structured"), default="dense")
    parser.add_argument("--manifest", type=Path, default=Path("data/independent_chars_train1/manifest.json"))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/plate_enhancement_ablation.json"))
    parser.add_argument("--methods", nargs="+", choices=("none", "clahe", "clahe_sharp"), default=("none", "clahe", "clahe_sharp"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [r for r in records if r["split"] == "val"]
    model = load_model(args.model, args.architecture, device)
    all_results = {
        "model": str(args.model),
        "architecture": args.architecture,
        "validation_plates": len(records),
        "source_root": str(args.source_root),
        "note": "Ground-truth plate box coordinates locate the ROI; no character count is passed to segmentation. Results include crop/segmentation failures.",
        "methods": {},
    }
    for method in args.methods:
        print(f"evaluating={method}", flush=True)
        result = score_records(records, method, model, device, args.source_root)
        all_results["methods"][method] = result
        print(json.dumps({"method": method, "groups": result["groups"]}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
