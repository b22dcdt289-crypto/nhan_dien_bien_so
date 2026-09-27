from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from train_independent_structured import estimate_blur, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES


DEFAULT_SOURCE = Path("data/OCR/OCR/images/train(1)/detection")
DEFAULT_OUTPUT = Path("data/two_row_frontal_dense_v1")


def identity_split(text: str) -> str:
    bucket = hashlib.sha256(text.upper().encode("ascii", errors="ignore")).digest()[0] % 100
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "val"
    return "train"


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.asarray(
        [points[np.argmin(sums)], points[np.argmin(differences)], points[np.argmax(sums)], points[np.argmax(differences)]],
        dtype=np.float32,
    )


def rectify_projective(gray: np.ndarray):
    """Use a real four-corner homography when the plate border is visible."""
    height, width = gray.shape[:2]
    frame_area = float(height * width)
    softened = cv2.GaussianBlur(gray, (3, 3), 0)
    masks = [
        cv2.Canny(softened, 30, 130),
        cv2.threshold(softened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(softened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    best = None
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.12 or area > frame_area * 0.98:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.015, 0.025, 0.04, 0.06):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
                    continue
                quad = order_quad(polygon)
                top = np.linalg.norm(quad[1] - quad[0])
                bottom = np.linalg.norm(quad[2] - quad[3])
                left = np.linalg.norm(quad[3] - quad[0])
                right = np.linalg.norm(quad[2] - quad[1])
                out_w = int(max(top, bottom))
                out_h = int(max(left, right))
                ratio = out_w / max(1, out_h)
                if out_w < 24 or out_h < 18 or not 0.65 <= ratio <= 8.0:
                    continue
                score = float(area) * min(ratio, 1 / ratio)
                if best is None or score > best[0]:
                    best = (score, quad, out_w, out_h)
    if best is None:
        return None
    _, quad, out_w, out_h = best
    destination = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(gray, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def read_candidate(path: Path, source_root: Path):
    parts = path.stem.split("_")
    if len(parts) < 8:
        return None, "bad_filename"
    text = parts[3].upper()
    if len(text) not in (7, 8, 9, 10) or any(character not in CLASS_NAMES for character in text):
        return None, "invalid_or_unsupported_text"
    try:
        x1, y1, x2, y2 = map(int, parts[4:8])
    except ValueError:
        return None, "invalid_box_coordinates"
    plate_type = "two_row_motorcycle" if path.parent.name == "two_rows_label_xe_may" else "two_row_car"
    return {
        "source_path": path,
        "source": path.relative_to(source_root).as_posix(),
        "text": text,
        "type": plate_type,
        "box": (x1, y1, x2, y2),
        "split": identity_split(text),
    }, None


def save_contact_sheet(output: Path, records: list[dict], limit: int = 40):
    half = max(1, limit // 2)
    selected = []
    for plate_type in ("two_row_car", "two_row_motorcycle"):
        typed = [
            record for record in records
            if record.get("ground_truth_segmentation_ok") and record["split"] == "train" and record["type"] == plate_type
        ]
        selected.extend(typed[:half])
    columns, tile_w, tile_h = 4, 300, 145
    rows = (len(selected) + columns - 1) // columns
    sheet = np.full((max(1, rows) * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, record in enumerate(selected):
        image = cv2.imread(str(output / record["plate_crop"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        preview = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(image)
        h, w = preview.shape
        scale = min((tile_w - 12) / max(1, w), (tile_h - 35) / max(1, h))
        resized = cv2.resize(preview, (max(1, round(w * scale)), max(1, round(h * scale))))
        x = (index % columns) * tile_w + (tile_w - resized.shape[1]) // 2
        y = (index // columns) * tile_h + 4
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        label = f"{record['type']} | {record['text']} | rows={record['detected_rows']}"
        cv2.putText(sheet, label, ((index % columns) * tile_w + 5, (index // columns + 1) * tile_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review / "two_row_frontal_examples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main():
    parser = argparse.ArgumentParser(description="Rectify and prepare identity-disjoint two-row Vietnamese car/motorcycle plates.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-images", type=int, default=0, help="Optional dry-run limit; zero means all images")
    parser.add_argument("--enhancement", choices=("clahe_sharp",), default="clahe_sharp")
    args = parser.parse_args()

    output = args.output
    (output / "plates").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / "plates" / split).mkdir(parents=True, exist_ok=True)
        for character in CLASS_NAMES:
            (output / "chars" / split / character).mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for folder in ("two_rows", "two_rows_label_xe_may")
        for path in (args.source / folder).glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images:
        paths = paths[: args.max_images]

    records: list[dict] = []
    skipped = Counter()
    types = Counter()
    perspectives = Counter()
    dimensions = Counter()
    row_count_hist = Counter()
    segmentation_success = Counter()
    for index, path in enumerate(paths, start=1):
        item, reason = read_candidate(path, args.source)
        if reason:
            skipped[reason] += 1
            continue
        types[item["type"]] += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped[f"{item['type']}:image_read_error"] += 1
            continue
        h, w = image.shape[:2]
        x1, y1, x2, y2 = item["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[1] < 20 or roi.shape[0] < 18:
            skipped[f"{item['type']}:empty_or_too_small_roi"] += 1
            continue
        raw_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_score = estimate_blur(raw_gray)
        quad_rectified = rectify_projective(raw_gray)
        if quad_rectified is not None:
            rectified, perspective_applied = quad_rectified, True
            correction_method = "four_corner_homography"
            angle = 0.0
        else:
            rectified, perspective_applied, angle = perspective_correct(roi)
            correction_method = "rotated_rectangle" if perspective_applied else "bbox_fallback"
            if rectified.size == 0:
                rectified = raw_gray
                perspective_applied = False
                correction_method = "bbox_fallback"
        segmented = segment_characters(rectified, expected_count=len(item["text"]), enhancement=args.enhancement)
        crops = []
        detected_rows = detected_count = 0
        segmentation_ok = False
        failure_reason = ""
        if segmented is None:
            failure_reason = "segmentation_failed_or_count_mismatch"
        else:
            crops, detected_rows, detected_count = segmented
            row_count_hist[(item["type"], str(detected_rows))] += 1
            segmentation_ok = detected_rows == 2 and detected_count == len(item["text"])
            if not segmentation_ok:
                failure_reason = "not_two_rows_or_count_mismatch"

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{item['type']}_{path.stem}")
        plate_rel = Path("plates") / item["split"] / f"{safe_name}.png"
        if not cv2.imwrite(str(output / plate_rel), rectified):
            skipped[f"{item['type']}:plate_save_error"] += 1
            continue

        crop_relatives = []
        if segmentation_ok:
            for position, (character, crop) in enumerate(zip(item["text"], crops)):
                relative = Path("chars") / item["split"] / character / f"{safe_name}_{position:02d}.png"
                if not cv2.imwrite(str(output / relative), crop):
                    segmentation_ok = False
                    failure_reason = "character_crop_save_error"
                    break
                crop_relatives.append(relative.as_posix())
        if not segmentation_ok:
            crop_relatives = []
        segmentation_success[(item["split"], item["type"])] += int(segmentation_ok)
        perspectives[(item["split"], item["type"], "applied" if perspective_applied else "fallback_roi")] += 1
        dimensions[(item["type"], "width_lt_height")] += int(rectified.shape[1] < rectified.shape[0])
        record = {
            "split": item["split"],
            "source": item["source"],
            "text": item["text"],
            "type": item["type"],
            "row_count": 2,
            "detected_rows": int(detected_rows),
            "detected_characters": int(detected_count),
            "plate_crop": plate_rel.as_posix(),
            "crop_files": crop_relatives,
            "ground_truth_segmentation_ok": bool(segmentation_ok),
            "segmentation_failure_reason": failure_reason,
            "perspective_corrected": bool(perspective_applied),
            "rectification_method": correction_method,
            "estimated_roll_degrees": round(float(angle), 3),
            "laplacian_blur_score_before_rectification": round(float(blur_score), 3),
            "enhancement": args.enhancement,
        }
        records.append(record)
        if index % 1000 == 0:
            print(f"processed={index}/{len(paths)} saved={len(records)} ref_crops={sum(segmentation_success.values())}", flush=True)

    split_rows = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
    split_ids = {split: {record["text"] for record in rows} for split, rows in split_rows.items()}
    overlaps = {
        f"{first}_{second}": len(split_ids[first] & split_ids[second])
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Identity overlap in split: {overlaps}")
    summary = {
        "dataset": "Rectified two-row Vietnamese car and motorcycle plate crops",
        "source": args.source.as_posix(),
        "source_images_seen": len(paths),
        "records_saved": len(records),
        "plate_frames_by_split": {split: len(rows) for split, rows in split_rows.items()},
        "unique_plate_identities_by_split": {split: len(split_ids[split]) for split in split_rows},
        "identity_overlap_counts": overlaps,
        "source_type_counts_before_processing": dict(types),
        "reference_segmentation_success_by_split_and_type": {"|".join(key): value for key, value in segmentation_success.items()},
        "detected_row_count_histogram": {"|".join(key): value for key, value in row_count_hist.items()},
        "perspective_correction_status": {"|".join(key): value for key, value in perspectives.items()},
        "rejected_or_unreadable": dict(skipped),
        "enhancement": args.enhancement,
        "split_rule": "deterministic SHA-256 plate-identity group split: 80% train, 10% validation, 10% test; same registration text never spans splits",
        "limitations": [
            "Source training collection has no official held-out test split; this split is group-disjoint by plate text but may overlap earlier model pretraining data.",
            "Frame labels are taken from source filenames and have not all been manually audited.",
            "Perspective warp is applied only when a plausible plate quadrilateral is detected; otherwise the annotated ROI is retained and marked fallback_roi.",
            "Reference character crops use the supplied plate text length to select a segmentation candidate; those metrics are conditional on count-assisted segmentation.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "preparation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_contact_sheet(output, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
