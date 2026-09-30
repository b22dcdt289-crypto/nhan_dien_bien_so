"""Curate two-row plate crops from the existing train(1) preparation.

No characters or pixels are generated. The source preparation uses the
annotated bounding box and a geometric homography where a border is found.
This script chooses readable, near-frontal crops and preserves provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("data/two_row_frontal_dense_v1")
OUTPUT = Path("data/two_row_front_curated_2000_v2")
GROUPS = {"two_row_car": "car_2row", "two_row_motorcycle": "motorcycle_2row"}


def quality(image: np.ndarray) -> dict | None:
    """Rank clarity and apparent alignment; this is not a ground-truth audit."""
    height, width = image.shape[:2]
    ratio = width / max(1, height)
    if width < 35 or height < 27 or not 0.75 <= ratio <= 2.6:
        return None
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    contrast = float(np.percentile(image, 90) - np.percentile(image, 10))
    edges = cv2.Canny(image, 55, 155)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(12, width // 8),
        minLineLength=max(12, width // 4), maxLineGap=5,
    )
    horizontal = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            angle = min(angle, 180 - angle)
            if angle <= 30:
                horizontal.append((length, angle))
    horizontal.sort(reverse=True)
    mean_angle = float(
        sum(length * angle for length, angle in horizontal[:4])
        / max(1.0, sum(length for length, _ in horizontal[:4]))
    ) if horizontal else 30.0
    # Keep the score explicitly as a proxy; off-axis depth is not recoverable.
    score = (
        1.8 * np.log1p(blur)
        + 0.025 * contrast
        + 0.018 * min(width, 170)
        + 0.018 * min(height, 130)
        - 0.27 * mean_angle
    )
    return {
        "width": width, "height": height, "aspect_ratio": round(ratio, 3),
        "laplacian_variance": round(blur, 2),
        "p90_p10_contrast": round(contrast, 2),
        "horizontal_line_angle_proxy_deg": round(mean_angle, 2),
        "selection_score": round(float(score), 3),
    }


def choose(records: list[dict], count: int) -> list[dict]:
    """Prefer distinct identities, then additional source frames if needed."""
    ranked = sorted(records, key=lambda row: (-row["quality"]["selection_score"], row["source"]))
    chosen, used_texts, used_sources = [], set(), set()
    for row in ranked:
        if row["text"] not in used_texts:
            chosen.append(row)
            used_texts.add(row["text"])
            used_sources.add(row["source"])
            if len(chosen) == count:
                return chosen
    for row in ranked:
        if row["source"] not in used_sources:
            chosen.append(row)
            used_sources.add(row["source"])
            if len(chosen) == count:
                return chosen
    raise RuntimeError(f"Only {len(chosen)} usable crops; need {count}")


def make_sheet(output: Path, group: str, rows: list[dict]) -> None:
    rng = random.Random(20260930)
    sample = rng.sample(rows, min(36, len(rows)))
    tile_w, tile_h, cols = 200, 155, 6
    canvas = np.full((6 * tile_h, cols * tile_w, 3), 235, np.uint8)
    for index, row in enumerate(sample):
        image = cv2.imread(str(output / row["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        scale = min((tile_w - 12) / image.shape[1], (tile_h - 32) / image.shape[0])
        thumb = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        x = (index % cols) * tile_w + (tile_w - thumb.shape[1]) // 2
        y = (index // cols) * tile_h + 5
        canvas[y:y + thumb.shape[0], x:x + thumb.shape[1]] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, row["id"], ((index % cols) * tile_w + 5, (index // cols + 1) * tile_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / f"review_{group}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-group", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output already exists and is non-empty: {args.output}")
    source_records = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    pool = {group: [] for group in GROUPS}
    rejected = Counter()
    for record in source_records:
        if record["type"] not in GROUPS or record["split"] != "train":
            continue
        if not record["ground_truth_segmentation_ok"] or record["detected_rows"] != 2:
            rejected[(record["type"], "not_two_rows_or_segmentation_failed")] += 1
            continue
        image = cv2.imread(str(args.source / record["plate_crop"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected[(record["type"], "unreadable_crop")] += 1
            continue
        metrics = quality(image)
        if metrics is None:
            rejected[(record["type"], "invalid_crop_geometry")] += 1
            continue
        if record["type"] == "two_row_car" and (metrics["width"] < 50 or metrics["height"] < 40):
            rejected[(record["type"], "car_crop_too_small_for_curated_set")] += 1
            continue
        if metrics["laplacian_variance"] < 300:
            rejected[(record["type"], "blur_proxy_below_300")] += 1
            continue
        pool[record["type"]].append({**record, "quality": metrics})
    selected = {group: choose(pool[group], args.per_group) for group in GROUPS}
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for group, rows in selected.items():
        folder = args.output / GROUPS[group]
        folder.mkdir(parents=True, exist_ok=True)
        group_rows = []
        for index, row in enumerate(rows, start=1):
            source_file = args.source / row["plate_crop"]
            image = cv2.imread(str(source_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Source crop disappeared: {source_file}")
            image_id = f"{GROUPS[group]}_{index:04d}"
            target = folder / f"{image_id}.png"
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"Could not write {target}")
            saved = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if saved is None or not np.array_equal(saved, image):
                raise RuntimeError(f"Pixel verification failed for {target}")
            out = {
                "id": image_id, "file": target.relative_to(args.output).as_posix(),
                "group": group, "text_from_source_filename": row["text"],
                "original_source": row["source"], "prepared_source": row["plate_crop"],
                "rectification_method": row["rectification_method"],
                "geometry_corrected": row["perspective_corrected"],
                "rows_detected": row["detected_rows"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "manual_review_recommended": bool(
                    row["quality"]["horizontal_line_angle_proxy_deg"] > 15
                    or not row["perspective_corrected"]
                ),
                **row["quality"],
            }
            output_rows.append(out)
            group_rows.append(out)
        make_sheet(args.output, GROUPS[group], group_rows)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "source": str(args.source), "output": str(args.output),
        "requested_per_group": args.per_group,
        "eligible_by_group": {group: len(pool[group]) for group in GROUPS},
        "selected_by_group": {group: len(selected[group]) for group in GROUPS},
        "unique_plate_texts_by_group": {group: len({row["text"] for row in selected[group]}) for group in GROUPS},
        "geometry_corrected_by_group": {group: sum(bool(row["perspective_corrected"]) for row in selected[group]) for group in GROUPS},
        "manual_review_recommended_by_group": {
            group: sum(
                row["quality"]["horizontal_line_angle_proxy_deg"] > 15
                or not row["perspective_corrected"] for row in selected[group]
            ) for group in GROUPS
        },
        "rejected": {"|".join(key): value for key, value in rejected.items()},
        "pixel_verification": "All saved grayscale PNGs reloaded and compared exactly",
        "label_note": "Plate text is from the original source filename; not manually verified for all 2000 images.",
        "frontal_note": "Geometric rectification and alignment-ranking yield approximately frontal crops; they cannot restore occluded text or prove an original head-on camera view.",
        "split_note": "Only source train split used; existing validation/test remain untouched.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
