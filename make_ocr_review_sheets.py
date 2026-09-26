from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Render OCR/filename-label disagreements for human verification.")
    parser.add_argument("--data", type=Path, default=Path("data/one_row_1000_curated_v4"))
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    source_csv = args.data / "review" / "ocr_mismatches.csv"
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = args.data / "review" / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 280, 145
    per_sheet = args.rows * args.columns
    for page, start in enumerate(range(0, len(rows), per_sheet), 1):
        batch = rows[start : start + per_sheet]
        sheet = np.full((args.rows * tile_h, args.columns * tile_w, 3), 248, dtype=np.uint8)
        for index, record in enumerate(batch):
            tile_x = (index % args.columns) * tile_w
            tile_y = (index // args.columns) * tile_h
            crop = cv2.imread(str(args.data / record["plate_crop"]))
            if crop is not None:
                h, w = crop.shape[:2]
                scale = min((tile_w - 16) / max(1, w), 92 / max(1, h))
                crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))))
                rh, rw = crop.shape[:2]
                left = tile_x + (tile_w - rw) // 2
                sheet[tile_y + 4 : tile_y + 4 + rh, left : left + rw] = crop
            cv2.putText(
                sheet,
                f"{record['sample_id']}  source label: {record['annotation_text']}",
                (tile_x + 6, tile_y + 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (20, 90, 20),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet,
                f"previous OCR: {record['audit_ocr_text']}",
                (tile_x + 6, tile_y + 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (20, 20, 210),
                1,
                cv2.LINE_AA,
            )
        out = output_dir / f"mismatch_{page:02d}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(out)


if __name__ == "__main__":
    main()
