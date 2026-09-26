from __future__ import annotations

import random
import re
import argparse
from pathlib import Path

import cv2
from train_lenet5 import CLASS_NAMES


ROOT = Path("data/VNLP/detection")
OUT = Path("data/VNLP_chars_corrected")
CHARS = set(CLASS_NAMES)


def parse_name(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 8:
        return None
    text = parts[3].upper()
    if not text or any(ch not in CHARS for ch in text):
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in parts[4:8])
    except ValueError:
        return None
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    if w < 10 or h < 10:
        return None
    return text, x, y, w, h


def order_boxes(boxes, height):
    if not boxes:
        return []
    centers = sorted((y + h / 2 for _, y, _, h in boxes))
    if len(boxes) < 2 or centers[-1] - centers[0] < height * 0.22:
        return sorted(boxes, key=lambda b: b[0])
    split = (centers[0] + centers[-1]) / 2
    top = [b for b in boxes if b[1] + b[3] / 2 <= split]
    bottom = [b for b in boxes if b[1] + b[3] / 2 > split]
    return sorted(top, key=lambda b: b[0]) + sorted(bottom, key=lambda b: b[0])


def find_character_boxes(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[:2, :] = binary[-2:, :] = 0
    binary[:, :2] = binary[:, -2:] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch >= h * 0.22 and ch <= h * 0.95 and cw >= w * 0.015 and cw <= w * 0.35:
            boxes.append((x, y, cw, ch))
    return order_boxes(boxes, h), gray


def main():
    parser = argparse.ArgumentParser(description="Create character folders from VNLP filename labels")
    parser.add_argument("--root", type=Path, default=ROOT, help="VNLP detection directory")
    parser.add_argument("--out", type=Path, default=OUT, help="output character-folder dataset")
    args = parser.parse_args()
    random.seed(42)
    for split in ("train", "val"):
        for char in CHARS:
            (args.out / split / char).mkdir(parents=True, exist_ok=True)
    paths = sorted(
        list((args.root / "one_row").glob("*.jpg"))
        + list((args.root / "two_rows").glob("*.jpg"))
        + list((args.root / "two_rows_label_xe_may").glob("*.jpg"))
    )
    random.shuffle(paths)
    accepted = skipped = saved = 0
    for index, path in enumerate(paths, 1):
        parsed = parse_name(path)
        if parsed is None:
            skipped += 1
            continue
        text, x, y, w, h = parsed
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        x, y = max(0, x), max(0, y)
        plate = image[y : y + h, x : x + w]
        boxes, gray = find_character_boxes(plate)
        if len(boxes) != len(text):
            skipped += 1
            continue
        split = "val" if accepted % 10 == 0 else "train"
        for char, (bx, by, bw, bh) in zip(text, boxes):
            crop = gray[max(0, by - 2) : min(gray.shape[0], by + bh + 2), max(0, bx - 2) : min(gray.shape[1], bx + bw + 2)]
            crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            out = args.out / split / char / f"{accepted:06d}_{saved:03d}.png"
            cv2.imwrite(str(out), crop)
            saved += 1
        accepted += 1
        if index % 1000 == 0:
            print(f"images={index} accepted={accepted} skipped={skipped} chars={saved}", flush=True)
    print(f"done images={len(paths)} accepted={accepted} skipped={skipped} chars={saved}", flush=True)


if __name__ == "__main__":
    main()
