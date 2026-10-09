"""Render 1,000 two-row motorcycle plates from train-split labels only.

These are synthetic images, not corrected photographs. Source filename labels
have not all been manually verified. No val/test source image is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from prepare_two_row_frontal_dense import identity_split
from render_frontal_car_plates_2000 import FONT_BOLD, FONT_REGULAR, SCALE, draw_fitted
from train_lenet5 import CLASS_NAMES
from train_synthetic_car_compare import synthetic_segment


def render_motorcycle(text: str, seed: int) -> Image.Image:
    if len(text) != 9 or any(char not in CLASS_NAMES for char in text):
        raise ValueError(text)
    width, height = 160, 144
    rng = random.Random(int(hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()[:16], 16))
    image = Image.new("L", (width * SCALE, height * SCALE), rng.randint(239, 249))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2 * SCALE, 2 * SCALE, (width - 3) * SCALE, (height - 3) * SCALE),
                           radius=6 * SCALE, fill=rng.randint(236, 249),
                           outline=rng.randint(35, 64), width=2 * SCALE)
    draw.rounded_rectangle((5 * SCALE, 5 * SCALE, (width - 6) * SCALE, (height - 6) * SCALE),
                           radius=4 * SCALE, outline=rng.randint(151, 185), width=SCALE)
    font = FONT_REGULAR if rng.random() < 0.7 else FONT_BOLD
    ink = rng.randint(19, 37)
    draw_fitted(draw, text[:4], (9, 20, 151, 65), font, ink)
    draw_fitted(draw, text[4:7] + "." + text[7:], (8, 74, 152, 132), font, ink)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.22:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.18))
    pixels = np.asarray(image, dtype=np.float32).copy()
    noise = np.random.default_rng(int(hashlib.sha256(f"noise:{seed}:{text}".encode()).hexdigest()[:16], 16))
    pixels += noise.normal(0, rng.uniform(1.0, 2.1), size=pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/OCR/OCR/images/train(1)/detection/two_rows_label_xe_may"))
    parser.add_argument("--output", type=Path, default=Path("data/motorcycle_frontal_synthetic_1000_v2"))
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    candidates = {}
    for path in sorted(args.source.glob("*.jpg")):
        parts = path.stem.split("_")
        if len(parts) < 8:
            continue
        text = parts[3].upper()
        if len(text) == 9 and identity_split(text) == "train" and all(char in CLASS_NAMES for char in text):
            candidates.setdefault(text, {"text": text, "source": path.as_posix()})
    ranked = sorted(candidates.values(), key=lambda row: hashlib.sha256(
        f"{args.seed}|{row['text']}".encode()).hexdigest())
    if len(ranked) < 1000:
        raise RuntimeError(f"Only {len(ranked)} distinct eligible labels")
    args.output.mkdir(parents=True)
    rows = []
    rejected_segmentation = 0
    for record in ranked:
        if len(rows) == 1000:
            break
        image = render_motorcycle(record["text"], args.seed)
        if synthetic_segment(np.asarray(image), "two_row_motorcycle") is None:
            rejected_segmentation += 1
            continue
        index = len(rows) + 1
        target = args.output / f"M2_{index:04d}.png"
        image.save(target)
        rows.append({"id": target.stem, "file": target.name, "layout": "two_row_motorcycle",
                     "label": record["text"], "source_image": record["source"],
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"count": len(rows), "unique_labels": len({row['label'] for row in rows}),
               "label_length": 9, "image_kind": "synthetic_frontal_from_existing_label",
               "labels_manually_verified": False, "source_folder": str(args.source),
               "rejected_unsegmentable": rejected_segmentation,
               "source_split": "train", "seed": args.seed}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
