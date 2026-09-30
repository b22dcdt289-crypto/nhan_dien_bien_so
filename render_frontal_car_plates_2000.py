"""Render frontal car-plate images from existing, distinct source labels.

The output is synthetic. No claim is made that the generated pixels are a
perspective-corrected photograph or that source filename labels are all right.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from prepare_two_row_frontal_dense import identity_split


SOURCE_ROOT = Path("data/OCR/OCR/images/train(1)/detection")
TWO_ROW_PREPARED = Path("data/two_row_frontal_dense_v1/manifest.json")
TWO_ROW_CURATED = Path("data/two_row_front_curated_2000_v2/manifest.csv")
ONE_ROW_PREPARED = Path("data/one_row_frontal_1000_v1/manifest.json")
OUTPUT = Path("data/car_frontal_synthetic_2x1000_v1")
FONT_REGULAR = Path("C:/Windows/Fonts/ARIALN.TTF")
FONT_BOLD = Path("C:/Windows/Fonts/arialn_b.ttf")
LABEL_PATTERN = re.compile(r"^[0-9]{2}[ABCDEFGHKLMNPSTUVXYZ][0-9]{5}$")
SCALE = 3


def label_valid(text: str) -> bool:
    return bool(LABEL_PATTERN.fullmatch(text)) and identity_split(text) == "train"


def candidates(root: Path, curated: Path, prepared_two: Path, prepared_one: Path) -> tuple[list[dict], list[dict], dict]:
    two_by_text: dict[str, dict] = {}
    one_by_text: dict[str, dict] = {}
    source_count = Counter()

    # Highest preference: images already selected as real two-row crops.
    with curated.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = row["text_from_source_filename"]
            if row["group"] != "two_row_car" or not label_valid(text):
                continue
            path = root / row["original_source"]
            if path.is_file() and text not in two_by_text:
                two_by_text[text] = {"text": text, "source": path, "source_tier": "curated_real_crop"}
                source_count["two_row_curated"] += 1

    # Supplement with earlier successfully segmented two-row source photos.
    for row in json.loads(prepared_two.read_text(encoding="utf-8")):
        text = row["text"]
        if (row["type"] != "two_row_car" or row["split"] != "train"
                or not row["ground_truth_segmentation_ok"] or not label_valid(text)
                or text in two_by_text):
            continue
        path = root / row["source"]
        if path.is_file():
            two_by_text[text] = {"text": text, "source": path, "source_tier": "two_row_prepared"}
            source_count["two_row_prepared_additional"] += 1

    # Prefer previously prepared frontal one-row examples where available.
    for row in json.loads(prepared_one.read_text(encoding="utf-8")):
        text = row["text"]
        if row["split"] != "train" or not label_valid(text) or text in one_by_text:
            continue
        path = Path(row["source"])
        if path.is_file():
            one_by_text[text] = {"text": text, "source": path, "source_tier": "one_row_prepared"}
            source_count["one_row_prepared"] += 1

    # Supplement by labels in the original one-row folder. No photo pixels
    # are used to render, but each label retains a traceable source photo.
    for path in sorted((root / "one_row").glob("*.jpg")):
        parts = path.stem.split("_")
        if len(parts) < 8:
            continue
        text = parts[3].upper()
        if label_valid(text) and text not in one_by_text:
            one_by_text[text] = {"text": text, "source": path, "source_tier": "original_one_row_filename"}
            source_count["one_row_original_additional"] += 1
    return list(two_by_text.values()), list(one_by_text.values()), dict(source_count)


def pick(rows: list[dict], count: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    exclude = exclude or set()
    tier_priority = {"curated_real_crop": 0, "one_row_prepared": 0,
                     "two_row_prepared": 1, "original_one_row_filename": 1}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["text"] not in exclude:
            groups.setdefault(row["text"][:3], []).append(row)
    for prefix, group in groups.items():
        group.sort(key=lambda row: (
            tier_priority[row["source_tier"]],
            hashlib.sha256(f"{seed}:{prefix}:{row['text']}".encode("ascii")).hexdigest(),
        ))
    if sum(map(len, groups.values())) < count:
        raise RuntimeError(f"Only {sum(map(len, groups.values()))} distinct eligible labels; need {count}")
    prefixes = sorted(groups, key=lambda prefix: hashlib.sha256(f"{seed}:{prefix}".encode()).hexdigest())
    selected = []
    depth = 0
    while len(selected) < count:
        for prefix in prefixes:
            if depth < len(groups[prefix]):
                selected.append(groups[prefix][depth])
                if len(selected) == count:
                    break
        depth += 1
    return selected


def draw_fitted(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font_file: Path,
                fill: int, min_size: int = 18) -> None:
    left, top, right, bottom = [value * SCALE for value in box]
    max_width, max_height = right - left, bottom - top
    chosen = None
    for size in range(max_height * 2, min_size * SCALE - 1, -1):
        font = ImageFont.truetype(str(font_file), size)
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=0)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if width <= max_width and height <= max_height:
            chosen = (font, bounds, width, height)
            break
    if chosen is None:
        raise RuntimeError(f"Text cannot fit: {text} in {box}")
    font, bounds, width, height = chosen
    x = left + (max_width - width) // 2 - bounds[0]
    y = top + (max_height - height) // 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def render(text: str, layout: str, seed: int) -> tuple[Image.Image, str]:
    if not LABEL_PATTERN.fullmatch(text):
        raise ValueError(f"Unsupported canonical plate text: {text}")
    two = layout == "two_row_car"
    width, height = (136, 144) if two else (312, 74)
    rng = random.Random(int(hashlib.sha256(f"{seed}|{layout}|{text}".encode()).hexdigest()[:16], 16))
    background = rng.randint(239, 249)
    face = rng.randint(236, 249)
    ink = rng.randint(19, 37)
    image = Image.new("L", (width * SCALE, height * SCALE), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2 * SCALE, 2 * SCALE, (width - 3) * SCALE, (height - 3) * SCALE),
        radius=(6 if two else 5) * SCALE, fill=face, outline=rng.randint(35, 64), width=2 * SCALE,
    )
    draw.rounded_rectangle(
        (5 * SCALE, 5 * SCALE, (width - 6) * SCALE, (height - 6) * SCALE),
        radius=(4 if two else 3) * SCALE, outline=rng.randint(151, 185), width=1 * SCALE,
    )
    font = FONT_REGULAR if rng.random() < 0.7 else FONT_BOLD
    if two:
        display = f"{text[:3]} / {text[3:6]}.{text[6:]}"
        draw_fitted(draw, text[:3], (10, 22, 126, 65), font, ink)
        draw_fitted(draw, f"{text[3:6]}.{text[6:]}", (7, 74, 129, 132), font, ink)
    else:
        display = f"{text[:3]}-{text[3:6]}.{text[6:]}"
        draw_fitted(draw, display, (9, 9, 303, 64), font, ink)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.22:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.18))
    array = np.asarray(image, dtype=np.float32).copy()
    noise = np.random.default_rng(int(hashlib.sha256(f"noise:{seed}:{layout}:{text}".encode()).hexdigest()[:16], 16))
    array += noise.normal(0, rng.uniform(1.0, 2.1), size=array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
    return image, display


def contact_sheet(output: Path, folder: str, rows: list[dict], seed: int) -> None:
    sample = random.Random(seed + (1 if folder == "two_row_car" else 2)).sample(rows, min(30, len(rows)))
    tile_w, tile_h = (160, 180) if folder == "two_row_car" else (340, 110)
    sheet = Image.new("RGB", (5 * tile_w, 6 * tile_h), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT_REGULAR), 16)
    for index, row in enumerate(sample):
        photo = Image.open(output / row["file"]).convert("RGB")
        x = (index % 5) * tile_w + (tile_w - photo.width) // 2
        y = (index // 5) * tile_h + 4
        sheet.paste(photo, (x, y))
        draw.text(((index % 5) * tile_w + 7, (index // 5 + 1) * tile_h - 27),
                  row["id"], font=font, fill="#333333")
    sheet.save(output / f"review_{folder}.jpg", quality=91)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count-per-layout", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output is non-empty; choose a new path: {args.output}")
    for font in (FONT_REGULAR, FONT_BOLD):
        if not font.is_file():
            raise FileNotFoundError(font)

    two, one, source_counts = candidates(args.source_root, TWO_ROW_CURATED, TWO_ROW_PREPARED, ONE_ROW_PREPARED)
    selected_two = pick(two, args.count_per_layout, args.seed)
    selected_one = pick(one, args.count_per_layout, args.seed + 1, {row["text"] for row in selected_two})
    selected = {"two_row_car": selected_two, "one_row_car": selected_one}
    output_rows: list[dict] = []
    for layout, rows in selected.items():
        (args.output / layout).mkdir(parents=True, exist_ok=True)
        layout_rows = []
        for index, row in enumerate(rows, start=1):
            image, display = render(row["text"], layout, args.seed)
            image_id = f"{'C2' if layout == 'two_row_car' else 'C1'}_{index:04d}"
            target = args.output / layout / f"{image_id}.png"
            image.save(target, format="PNG")
            with Image.open(target) as saved:
                if saved.mode != "L" or saved.size != image.size or not np.array_equal(np.asarray(saved), np.asarray(image)):
                    raise RuntimeError(f"Pixel verification failed: {target}")
            info = {
                "id": image_id, "file": target.relative_to(args.output).as_posix(),
                "layout": layout, "label": row["text"], "printed_text": display,
                "image_kind": "synthetic_frontal_from_existing_label",
                "source_image": row["source"].as_posix(), "source_tier": row["source_tier"],
                "width": image.width, "height": image.height,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            output_rows.append(info)
            layout_rows.append(info)
        contact_sheet(args.output, layout, layout_rows, args.seed)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "count_by_layout": {layout: len(rows) for layout, rows in selected.items()},
        "unique_labels_total": len({row["label"] for row in output_rows}),
        "source_tier_by_layout": {layout: dict(Counter(row["source_tier"] for row in rows)) for layout, rows in selected.items()},
        "available_label_counts": {"two_row_car": len(two), "one_row_car": len(one)},
        "source_inventory": source_counts,
        "prefix_count_by_layout": {layout: len({row["text"][:3] for row in rows}) for layout, rows in selected.items()},
        "format": "8 alphanumeric characters from source, NNANNNNN; displayed two-row as NNA / NNN.NN or one-row as NNA-NNN.NN",
        "image_kind": "Synthetic frontal grayscale PNG, never a real photo",
        "pixel_verification": "All output PNGs reopened and compared pixel-for-pixel",
        "warning": "Source labels come from filenames and are not all visually audited. Synthetic-only evaluation cannot establish real-world OCR accuracy.",
        "seed": args.seed,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
