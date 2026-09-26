from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes, run_epoch


SOURCE_ROOT = Path("data/OCR/OCR/images/train(1)/detection")
DATA_ROOT = Path("data/independent_chars_train1")
METRICS_PATH = Path("artifacts/independent_pipeline_metrics.json")
CHAR_SET = set(CLASS_NAMES)


def parse_source_name(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 8:
        return None
    text = parts[3].upper()
    if not text or any(ch not in CHAR_SET for ch in text):
        return None
    try:
        x1, y1, x2, y2 = (int(value) for value in parts[4:8])
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if path.parent.name == "one_row":
        plate_type = "one_row"
    elif path.parent.name == "two_rows_label_xe_may":
        plate_type = "two_row_motorcycle"
    else:
        plate_type = "two_row_car"
    return text, x1, y1, x2, y2, plate_type


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    return np.array(
        [points[np.argmin(total)], points[np.argmin(diff)], points[np.argmax(total)], points[np.argmax(diff)]],
        dtype=np.float32,
    )


def find_plate_quad(gray: np.ndarray):
    """Find a plate-like quadrilateral inside the supplied plate crop."""
    h, w = gray.shape[:2]
    area = float(h * w)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    masks = [
        cv2.Canny(blurred, 40, 160),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    best = None
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < area * 0.25:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 12:
                continue
            ratio = max(rw, rh) / max(1.0, min(rw, rh))
            if ratio < 1.2 or ratio > 8.0:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            margin = max(2.0, min(h, w) * 0.01)
            if np.any(box[:, 0] < -margin) or np.any(box[:, 0] > w + margin):
                continue
            if np.any(box[:, 1] < -margin) or np.any(box[:, 1] > h + margin):
                continue
            score = contour_area / area
            if best is None or score > best[0]:
                best = (score, box)
    return None if best is None else order_points(best[1])


def perspective_correct(plate: np.ndarray):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
    quad = find_plate_quad(gray)
    if quad is None:
        return gray, False, 0.0
    width = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    height = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    if width < 40 or height < 15:
        return gray, False, 0.0
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(gray, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    angle = float(np.degrees(np.arctan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0])))
    return warped, True, abs(angle)


def estimate_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def remove_duplicate_boxes(boxes):
    kept = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        x, y, w, h = box
        duplicate = False
        for px, py, pw, ph in kept:
            ix1, iy1 = max(x, px), max(y, py)
            ix2, iy2 = min(x + w, px + pw), min(y + h, py + ph)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = w * h + pw * ph - inter
            if union and inter / union > 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def merge_fragment_boxes(boxes):
    """Merge split parts of one glyph while keeping adjacent glyphs separate."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            ax, ay, aw, ah = boxes[i]
            for j in range(i + 1, len(boxes)):
                bx, by, bw, bh = boxes[j]
                vertical = max(0, min(ay + ah, by + bh) - max(ay, by)) / max(1, min(ah, bh))
                overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                if vertical > 0.60 and (overlap > min(aw, bw) * 0.25 or gap <= min(ah, bh) * 0.04):
                    x1, y1 = min(ax, bx), min(ay, by)
                    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = (x1, y1, x2 - x1, y2 - y1)
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def classify_and_sort_rows(boxes, height: int):
    if not boxes:
        return [], 0
    boxes = sorted(boxes, key=lambda item: item[1] + item[3] / 2.0)
    centers = np.asarray([y + h / 2.0 for _, y, _, h in boxes], dtype=np.float32)
    if len(boxes) < 4:
        return [sorted(boxes, key=lambda item: item[0])], 1
    gaps = np.diff(centers)
    split_index = int(np.argmax(gaps))
    if gaps[split_index] < height * 0.18:
        return [sorted(boxes, key=lambda item: item[0])], 1
    top = sorted(boxes[: split_index + 1], key=lambda item: item[0])
    bottom = sorted(boxes[split_index + 1 :], key=lambda item: item[0])
    if len(top) < 2 or len(bottom) < 2:
        return [sorted(boxes, key=lambda item: item[0])], 1
    return [top, bottom], 2


def enhance_plate_for_ocr(gray: np.ndarray, method: str = "clahe_sharp") -> np.ndarray:
    """Normalize local contrast and selectively sharpen strokes before segmentation."""
    if method == "none":
        return gray
    if method not in {"clahe", "clahe_sharp"}:
        raise ValueError(f"Unknown plate enhancement method: {method}")
    # Mild edge-preserving denoising avoids boosting sensor/compression noise.
    denoised = cv2.bilateralFilter(gray, 5, 25, 25)
    normalized = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(denoised)
    if method == "clahe":
        return normalized
    low_frequency = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(normalized, 1.35, low_frequency, -0.35, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def segment_characters(
    plate: np.ndarray,
    expected_count: int | None = None,
    enhancement: str = "none",
):
    gray = plate if plate.ndim == 2 else cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    height = 160
    width = max(80, int(round(gray.shape[1] * height / max(1, gray.shape[0]))))
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = enhance_plate_for_ocr(gray, enhancement)
    if enhancement == "none":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates = []
    masks = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8),
    ]
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if h < height * 0.22 or h > height * 0.95:
                continue
            if w < width * 0.012 or w > width * 0.40:
                continue
            if area < height * width * 0.001:
                continue
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                continue
            boxes.append((x, y, w, h))
        boxes = remove_duplicate_boxes(boxes)
        rows, row_count = classify_and_sort_rows(boxes, height)
        ordered = [box for row in rows for box in row]
        if not ordered:
            continue
        count = len(ordered)
        count_penalty = abs(count - expected_count) if expected_count is not None else abs(count - 8) * 0.15
        spacing = np.diff(sorted(x for x, _, _, _ in ordered))
        spacing_penalty = 0.0 if len(spacing) == 0 else float(np.std(spacing) / max(1.0, np.mean(spacing)))
        score = count_penalty + spacing_penalty
        candidates.append((score, ordered, rows, gray))
    if not candidates:
        return None
    if expected_count is not None:
        exact = [candidate for candidate in candidates if len(candidate[1]) == expected_count]
        if exact:
            candidates = exact
    score, boxes, rows, gray = min(candidates, key=lambda item: item[0])
    if expected_count is not None and len(boxes) != expected_count:
        return None
    crops = []
    for x, y, w, h in boxes:
        crop = gray[max(0, y - 3) : min(gray.shape[0], y + h + 3), max(0, x - 3) : min(gray.shape[1], x + w + 3)]
        crop = ImageOps.pad(Image.fromarray(crop, mode="L"), (32, 32), color=0, centering=(0.5, 0.5))
        crops.append(np.asarray(crop, dtype=np.uint8))
    return crops, len(rows), len(boxes)


def source_files(root: Path):
    paths = []
    for folder in ("one_row", "two_rows", "two_rows_label_xe_may"):
        paths.extend(sorted((root / folder).glob("*.jpg")))
    return paths


def condition_bucket(counter: dict, name: str):
    item = counter.setdefault(name, {"images": 0, "skipped": 0})
    return item


def prepare_dataset(args):
    data_root = args.data
    if args.clean and data_root.exists():
        shutil.rmtree(data_root)
    for split in ("train", "val"):
        for char in CLASS_NAMES:
            (data_root / split / char).mkdir(parents=True, exist_ok=True)
    random.seed(42)
    paths = source_files(args.source)
    random.shuffle(paths)
    records = []
    reasons = Counter()
    types = Counter()
    accepted_types = Counter()
    condition_stats = {}
    accepted = 0
    total_crops = 0
    for index, path in enumerate(paths, start=1):
        parsed = parse_source_name(path)
        if parsed is None:
            reasons["bad_filename"] += 1
            continue
        text, x1, y1, x2, y2, plate_type = parsed
        types[plate_type] += 1
        image = cv2.imread(str(path))
        if image is None:
            reasons["read_error"] += 1
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        plate = image[y1:y2, x1:x2]
        if plate.size == 0:
            reasons["empty_plate_crop"] += 1
            continue
        raw_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        blur_score = estimate_blur(raw_gray)
        corrected, corrected_ok, angle = perspective_correct(plate)
        angled = angle >= args.angle_threshold
        blurred = blur_score < args.blur_threshold
        for condition, enabled in (("blur", blurred), ("angled", angled)):
            if enabled:
                condition_bucket(condition_stats, condition)["images"] += 1
        segmented = segment_characters(corrected, expected_count=len(text), enhancement=args.enhancement)
        if segmented is None:
            reasons["segmentation_or_count"] += 1
            for condition, enabled in (("blur", blurred), ("angled", angled)):
                if enabled:
                    condition_bucket(condition_stats, condition)["skipped"] += 1
            continue
        crops, row_count, char_count = segmented
        split = "val" if accepted % 10 == 0 else "train"
        crop_files = []
        safe_stem = f"{plate_type}_{path.stem}"
        for char_index, (char, crop) in enumerate(zip(text, crops)):
            out = data_root / split / char / f"{safe_stem}_{char_index:02d}.png"
            cv2.imwrite(str(out), crop)
            crop_files.append(str(out.relative_to(data_root)))
            total_crops += 1
        accepted += 1
        accepted_types[plate_type] += 1
        records.append({
            "split": split,
            "source": str(path.relative_to(args.source)).replace("\\", "/"),
            "type": plate_type,
            "text": text,
            "char_count": char_count,
            "row_count": row_count,
            "perspective_corrected": corrected_ok,
            "angle_deg": round(angle, 3),
            "blur_score": round(blur_score, 3),
            "enhancement": args.enhancement,
            "crop_files": crop_files,
        })
        if index % 1000 == 0:
            print(f"images={index} accepted={accepted} skipped={index - accepted} chars={total_crops}", flush=True)
    for condition in condition_stats.values():
        condition["skip_rate"] = condition["skipped"] / max(1, condition["images"])
    summary = {
        "source": str(args.source),
        "source_images": len(paths),
        "accepted_images": accepted,
        "skipped_images": len(paths) - accepted,
        "skip_rate": (len(paths) - accepted) / max(1, len(paths)),
        "character_crops": total_crops,
        "source_types": dict(types),
        "accepted_types": dict(accepted_types),
        "skip_reasons": dict(reasons),
        "condition_skip_rates": condition_stats,
        "blur_threshold_laplacian_variance": args.blur_threshold,
        "angle_threshold_degrees": args.angle_threshold,
        "perspective_correction": {
            "attempted": accepted,
            "applied": sum(1 for record in records if record["perspective_corrected"]),
        },
        "row_classification": dict(Counter(record["row_count"] for record in records)),
    }
    (data_root / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps({"prepare": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


class FolderChars(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.items = []
        self.augment = augment
        for label, char in enumerate(CLASS_NAMES):
            self.items.extend((path, label) for path in (root / split / char).glob("*.png"))
        # The crops are only 32x32. Keeping uint8 pixels in RAM avoids reopening
        # roughly 200,000 small PNG files on every epoch on Windows.
        def load_item(item):
            path, label = item
            with Image.open(path) as image:
                return np.asarray(image.convert("L"), dtype=np.uint8).copy(), label

        with ThreadPoolExecutor(max_workers=16) as executor:
            self.cache = list(executor.map(load_item, self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pixels, label = self.cache[index]
        image = Image.fromarray(pixels, mode="L")
        if self.augment and random.random() < 0.55:
            image = image.rotate(random.uniform(-6, 6), resample=Image.Resampling.BILINEAR, fillcolor=0)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0), label


class LeNet5Structured50(nn.Module):
    """Physically compact LeNet-5 with approximately 50% fewer dense MACs."""

    channels = (4, 10, 100, 70)

    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        c1, c2, h1, h2 = self.channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, 5), nn.Tanh(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c2 * 5 * 5, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    @classmethod
    def macs(cls):
        c1, c2, h1, h2 = cls.channels
        return 1 * c1 * 25 * 28 * 28 + c1 * c2 * 25 * 10 * 10 + c2 * 25 * h1 + h1 * h2 + h2 * len(CLASS_NAMES)


def top_indices(weight: torch.Tensor, count: int, dim: int = 0):
    reduce_dims = tuple(i for i in range(weight.ndim) if i != dim)
    score = weight.detach().abs().sum(dim=reduce_dims)
    return torch.topk(score, count).indices.sort().values


def transfer_dense_to_structured(dense: LeNet5, small: LeNet5Structured50):
    c1, c2, h1, h2 = small.channels
    conv1, conv2 = dense.features[0], dense.features[3]
    fc1, fc2, fc3 = dense.classifier[0], dense.classifier[2], dense.classifier[4]
    i1 = top_indices(conv1.weight, c1)
    i2 = top_indices(conv2.weight, c2)
    input_cols = i2.repeat_interleave(25)
    ih1 = top_indices(fc1.weight[:, input_cols], h1)
    ih2 = top_indices(fc2.weight[:, ih1], h2)
    with torch.no_grad():
        small.features[0].weight.copy_(conv1.weight[i1])
        small.features[0].bias.copy_(conv1.bias[i1])
        small.features[3].weight.copy_(conv2.weight[i2][:, i1])
        small.features[3].bias.copy_(conv2.bias[i2])
        small.classifier[0].weight.copy_(fc1.weight[ih1][:, input_cols])
        small.classifier[0].bias.copy_(fc1.bias[ih1])
        small.classifier[2].weight.copy_(fc2.weight[ih2][:, ih1])
        small.classifier[2].bias.copy_(fc2.bias[ih2])
        small.classifier[4].weight.copy_(fc3.weight[:, ih2])
        small.classifier[4].bias.copy_(fc3.bias)


def plate_metrics(model, device, data_root: Path):
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest if record["split"] == "val"]
    model.eval()
    total_chars = correct_chars = exact = 0
    by_type = defaultdict(lambda: {"plates": 0, "exact": 0, "chars": 0, "correct": 0})
    with torch.no_grad():
        for record in records:
            tensors = []
            for relative in record["crop_files"]:
                with Image.open(data_root / relative) as image:
                    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
            logits = model(torch.stack(tensors).to(device))
            predictions = [CLASS_NAMES[i] for i in logits.argmax(1).cpu().tolist()]
            target = list(record["text"])
            char_ok = sum(a == b for a, b in zip(predictions, target))
            is_exact = predictions == target
            total_chars += len(target)
            correct_chars += char_ok
            exact += int(is_exact)
            stats = by_type[record["type"]]
            stats["plates"] += 1
            stats["exact"] += int(is_exact)
            stats["chars"] += len(target)
            stats["correct"] += char_ok
    return {
        "validation_source_images": len(records),
        "character_accuracy": correct_chars / max(1, total_chars),
        "plate_exact_match": exact / max(1, len(records)),
        "by_type": {
            key: {
                "plates": value["plates"],
                "plate_exact_match": value["exact"] / max(1, value["plates"]),
                "character_accuracy": value["correct"] / max(1, value["chars"]),
            }
            for key, value in by_type.items()
        },
    }


def train_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(42)
    torch.manual_seed(42)
    train_set = FolderChars(args.data, "train", augment=True)
    val_set = FolderChars(args.data, "val", augment=False)
    if not train_set or not val_set:
        raise RuntimeError(f"Character dataset is empty: {args.data}. Run --mode prepare first.")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dense = LeNet5(len(CLASS_NAMES)).to(device)
    if args.dense_init and args.dense_init.exists():
        checkpoint = torch.load(args.dense_init, map_location=device, weights_only=False)
        require_compatible_classes(checkpoint, args.dense_init)
        dense.load_state_dict(checkpoint["model"])
        print(f"dense_init={args.dense_init}", flush=True)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=args.dense_lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    best_dense = 0.0
    print(f"device={device} train_crops={len(train_set)} val_crops={len(val_set)}", flush=True)
    for epoch in range(1, args.dense_epochs + 1):
        train_loss, train_acc = run_epoch(dense, train_loader, loss_fn, dense_opt, device, True)
        val_loss, val_acc = run_epoch(dense, val_loader, loss_fn, dense_opt, device, False)
        print(f"dense epoch {epoch:02d}/{args.dense_epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_dense:
            best_dense = val_acc
            torch.save({"model": dense.state_dict(), "classes": CLASS_NAMES, "val_acc": val_acc, "arch": "LeNet5", "enhancement": args.enhancement}, args.dense_output)
    small = LeNet5Structured50(len(CLASS_NAMES)).to(device)
    transfer_dense_to_structured(dense, small)
    small_opt = torch.optim.AdamW(small.parameters(), lr=args.structured_lr, weight_decay=1e-5)
    best_small = 0.0
    for epoch in range(1, args.structured_epochs + 1):
        train_loss, train_acc = run_epoch(small, train_loader, loss_fn, small_opt, device, True)
        val_loss, val_acc = run_epoch(small, val_loader, loss_fn, small_opt, device, False)
        print(f"structured epoch {epoch:02d}/{args.structured_epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_small:
            best_small = val_acc
            torch.save({
                "model": small.state_dict(), "classes": CLASS_NAMES, "val_acc": val_acc,
                "arch": "LeNet5Structured50", "channels": small.channels, "macs_per_character": small.macs(),
                "enhancement": args.enhancement,
            }, args.output)
    small.load_state_dict(torch.load(args.output, map_location=device, weights_only=False)["model"])
    end_to_end = plate_metrics(small, device, args.data)
    prepare_metrics = {}
    if args.metrics.exists():
        prepare_metrics = json.loads(args.metrics.read_text(encoding="utf-8")).get("prepare", {})
    dense_macs = 418200
    metrics = {
        "source": str(args.source),
        "data_root": str(args.data),
        "dense_validation_character_accuracy": best_dense,
        "structured_validation_character_accuracy": best_small,
        "structured_channels": list(small.channels),
        "dense_macs_per_character": dense_macs,
        "structured_macs_per_character": small.macs(),
        "mac_reduction_ratio": 1.0 - small.macs() / dense_macs,
        "structured_macs_per_plate": {str(count): small.macs() * count for count in (8, 9, 10)},
        "end_to_end_validation": end_to_end,
        "prepare": prepare_metrics,
        "checkpoint": str(args.output),
        "enhancement": args.enhancement,
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def evaluate_checkpoint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, args.eval_checkpoint)
    model = LeNet5Structured50(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    end_to_end = plate_metrics(model, device, args.data)
    prepare_metrics = json.loads(args.metrics.read_text(encoding="utf-8")).get("prepare", {})
    dense_accuracy = None
    if args.dense_output.exists():
        dense_accuracy = torch.load(args.dense_output, map_location="cpu", weights_only=False).get("val_acc")
    dense_macs = 418200
    metrics = {
        "source": str(args.source), "data_root": str(args.data),
        "dense_validation_character_accuracy": dense_accuracy,
        "structured_validation_character_accuracy": checkpoint.get("val_acc"),
        "structured_channels": list(model.channels),
        "dense_macs_per_character": dense_macs,
        "structured_macs_per_character": model.macs(),
        "mac_reduction_ratio": 1.0 - model.macs() / dense_macs,
        "structured_macs_per_plate": {str(count): model.macs() * count for count in (8, 9, 10)},
        "end_to_end_validation": end_to_end,
        "prepare": prepare_metrics,
        "checkpoint": str(args.eval_checkpoint),
        "enhancement": checkpoint.get("enhancement", "none"),
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Independent license plate pipeline without YOLOv5")
    parser.add_argument("--mode", choices=("prepare", "train", "eval", "all"), default="all")
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--enhancement", choices=("none", "clahe", "clahe_sharp"), default="none")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--angle-threshold", type=float, default=8.0)
    parser.add_argument("--dense-epochs", type=int, default=5)
    parser.add_argument("--structured-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dense-lr", type=float, default=5e-4)
    parser.add_argument("--structured-lr", type=float, default=2e-4)
    parser.add_argument("--dense-init", type=Path, default=None)
    parser.add_argument("--dense-output", type=Path, default=Path("artifacts/independent_lenet5_dense_train1.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/independent_lenet5_structured50_train1.pt"))
    parser.add_argument("--eval-checkpoint", type=Path, default=Path("artifacts/independent_lenet5_structured50_train1.pt"))
    args = parser.parse_args()
    if args.mode in ("prepare", "all"):
        prepare_dataset(args)
    if args.mode in ("train", "all"):
        train_models(args)
    if args.mode == "eval":
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
