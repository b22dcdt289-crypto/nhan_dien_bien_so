from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from prepare_frontal_car_1000 import candidate_from_image
from train_independent_structured import estimate_blur, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes


DEFAULT_OUTPUT = Path("data/one_row_combined_frontal_1000_perspective_v1")
DEFAULT_PRIOR_DATA = Path("data/one_row_frontal_1000_v1")
DEFAULT_TEACHER = Path("artifacts/lenet5_dense_front_one_row_1000_30class.pt")
DEFAULT_VISUAL_CORRECTIONS = Path("data/legacy_visual_label_corrections.json")
SOURCE_CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_SET = set(CLASS_NAMES)


def load_visual_corrections(path: Path) -> dict[str, str]:
    """Load private, manually reviewed pseudo-label fixes kept outside version control."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Visual corrections must be a JSON object: {path}")
    corrections = {str(key): str(value).upper() for key, value in payload.items()}
    invalid = {
        key: value for key, value in corrections.items()
        if not value or any(character not in CLASS_SET for character in value)
    }
    if invalid:
        raise ValueError(f"Invalid visual corrections in {path}: {sorted(invalid)}")
    return corrections


def load_teacher(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, checkpoint_path)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def classify_crops(model, device: torch.device, crops: list[np.ndarray]):
    tensors = []
    for crop in crops:
        pixels = crop.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
    with torch.inference_mode():
        probs = torch.softmax(model(torch.stack(tensors).to(device)), dim=1)
        confidences, indices = probs.max(dim=1)
    text = "".join(CLASS_NAMES[index] for index in indices.cpu().tolist())
    return text, confidences.cpu().tolist()


def prior_labels(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Previous split manifest not found: {manifest_path}")
    return {str(row["text"]).upper() for row in json.loads(manifest_path.read_text(encoding="utf-8"))}


def legacy_annotation_text(label_path: Path):
    """Return annotation text sorted left-to-right; this is audited, never trusted as truth."""
    if not label_path.is_file():
        return None
    items = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            source_class = int(fields[0])
            center_x = float(fields[1])
            if not 0 <= source_class < len(SOURCE_CLASS_NAMES):
                continue
            character = SOURCE_CLASS_NAMES[source_class]
            if character in CLASS_SET:
                items.append((center_x, character))
    except (OSError, ValueError):
        return None
    return "".join(character for _, character in sorted(items))


def legacy_annotation_box_count(label_path: Path):
    if not label_path.is_file():
        return None
    count = 0
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 5:
                int(fields[0])
                [float(value) for value in fields[1:]]
                count += 1
    except (OSError, ValueError):
        return None
    return count


def legacy_candidate(path: Path, args, model, device: torch.device, reasons: Counter, visual_corrections: dict[str, str]):
    image = cv2.imread(str(path))
    if image is None:
        reasons["legacy_image_read_error"] += 1
        return None
    height, width = image.shape[:2]
    if width < args.min_width or height < args.min_height:
        reasons["legacy_low_resolution"] += 1
        return None
    aspect = width / max(1, height)
    if not args.min_aspect <= aspect <= args.max_aspect:
        reasons["legacy_aspect_filter"] += 1
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = estimate_blur(gray)
    if blur < args.legacy_min_blur:
        reasons["legacy_blur_filter"] += 1
        return None
    contrast = float(gray.std())
    if contrast < args.min_contrast:
        reasons["legacy_contrast_filter"] += 1
        return None

    rectified, corrected, angle = perspective_correct(image)
    if angle > args.max_angle:
        reasons["legacy_roll_filter"] += 1
        return None
    first = segment_characters(rectified, expected_count=None, enhancement="clahe_sharp")
    second = segment_characters(rectified, expected_count=None, enhancement="clahe")
    if first is None or second is None:
        reasons["legacy_segmentation_failure"] += 1
        return None
    crops, rows, count = first
    crops2, rows2, count2 = second
    if rows != 1 or rows2 != 1 or count not in (7, 8, 9) or count2 != count:
        reasons["legacy_not_one_row_or_unsupported_count"] += 1
        return None
    text, confidences = classify_crops(model, device, crops)
    text2, confidences2 = classify_crops(model, device, crops2)
    if text != text2:
        reasons["legacy_preprocessing_disagreement"] += 1
        return None
    teacher_text = text
    confidence_min = min(confidences + confidences2)
    if confidence_min < args.min_pseudo_confidence:
        reasons["legacy_low_pseudo_confidence"] += 1
        return None

    label_path = args.legacy_labels / f"{path.stem}.txt"
    annotation_box_count = legacy_annotation_box_count(label_path)
    if annotation_box_count is not None and annotation_box_count != count:
        reasons["legacy_annotation_box_count_mismatch"] += 1
        return None
    annotation = legacy_annotation_text(label_path)
    if path.stem in visual_corrections:
        text = visual_corrections[path.stem]
        if len(text) != count or any(character not in CLASS_SET for character in text):
            reasons["legacy_visual_correction_length_or_vocabulary_error"] += 1
            return None
        reasons["legacy_visual_label_manually_corrected"] += 1
    if not label_path.is_file():
        reasons["legacy_missing_yolo_label_file"] += 1
    elif annotation is None:
        reasons["legacy_unreadable_yolo_label_file"] += 1
    else:
        reasons["legacy_yolo_label_files_audited"] += 1
        if annotation != text:
            reasons["legacy_yolo_annotation_disagrees_with_teacher"] += 1

    reasons["legacy_pseudo_label_accepted"] += 1
    return {
        "text": text,
        "teacher_predicted_text": teacher_text,
        "source_text": annotation,
        "source_annotation_box_count": annotation_box_count,
        "source_path": path,
        "plate": image,
        "rectified": rectified,
        "crops": crops,
        "angle": float(angle),
        "blur": float(blur),
        "contrast": contrast,
        "aspect_ratio": float(aspect),
        "perspective_applied": bool(corrected),
        "pseudo_min_confidence": float(confidence_min),
        "pseudo_mean_confidence": float(np.mean(confidences + confidences2)),
        "label_source": (
            "previous_dense_lenet_dual_preprocessing_consensus_with_manual_visual_correction"
            if path.stem in visual_corrections
            else "previous_dense_lenet_dual_preprocessing_consensus"
        ),
    }


def perspective_view(image: np.ndarray, rng: random.Random, strength: float):
    height, width = image.shape[:2]
    max_dx = max(1.0, width * strength)
    max_dy = max(1.0, height * strength * 1.25)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = np.array(
        [
            [rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [width - 1 - rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [width - 1 - rng.uniform(0, max_dx), height - 1 - rng.uniform(0, max_dy)],
            [rng.uniform(0, max_dx), height - 1 - rng.uniform(0, max_dy)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    angle = rng.uniform(-2.5, 2.5)
    rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    warped = cv2.warpAffine(
        warped,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, {"perspective_strength": strength, "roll_deg": round(angle, 3)}


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def source_relative(path: Path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def round_robin_training_ids(identities: list[str], count: int, rng: random.Random):
    buckets: dict[str, list[str]] = defaultdict(list)
    for identity in identities:
        series = identity[2] if len(identity) > 2 and identity[2].isalpha() else "_"
        buckets[series].append(identity)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    names = list(buckets)
    rng.shuffle(names)
    selected = []
    while len(selected) < count:
        progress = False
        for name in names:
            if buckets[name]:
                selected.append(buckets[name].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise RuntimeError(f"Could not select {count} primary-source training plates.")
    return selected


def make_preview(path: Path, triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    if not triplets:
        return
    tile_w, tile_h = 300, 100
    tiles = []
    for index, triplet in enumerate(triplets):
        for label, image in zip(("SOURCE", "PERSPECTIVE A", "PERSPECTIVE B"), triplet):
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tile = cv2.resize(gray, (tile_w, tile_h - 22), interpolation=cv2.INTER_CUBIC)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, f"{index + 1:02d} {label}", (6, tile_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 220, 30), 1, cv2.LINE_AA)
            tiles.append(tile)
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * (tile_h - 22), columns * tile_w, 3), 24, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, columns)
        sheet[y * (tile_h - 22):(y + 1) * (tile_h - 22), x * tile_w:(x + 1) * tile_w] = tile
    save_image(path, sheet)


def save_legacy_review(output: Path, candidates: list[dict]):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Audit output is not empty; choose a fresh directory: {output}")
    tile_w, tile_h = 330, 105
    columns, rows_per_sheet = 4, 5
    for start in range(0, len(candidates), columns * rows_per_sheet):
        sheet_candidates = candidates[start:start + columns * rows_per_sheet]
        sheet = np.full((rows_per_sheet * tile_h, columns * tile_w, 3), 30, dtype=np.uint8)
        for offset, item in enumerate(sheet_candidates):
            image = item["plate"]
            height, width = image.shape[:2]
            scale = min((tile_w - 12) / max(1, width), (tile_h - 34) / max(1, height))
            resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC)
            row, col = divmod(offset, columns)
            x = col * tile_w + (tile_w - resized.shape[1]) // 2
            y = row * tile_h + 24 + max(0, (tile_h - 30 - resized.shape[0]) // 2)
            sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            label = f"{start + offset + 1:03d} {item['text']} p={item['pseudo_min_confidence']:.2f}"
            cv2.putText(sheet, label, (col * tile_w + 5, row * tile_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 240, 80), 1, cv2.LINE_AA)
        save_image(output / f"legacy_train_pseudo_sheet_{start // (columns * rows_per_sheet) + 1:02d}.jpg", sheet)
    with (output / "legacy_train_pseudo_review.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_file", "predicted_text", "min_confidence", "mean_confidence", "legacy_annotation_text"))
        writer.writeheader()
        for item in candidates:
            writer.writerow({
                "source_file": item["source_path"].name,
                "predicted_text": item["text"],
                "min_confidence": item["pseudo_min_confidence"],
                "mean_confidence": item["pseudo_mean_confidence"],
                "legacy_annotation_text": item.get("source_text"),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a 1,000-plate one-row car OCR set from train + train(1), with train-only perspective views."
    )
    parser.add_argument("--primary-root", type=Path, default=Path("data/OCR/OCR/images/train(1)/detection/one_row"))
    parser.add_argument("--legacy-root", type=Path, default=Path("data/OCR/OCR/images/train"))
    parser.add_argument("--legacy-labels", type=Path, default=Path("data/OCR/OCR/labels/train"))
    parser.add_argument("--visual-corrections", type=Path, default=DEFAULT_VISUAL_CORRECTIONS)
    parser.add_argument("--prior-data", type=Path, default=DEFAULT_PRIOR_DATA)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--legacy-max", type=int, default=400)
    parser.add_argument("--min-pseudo-confidence", type=float, default=0.85)
    parser.add_argument("--audit-legacy-only", action="store_true", help="inspect train/CarLongPlate label quality without preparing or training")
    parser.add_argument("--audit-output", type=Path, default=Path("data/legacy_train_pseudo_audit_v3"))
    parser.add_argument("--perspective-views", type=int, default=2)
    parser.add_argument("--perspective-strength", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--min-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=28)
    parser.add_argument("--min-aspect", type=float, default=3.0)
    parser.add_argument("--max-aspect", type=float, default=7.0)
    parser.add_argument("--max-angle", type=float, default=5.0)
    parser.add_argument("--min-blur", type=float, default=80.0)
    parser.add_argument("--legacy-min-blur", type=float, default=0.0)
    parser.add_argument("--min-contrast", type=float, default=25.0)
    parser.add_argument("--enhancement", choices=("clahe", "clahe_sharp"), default="clahe_sharp")
    args = parser.parse_args()

    if args.train_count != 1000:
        raise ValueError("This first-stage pipeline is intentionally fixed at exactly 1,000 training plates.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a fresh directory: {args.output}")
    if args.perspective_views < 0 or args.legacy_max < 1:
        raise ValueError("perspective-views must be nonnegative and legacy-max must be positive.")

    old_ids = prior_labels(args.prior_data / "manifest.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    teacher = load_teacher(args.teacher, device)
    visual_corrections = load_visual_corrections(args.visual_corrections)
    print(f"device={device} prior_plate_identities_excluded={len(old_ids)}", flush=True)

    legacy_files = sorted(path for path in args.legacy_root.glob("*.jpg") if re.match(r"^\d+CarLongPlate\d+$", path.stem, re.IGNORECASE))
    legacy_reasons: Counter = Counter()
    legacy_by_identity: dict[str, dict] = {}
    for path in legacy_files:
        candidate = legacy_candidate(path, args, teacher, device, legacy_reasons, visual_corrections)
        if candidate is None:
            continue
        if candidate["text"] in old_ids:
            legacy_reasons["legacy_identity_seen_in_previous_run_excluded"] += 1
            continue
        incumbent = legacy_by_identity.get(candidate["text"])
        if incumbent is None or candidate["pseudo_mean_confidence"] > incumbent["pseudo_mean_confidence"]:
            legacy_by_identity[candidate["text"]] = candidate

    legacy_audit = {
        "legacy_train_total_images": sum(1 for _ in args.legacy_root.glob("*.jpg")),
        "car_long_plate_candidates": len(legacy_files),
        "passing_candidate_images": len(legacy_by_identity),
        "reasons": dict(legacy_reasons),
    }
    if args.audit_legacy_only and legacy_by_identity:
        review_items = sorted(legacy_by_identity.values(), key=lambda item: item["pseudo_mean_confidence"], reverse=True)
        save_legacy_review(args.audit_output, review_items)
        legacy_audit["manual_review_sheets"] = args.audit_output.as_posix()
    print(json.dumps(legacy_audit, ensure_ascii=False, indent=2), flush=True)
    if args.audit_legacy_only:
        return
    if not legacy_by_identity:
        raise RuntimeError(f"No CarLongPlate sample passed; audit={json.dumps(legacy_audit, ensure_ascii=False)}")

    primary_paths = sorted(args.primary_root.glob("*.jpg"))
    best_primary: dict[str, dict] = {}
    primary_rejected: Counter = Counter()
    for index, path in enumerate(primary_paths, 1):
        parsed = candidate_from_image(path, args, primary_rejected)
        if parsed is None:
            continue
        identity = parsed["text"]
        if identity in old_ids:
            primary_rejected["identity_seen_in_previous_run_excluded"] += 1
            continue
        summary = {key: parsed[key] for key in ("text", "source_text", "source_path", "score")}
        if identity not in best_primary or summary["score"] > best_primary[identity]["score"]:
            best_primary[identity] = summary
        if index % 2000 == 0:
            print(f"scanned_train1={index}/{len(primary_paths)} fresh_readable_identities={len(best_primary)}", flush=True)
    required_eval = args.val_count + args.test_count
    required = args.train_count + required_eval
    if len(best_primary) < required:
        raise RuntimeError(f"Only {len(best_primary)} fresh readable train(1) plates; {required} are required.")

    rng = random.Random(args.seed)
    primary_ids = list(best_primary)
    rng.shuffle(primary_ids)
    val_ids = primary_ids[:args.val_count]
    test_ids = primary_ids[args.val_count:required_eval]
    heldout_ids = set(val_ids) | set(test_ids)
    legacy_training = [
        item for identity, item in legacy_by_identity.items()
        if identity not in heldout_ids and identity not in old_ids
    ]
    legacy_training.sort(key=lambda item: (item["pseudo_mean_confidence"], item["pseudo_min_confidence"]), reverse=True)
    legacy_training = legacy_training[:min(args.legacy_max, args.train_count - 1)]
    if not legacy_training:
        raise RuntimeError(
            "No train-folder CarLongPlate sample remained after identity leakage checks. "
            f"Do not silently call this a mixed-source training run; audit={json.dumps(legacy_audit, ensure_ascii=False)}"
        )
    legacy_ids = {item["text"] for item in legacy_training}
    available_primary_train = [identity for identity in primary_ids[required_eval:] if identity not in legacy_ids]
    primary_train_count = args.train_count - len(legacy_training)
    primary_train_ids = round_robin_training_ids(available_primary_train, primary_train_count, rng)
    split_ids = {"val": val_ids, "test": test_ids, "train": primary_train_ids + sorted(legacy_ids)}
    if any(
        set(split_ids[a]) & set(split_ids[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Plate identity leakage across splits; refusing to write this dataset.")
    if len(split_ids["train"]) != 1000 or len(split_ids["val"]) != 300 or len(split_ids["test"]) != 500:
        raise RuntimeError("The requested 1,000/300/500 split could not be filled exactly.")

    selected: dict[str, dict] = {item["text"]: item for item in legacy_training}
    for split in ("val", "test", "train"):
        if split == "train":
            for identity in primary_train_ids:
                path = best_primary[identity]["source_path"]
                candidate = candidate_from_image(path, args, Counter())
                if candidate is None:
                    raise RuntimeError(f"Selected primary image failed the repeated quality check: {path.name}")
                candidate["label_source"] = "train1_filename_annotation"
                selected[identity] = candidate
        else:
            for identity in split_ids[split]:
                path = best_primary[identity]["source_path"]
                candidate = candidate_from_image(path, args, Counter())
                if candidate is None:
                    raise RuntimeError(f"Selected held-out image failed the repeated quality check: {path.name}")
                candidate["label_source"] = "train1_filename_annotation"
                selected[identity] = candidate

    args.output.mkdir(parents=True, exist_ok=True)
    for split in split_ids:
        (args.output / "plates" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "rectified" / split).mkdir(parents=True, exist_ok=True)
        for character in CLASS_NAMES:
            (args.output / "chars" / split / character).mkdir(parents=True, exist_ok=True)
    (args.output / "perspective_views" / "train").mkdir(parents=True, exist_ok=True)
    records = []
    class_counts = {split: Counter() for split in split_ids}
    rng_views = random.Random(args.seed + 1)
    preview_triplets = []
    augmentation_stats = Counter()
    split_source_counts = {split: Counter() for split in split_ids}
    for split in ("val", "test", "train"):
        for index, identity in enumerate(split_ids[split], 1):
            item = selected[identity]
            sample_id = f"{split}_{index:04d}"
            plate_rel = Path("plates") / split / f"{sample_id}.png"
            rect_rel = Path("rectified") / split / f"{sample_id}.png"
            save_image(args.output / plate_rel, item["plate"])
            save_image(args.output / rect_rel, item["rectified"])
            crop_files = []
            for char_index, (character, crop) in enumerate(zip(identity, item["crops"])):
                crop_rel = Path("chars") / split / character / f"{sample_id}_{char_index:02d}.png"
                save_image(args.output / crop_rel, crop)
                crop_files.append(crop_rel.as_posix())
                class_counts[split][character] += 1

            view_success = view_failure = 0
            sample_views = [item["plate"].copy()]
            if split == "train":
                for view_index in range(1, args.perspective_views + 1):
                    strength = args.perspective_strength if view_index == 1 else min(0.12, args.perspective_strength * 1.6)
                    warped, transform = perspective_view(item["plate"], rng_views, strength)
                    augmented_rectified, _, _ = perspective_correct(warped)
                    segmented = segment_characters(augmented_rectified, expected_count=len(identity), enhancement=args.enhancement)
                    if segmented is None or segmented[1] != 1 or segmented[2] != len(identity):
                        view_failure += 1
                        augmentation_stats["perspective_view_segmentation_failure"] += 1
                        continue
                    aug_crops, _, _ = segmented
                    view_rel = Path("perspective_views") / "train" / f"{sample_id}_p{view_index:02d}.png"
                    save_image(args.output / view_rel, warped)
                    for char_index, (character, crop) in enumerate(zip(identity, aug_crops)):
                        crop_rel = Path("chars") / "train" / character / f"{sample_id}_p{view_index:02d}_{char_index:02d}.png"
                        save_image(args.output / crop_rel, crop)
                        crop_files.append(crop_rel.as_posix())
                        class_counts["train"][character] += 1
                    view_success += 1
                    augmentation_stats["perspective_views_saved"] += 1
                    sample_views.append(warped.copy())
                    item.setdefault("perspective_transforms", []).append(transform)
            if split == "train" and len(preview_triplets) < 12 and len(sample_views) >= 3:
                preview_triplets.append((sample_views[0], sample_views[1], sample_views[2]))

            split_source_counts[split][item.get("label_source", "unknown")] += 1
            source_path = item["source_path"]
            records.append({
                "sample_id": sample_id,
                "split": split,
                "text": identity,
                "source_filename_text": item.get("source_text"),
                "source": source_relative(source_path),
                "source_kind": item.get("label_source"),
                "label_source": item.get("label_source"),
                "pseudo_min_confidence": item.get("pseudo_min_confidence"),
                "pseudo_mean_confidence": item.get("pseudo_mean_confidence"),
                "plate_crop": plate_rel.as_posix(),
                "rectified_crop": rect_rel.as_posix(),
                "crop_files": crop_files,
                "char_count": len(identity),
                "row_count": 1,
                "ground_truth_segmentation_ok": True,
                "ground_truth_segmentation_status": "ok",
                "perspective_corrected": bool(item["perspective_applied"]),
                "angle_deg": round(item["angle"], 3),
                "blur_score": round(item["blur"], 3),
                "contrast_std": round(item["contrast"], 3),
                "plate_aspect_ratio": round(item["aspect_ratio"], 3),
                "perspective_views_requested": args.perspective_views if split == "train" else 0,
                "perspective_views_saved": view_success,
                "perspective_views_rejected": view_failure,
                "perspective_transforms": item.get("perspective_transforms", []),
                "enhancement": args.enhancement,
            })
        print(f"saved {split}={len(split_ids[split])}", flush=True)

    make_preview(args.output / "review" / "perspective_examples.png", preview_triplets)
    (args.output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": "mixed-source frontal one-row car plates with train-only perspective augmentation",
        "source_description": f"Fresh 1,000-plate LeNet training set: {primary_train_count} source-filename-labelled one-row plates from train(1) plus {len(legacy_training)} confidence-filtered, visually audited CarLongPlate crops from train, with perspective variants only for training",
        "teacher_checkpoint_used_for_legacy_pseudo_labels": args.teacher.as_posix(),
        "primary_source": args.primary_root.as_posix(),
        "legacy_source": args.legacy_root.as_posix(),
        "primary_images_scanned": len(primary_paths),
        "fresh_primary_unique_readable_plate_identities_after_prior_exclusion": len(best_primary),
        "primary_frames_excluded_for_prior_identities": primary_rejected["identity_seen_in_previous_run_excluded"],
        "previous_run_identities_excluded_from_new_train_val_test": len(old_ids),
        "primary_rejected_by_reason": dict(primary_rejected),
        "legacy_train_source_images_total": sum(1 for _ in args.legacy_root.glob("*.jpg")),
        "legacy_car_long_plate_candidates": len(legacy_files),
        "legacy_selected_pseudo_label_plates": len(legacy_training),
        "legacy_pseudo_label_threshold": {
            "minimum_per_character_softmax_confidence_across_two_preprocessings": args.min_pseudo_confidence,
            "preprocessings": ["clahe_sharp", "clahe"],
            "both_views_must_predict_identical_text": True,
            "allowed_rows": 1,
            "allowed_character_counts": [7, 8, 9],
        },
        "legacy_pseudo_label_rejections_and_annotation_audit": dict(legacy_reasons),
        "legacy_manual_visual_corrections_count": legacy_reasons["legacy_visual_label_manually_corrected"],
        "source_images_used_by_split": {split: dict(counts) for split, counts in split_source_counts.items()},
        "selected_plate_counts": {split: len(ids) for split, ids in split_ids.items()},
        "unique_plate_identities_across_splits": len({row["text"] for row in records}),
        "new_test_and_validation_identities_excluded_from_previous_training_run": True,
        "perspective_augmentation": {
            "training_only": True,
            "views_per_unique_training_plate_requested": args.perspective_views,
            "views_saved": augmentation_stats["perspective_views_saved"],
            "views_rejected_for_segmentation": augmentation_stats["perspective_view_segmentation_failure"],
            "strengths": [args.perspective_strength, min(0.12, args.perspective_strength * 1.6)],
            "roll_degrees": [-2.5, 2.5],
            "preview": (args.output / "review" / "perspective_examples.png").as_posix(),
        },
        "selected_characters_per_split_including_saved_perspective_views": {split: dict(counts) for split, counts in class_counts.items()},
        "quality_filters": {
            "plate_aspect_ratio": [args.min_aspect, args.max_aspect],
            "plate_roll_angle_max_degrees": args.max_angle,
            "laplacian_blur_minimum": args.min_blur,
            "grayscale_contrast_std_minimum": args.min_contrast,
            "minimum_plate_crop_size_px": [args.min_width, args.min_height],
            "segmentation_must_match_filename_or_pseudo_label_length": True,
        },
        "label_policy": "train(1) text comes from source filenames with reviewed corrections; train/CarLongPlate text is teacher pseudo-label only after dual-preprocessing agreement, confidence filter, annotation-box-count check, and visual audit; legacy YOLO character classes are not used as ground truth because audited annotations disagree with visible images",
        "source_images_deleted": 0,
        "seed": args.seed,
    }
    (args.output / "preparation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
