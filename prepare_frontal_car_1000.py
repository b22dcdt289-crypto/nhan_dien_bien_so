from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from train_independent_structured import estimate_blur, parse_source_name, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES


DEFAULT_ROOTS = [
    Path("data/OCR/OCR/images/train(1)/detection/one_row"),
    Path("data/VNLP/detection/one_row"),
]
DEFAULT_OUTPUT = Path("data/one_row_frontal_1000_v1")
LABEL_CORRECTIONS = {
    "B14A14304": "14A14304",
    "B36A38895": "36A38895",
    "B36M2390": "36M2390",
    "BC36C26957": "36C26957",
    "BD36B0734": "36B0734",
    "BD36N1662": "36N1662",
    "6C15747": "36C15747",
    "33H522": "33H5242",
}


def source_images(roots: list[Path]) -> tuple[list[Path], Counter, list[str]]:
    """Combine roots and quickly recognize byte-identical mirrored datasets."""
    unique: list[Path] = []
    counts: Counter = Counter()
    mirror_notes: list[str] = []
    seen_by_name: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            counts[f"missing_root:{root.as_posix()}"] += 1
            continue
        files = sorted(path for path in root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        names = {path.name for path in files}
        same_inventory = bool(seen_by_name) and names == set(seen_by_name)
        same_sizes = same_inventory and all(path.stat().st_size == seen_by_name[path.name].stat().st_size for path in files)
        mirror_sample_count = 0
        if same_sizes:
            stride = max(1, len(files) // 24)
            sample_files = files[::stride][:24]
            mirror_sample_count = len(sample_files)
            same_sizes = all(path.read_bytes() == seen_by_name[path.name].read_bytes() for path in sample_files)
        if same_sizes:
            counts[f"files:{root.as_posix()}"] = len(files)
            counts[f"unique:{root.as_posix()}"] = 0
            counts[f"duplicate_copies:{root.as_posix()}"] = len(files)
            mirror_notes.append(
                f"{root.as_posix()}: same filename/size inventory as prior root and {mirror_sample_count} byte-identical samples; treated as a mirrored copy"
            )
            continue
        root_unique = 0
        root_duplicate = 0
        for path in files:
            previous = seen_by_name.get(path.name)
            if previous is not None and previous.stat().st_size == path.stat().st_size and previous.read_bytes() == path.read_bytes():
                root_duplicate += 1
                continue
            unique.append(path)
            root_unique += 1
            # Preserve same-named, non-identical files as independent candidates.
            seen_by_name.setdefault(path.name, path)
        counts[f"files:{root.as_posix()}"] = len(files)
        counts[f"unique:{root.as_posix()}"] = root_unique
        counts[f"duplicate_copies:{root.as_posix()}"] = root_duplicate
        if root_duplicate:
            mirror_notes.append(
                f"{root.as_posix()}: skipped {root_duplicate} byte-identical copies already present in an earlier source root"
            )
    return unique, counts, mirror_notes


def candidate_from_image(path: Path, args, rejected: Counter):
    parsed = parse_source_name(path)
    if parsed is None:
        rejected["bad_or_out_of_vocabulary_filename"] += 1
        return None
    source_text, x1, y1, x2, y2, plate_type = parsed
    if plate_type != "one_row":
        rejected["not_one_row_car"] += 1
        return None
    text = LABEL_CORRECTIONS.get(source_text, source_text)
    if len(text) not in (7, 8, 9):
        rejected["unsupported_label_length"] += 1
        return None
    if any(character not in CLASS_NAMES for character in text):
        rejected["out_of_vocabulary_label"] += 1
        return None

    image = cv2.imread(str(path))
    if image is None:
        rejected["image_read_error"] += 1
        return None
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    plate = image[y1:y2, x1:x2]
    if plate.size == 0:
        rejected["empty_plate_crop"] += 1
        return None

    plate_height, plate_width = plate.shape[:2]
    if plate_width < args.min_width or plate_height < args.min_height:
        rejected["low_resolution"] += 1
        return None
    aspect_ratio = plate_width / max(1, plate_height)
    if not args.min_aspect <= aspect_ratio <= args.max_aspect:
        rejected["not_frontal_aspect_ratio"] += 1
        return None

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur_score = estimate_blur(gray)
    if blur_score < args.min_blur:
        rejected["too_blurred"] += 1
        return None
    contrast = float(gray.std())
    if contrast < args.min_contrast:
        rejected["low_contrast"] += 1
        return None

    rectified, perspective_applied, angle = perspective_correct(plate)
    if angle > args.max_angle:
        rejected["excessive_plate_rotation"] += 1
        return None
    segmented = segment_characters(rectified, expected_count=len(text), enhancement=args.enhancement)
    if segmented is None:
        rejected["character_segmentation_failed_or_wrong_count"] += 1
        crops, segmentation_ok = [], False
    else:
        crops, row_count, character_count = segmented
        segmentation_ok = row_count == 1 and character_count == len(text)
        if not segmentation_ok:
            rejected["not_one_row_or_wrong_character_count"] += 1
            crops = []

    # Selection score only ranks repeated images of the same plate identity.
    # Every image must still pass the same front-view/readability gates above.
    score = (
        np.log1p(min(blur_score, 10000.0))
        + 0.35 * np.log1p(plate_width * plate_height)
        - 1.5 * abs(np.log(aspect_ratio / 4.7))
        - 0.08 * angle
    )
    return {
        "text": text,
        "source_text": source_text,
        "source_path": path,
        "plate": plate,
        "rectified": rectified,
        "crops": crops,
        "segmentation_ok": segmentation_ok,
        "angle": float(angle),
        "blur": float(blur_score),
        "contrast": contrast,
        "aspect_ratio": float(aspect_ratio),
        "perspective_applied": bool(perspective_applied),
        "score": float(score),
    }


def save_frontal_preview(output: Path, records: list[dict], limit: int = 50):
    review_dir = output / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    samples = [record for record in records if record["split"] == "train"][:limit]
    columns, tile_width, tile_height = 5, 250, 94
    rows = (len(samples) + columns - 1) // columns
    sheet = np.full((max(1, rows) * tile_height, columns * tile_width, 3), 242, dtype=np.uint8)
    for index, record in enumerate(samples):
        image = cv2.imread(str(output / record["rectified_crop"]))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min((tile_width - 12) / max(1, w), 66 / max(1, h))
        resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
        x = (index % columns) * tile_width + (tile_width - resized.shape[1]) // 2
        y = (index // columns) * tile_height + 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        label_y = (index // columns) * tile_height + tile_height - 13
        cv2.putText(
            sheet,
            f"{record['sample_id']}  {record['text']}",
            ((index % columns) * tile_width + 6, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 35, 25),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(review_dir / "frontal_train_examples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main():
    parser = argparse.ArgumentParser(
        description="Select exactly 1,000 readable, near-frontal, one-row car plates from all supplied source roots."
    )
    parser.add_argument("--sources", nargs="+", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--label-length", type=int, default=8, help="Require this many visible alphanumeric characters in train/validation/test.")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--test-exclude-manifest",
        type=Path,
        default=None,
        help="Choose test identities only from plates absent from this earlier manifest (prevents reusing prior-model identities).",
    )
    parser.add_argument("--min-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=28)
    parser.add_argument("--min-aspect", type=float, default=3.0)
    parser.add_argument("--max-aspect", type=float, default=7.0)
    parser.add_argument("--max-angle", type=float, default=5.0)
    parser.add_argument("--min-blur", type=float, default=80.0)
    parser.add_argument("--min-contrast", type=float, default=25.0)
    parser.add_argument("--enhancement", choices=("clahe", "clahe_sharp"), default="clahe_sharp")
    args = parser.parse_args()

    prior_identities = set()
    if args.test_exclude_manifest is not None:
        if not args.test_exclude_manifest.is_file():
            raise FileNotFoundError(args.test_exclude_manifest)
        prior_identities = {
            str(record["text"]).upper()
            for record in json.loads(args.test_exclude_manifest.read_text(encoding="utf-8"))
        }

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; use another path: {args.output}")
    paths, source_counts, mirror_notes = source_images(args.sources)
    print(f"unique_source_images_to_scan={len(paths)}; beginning image-quality scan", flush=True)
    rejected: Counter = Counter()
    by_identity: dict[str, list[dict]] = defaultdict(list)
    for index, path in enumerate(paths, 1):
        candidate = candidate_from_image(path, args, rejected)
        if candidate is not None:
            by_identity[candidate["text"]].append(candidate)
        if index % 1000 == 0:
            print(
                f"scanned={index}/{len(paths)} eligible_frames={sum(map(len, by_identity.values()))} "
                f"unique_plate_ids={len(by_identity)}",
                flush=True,
            )

    best_by_identity = {
        identity: max(candidates, key=lambda item: item["score"])
        for identity, candidates in by_identity.items()
    }
    trainable_by_identity = {
        identity: max((item for item in candidates if item["segmentation_ok"]), key=lambda item: item["score"])
        for identity, candidates in by_identity.items()
        if len(identity) == args.label_length and any(item["segmentation_ok"] for item in candidates)
    }
    required = args.train_count + args.val_count + args.test_count
    if len(trainable_by_identity) < args.train_count + args.val_count:
        raise RuntimeError(
            f"Only {len(trainable_by_identity)} unique segmentable plates passed the filters; "
            f"{args.train_count + args.val_count} are required for train/validation. Rejected counts: {dict(rejected)}"
        )

    rng = random.Random(args.seed)
    identities = sorted(best_by_identity)
    rng.shuffle(identities)
    unseen_test_candidates = [
        identity for identity in identities
        if identity not in prior_identities and len(identity) == args.label_length
    ]
    if len(unseen_test_candidates) < args.test_count:
        raise RuntimeError(
            f"Only {len(unseen_test_candidates)} eligible unseen 8-character identities passed image-quality gates; "
            f"{args.test_count} are required. Relax quality filters or reduce test-count."
        )
    test_ids = unseen_test_candidates[: args.test_count]
    test_id_set = set(test_ids)
    remaining_ids = [identity for identity in trainable_by_identity if identity not in test_id_set]
    rng.shuffle(remaining_ids)
    val_ids = remaining_ids[: args.val_count]
    train_candidates = remaining_ids[args.val_count :]
    split_ids = {"val": val_ids, "test": test_ids}

    # Keep the held-out pool random and natural. Balance only training plates
    # by the series letter at position 3 so rare series are not accidentally
    # absent from the 1,000 examples used to fit the classifier.
    train_buckets: dict[str, list[str]] = defaultdict(list)
    for identity in train_candidates:
        series = identity[2] if len(identity) > 2 and identity[2].isalpha() else "_"
        train_buckets[series].append(identity)
    for bucket in train_buckets.values():
        rng.shuffle(bucket)
    bucket_names = sorted(train_buckets)
    rng.shuffle(bucket_names)
    train_ids = []
    while len(train_ids) < args.train_count:
        made_progress = False
        for series in bucket_names:
            if train_buckets[series]:
                train_ids.append(train_buckets[series].pop())
                made_progress = True
                if len(train_ids) == args.train_count:
                    break
        if not made_progress:
            raise RuntimeError("Could not fill the requested training count from the remaining identities.")
    split_ids["train"] = train_ids
    if (
        set(split_ids["train"]) & set(split_ids["val"])
        or set(split_ids["train"]) & set(split_ids["test"])
        or set(split_ids["val"]) & set(split_ids["test"])
    ):
        raise RuntimeError("Plate identities leaked across splits.")

    for split in split_ids:
        (args.output / "plates" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "rectified" / split).mkdir(parents=True, exist_ok=True)
        for character in CLASS_NAMES:
            (args.output / "chars" / split / character).mkdir(parents=True, exist_ok=True)

    records = []
    class_counts = {split: Counter() for split in split_ids}
    for split, split_identities in split_ids.items():
        for index, identity in enumerate(split_identities, 1):
            item = best_by_identity[identity] if split == "test" else trainable_by_identity[identity]
            sample_id = f"{split}_{index:04d}"
            plate_rel = Path("plates") / split / f"{sample_id}.png"
            rectified_rel = Path("rectified") / split / f"{sample_id}.png"
            cv2.imwrite(str(args.output / plate_rel), item["plate"])
            cv2.imwrite(str(args.output / rectified_rel), item["rectified"])
            crop_files = []
            for char_index, (character, crop) in enumerate(zip(item["text"], item["crops"])):
                crop_rel = Path("chars") / split / character / f"{sample_id}_{char_index:02d}.png"
                cv2.imwrite(str(args.output / crop_rel), crop)
                crop_files.append(crop_rel.as_posix())
                class_counts[split][character] += 1
            source_path = item["source_path"].resolve()
            try:
                source_name = source_path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                source_name = source_path.as_posix()
            records.append({
                "sample_id": sample_id,
                "split": split,
                "text": identity,
                "source_filename_text": item["source_text"],
                "label_corrected_by_known_mapping": item["source_text"] != identity,
                "label_visually_reviewed": False,
                "source": source_name,
                "plate_crop": plate_rel.as_posix(),
                "rectified_crop": rectified_rel.as_posix(),
                "crop_files": crop_files,
                "char_count": len(identity),
                "row_count": 1,
                "ground_truth_segmentation_ok": item["segmentation_ok"],
                "ground_truth_segmentation_status": "ok" if item["segmentation_ok"] else "unsegmentable_by_current_pipeline",
                "perspective_corrected": item["perspective_applied"],
                "angle_deg": round(item["angle"], 3),
                "blur_score": round(item["blur"], 3),
                "contrast_std": round(item["contrast"], 3),
                "plate_aspect_ratio": round(item["aspect_ratio"], 3),
                "enhancement": args.enhancement,
            })
        print(f"saved {split}={len(split_identities)}", flush=True)

    (args.output / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "dataset": "front-facing, one-row Vietnamese car plates",
        "source_roots": [path.as_posix() for path in args.sources],
        "source_file_counts_and_deduplication": dict(source_counts),
        "mirror_notes": mirror_notes,
        "scanned_unique_images": len(paths),
        "eligible_frames_after_quality_filters": sum(map(len, by_identity.values())),
        "unique_eligible_plate_identities": len(best_by_identity),
        "unique_trainable_plate_identities": len(trainable_by_identity),
        "eligible_identities_absent_from_prior_manifest": len(unseen_test_candidates),
        "prior_identity_exclusion_manifest": args.test_exclude_manifest.as_posix() if args.test_exclude_manifest else None,
        "test_identities_absent_from_prior_model_dataset": not bool(set(split_ids["test"]) & prior_identities),
        "selected_plate_counts": {split: len(ids) for split, ids in split_ids.items()},
        "selected_segmentable_plate_counts": {
            split: sum(best_by_identity[identity]["segmentation_ok"] if split == "test" else trainable_by_identity[identity]["segmentation_ok"] for identity in ids)
            for split, ids in split_ids.items()
        },
        "selected_characters_per_split": {split: dict(counts) for split, counts in class_counts.items()},
        "rejected_frames_by_reason": dict(rejected),
        "filters": {
            "plate_aspect_ratio": [args.min_aspect, args.max_aspect],
            "plate_roll_angle_max_degrees": args.max_angle,
            "laplacian_blur_minimum": args.min_blur,
            "grayscale_contrast_std_minimum": args.min_contrast,
            "minimum_plate_crop_size_px": [args.min_width, args.min_height],
            "train_and_validation_segmentation_must_match_filename_length": True,
            "test_selection_requires_image_quality_and_8_char_label_but_not_character_segmentation": True,
            "row_count_must_equal": 1,
            "allowed_character_count_in_selected_splits": [args.label_length],
        },
        "selection_policy": "one best-quality frame per plate identity; natural random validation/test; training subset is round-robin balanced by the series letter at position 3",
        "unreviewed_legacy_crop_source_excluded": "data/OCR/OCR/images/{train,val} CarLongPlate images were not mixed because inspected character-box labels do not match visible plate text; use only after a full label audit",
        "source_images_deleted": 0,
        "enhancement": args.enhancement,
        "seed": args.seed,
    }
    (args.output / "preparation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_frontal_preview(args.output, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
