from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from train_independent_structured import (
    estimate_blur,
    parse_source_name,
    perspective_correct,
    segment_characters,
)
from train_lenet5 import CLASS_NAMES, LeNet5


DEFAULT_SOURCE = Path("data/OCR/OCR/images/train(1)/detection/one_row")
DEFAULT_OUTPUT = Path("data/one_row_1000_curated_v4")
DEFAULT_AUDIT_MODEL = Path("artifacts/independent_lenet5_dense_enhanced_train1.pt")
VISUALLY_CONFIRMED_LABEL_CORRECTIONS = {
    "B14A14304": "14A14304",
    "B36A38895": "36A38895",
    "B36M2390": "36M2390",
    "BC36C26957": "36C26957",
    "BD36B0734": "36B0734",
    "BD36N1662": "36N1662",
    "6C15747": "36C15747",
    "33H522": "33H5242",
}


def load_audit_model(path: Path, device: torch.device):
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_crops(model, crops: list[np.ndarray], device: torch.device, batch_size: int = 512):
    if model is None or not crops:
        return [None] * len(crops)
    output = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            block = np.stack(crops[start : start + batch_size]).astype(np.float32) / 255.0
            tensor = torch.from_numpy((block[:, None, :, :] - 0.5) / 0.5).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)
            confidence, indices = probabilities.max(1)
            output.extend(
                (CLASS_NAMES[int(index)], float(score))
                for index, score in zip(indices.cpu(), confidence.cpu())
            )
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_preview_sheets(root: Path, records: list[dict], per_sheet: int = 50):
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    items = [record for record in records if record["split"] == "train"]
    for start in range(0, min(len(items), 100), per_sheet):
        batch = items[start : start + per_sheet]
        columns, rows = 5, 10
        tile_w, tile_h = 220, 120
        sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
        for offset, record in enumerate(batch):
            image = cv2.imread(str(root / record["plate_crop"]))
            if image is None:
                continue
            x, y = (offset % columns) * tile_w, (offset // columns) * tile_h
            h, w = image.shape[:2]
            scale = min((tile_w - 12) / max(1, w), 76 / max(1, h))
            resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
            rh, rw = resized.shape[:2]
            sheet[y + 4 : y + 4 + rh, x + (tile_w - rw) // 2 : x + (tile_w + rw) // 2] = resized
            audit = record.get("audit_prediction") or "(OCR unavailable)"
            caption = f"GT {record['text']} | OCR {audit}"
            color = (25, 115, 25) if audit == record["text"] else (20, 20, 210)
            cv2.putText(sheet, caption[:30], (x + 6, y + 101), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(review_dir / f"train_preview_{start // per_sheet + 1:02d}.jpg"), sheet)


def main():
    parser = argparse.ArgumentParser(description="Create a separate, identity-disjoint one-row plate subset and OCR audit.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--enhancement", choices=("none", "clahe", "clahe_sharp"), default="clahe_sharp")
    parser.add_argument("--audit-model", type=Path, default=DEFAULT_AUDIT_MODEL)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a new path or move it yourself: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.output / "plates" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "rectified" / split).mkdir(parents=True, exist_ok=True)
        for char in CLASS_NAMES:
            (args.output / "chars" / split / char).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    by_text: dict[str, list[Path]] = defaultdict(list)
    original_labels: dict[Path, str] = {}
    bad_names = 0
    for path in args.source.glob("*.jpg"):
        parsed = parse_source_name(path)
        if parsed is None or parsed[5] != "one_row":
            bad_names += 1
            continue
        original_labels[path] = parsed[0]
        corrected_text = VISUALLY_CONFIRMED_LABEL_CORRECTIONS.get(parsed[0], parsed[0])
        by_text[corrected_text].append(path)
    identities = list(by_text)
    rng.shuffle(identities)
    corrected_ids = set(VISUALLY_CONFIRMED_LABEL_CORRECTIONS.values()) & set(by_text)
    remaining_identities = [text for text in identities if text not in corrected_ids]
    n_train = max(0, int(len(identities) * 0.70) - len(corrected_ids))
    n_val = int(len(identities) * 0.10)
    identity_split = {
        **{text: "train" for text in corrected_ids},
        **{text: "train" for text in remaining_identities[:n_train]},
        **{text: "val" for text in remaining_identities[n_train : n_train + n_val]},
        **{text: "test" for text in remaining_identities[n_train + n_val :]},
    }
    targets = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    for choices in by_text.values():
        rng.shuffle(choices)
    ordered_ids = {name: [text for text in identities if identity_split[text] == name] for name in targets}
    for ids in ordered_ids.values():
        rng.shuffle(ids)
    ordered_ids["train"] = sorted(corrected_ids) + [text for text in ordered_ids["train"] if text not in corrected_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    audit_model = load_audit_model(args.audit_model, device)
    records: list[dict] = []
    rejected = Counter()
    source_attempts = Counter()
    accepted_by_split = Counter()
    audit_char_crops: list[np.ndarray] = []
    audit_char_ranges: list[tuple[int, int]] = []
    audit_char_targets: list[str] = []
    audit_no_count_crops: list[np.ndarray] = []
    audit_no_count_ranges: list[tuple[int, int]] = []

    next_index = 0
    for split in ("train", "val", "test"):
        for text in ordered_ids[split]:
            if accepted_by_split[split] >= targets[split]:
                break
            accepted_this_identity = False
            for source_path in by_text[text]:
                source_attempts[split] += 1
                parsed = parse_source_name(source_path)
                _, x1, y1, x2, y2, _ = parsed
                image = cv2.imread(str(source_path))
                if image is None:
                    rejected["read_error"] += 1
                    continue
                h, w = image.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                plate = image[y1:y2, x1:x2]
                if plate.size == 0:
                    rejected["empty_box_crop"] += 1
                    continue
                corrected, perspective_applied, angle = perspective_correct(plate)
                segmented = segment_characters(corrected, expected_count=len(text), enhancement=args.enhancement)
                segmentation_ok = segmented is not None
                row_count = None
                char_count = 0
                char_crops = []
                if segmented is not None:
                    char_crops, row_count, char_count = segmented
                    segmentation_ok = row_count == 1 and char_count == len(text)
                if not segmentation_ok:
                    rejected[f"{split}_ground_truth_segmentation_failed"] += 1
                    # Train/validation require usable character crops. Test keeps
                    # the randomly selected ROI so failures count against coverage.
                    if split != "test":
                        continue

                sample_id = f"{split}_{accepted_by_split[split] + 1:04d}"
                plate_rel = Path("plates") / split / f"{sample_id}.png"
                rectified_rel = Path("rectified") / split / f"{sample_id}.png"
                cv2.imwrite(str(args.output / plate_rel), plate)
                cv2.imwrite(str(args.output / rectified_rel), corrected)
                char_relatives = []
                audit_start = len(audit_char_crops)
                for char_index, (target_char, char_crop) in enumerate(zip(text, char_crops)):
                    char_rel = Path("chars") / split / target_char / f"{sample_id}_{char_index:02d}.png"
                    cv2.imwrite(str(args.output / char_rel), char_crop)
                    char_relatives.append(char_rel.as_posix())
                    audit_char_crops.append(char_crop)
                    audit_char_targets.append(target_char)
                audit_char_ranges.append((audit_start, len(audit_char_targets)))

                runtime_seg = segment_characters(corrected, expected_count=None, enhancement=args.enhancement)
                if runtime_seg is None:
                    runtime_crops = []
                else:
                    runtime_crops, runtime_rows, runtime_count = runtime_seg
                    if runtime_rows != 1:
                        runtime_crops = []
                audit_no_count_ranges.append((len(audit_no_count_crops), len(audit_no_count_crops) + len(runtime_crops)))
                audit_no_count_crops.extend(runtime_crops)

                records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "text": text,
                    "source_filename_text": original_labels[source_path],
                    "label_corrected_after_visual_review": original_labels[source_path] != text,
                    "source": source_path.name,
                    "plate_crop": plate_rel.as_posix(),
                    "rectified_crop": rectified_rel.as_posix(),
                    "crop_files": char_relatives,
                    "char_count": char_count,
                    "row_count": row_count,
                    "ground_truth_segmentation_ok": bool(segmentation_ok),
                    "ground_truth_segmentation_status": "ok" if segmentation_ok else "not_segmentable_with_current_rules",
                    "perspective_corrected": bool(perspective_applied),
                    "angle_deg": round(float(angle), 3),
                    "blur_score": round(estimate_blur(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)), 3),
                    "enhancement": args.enhancement,
                    "audit_prediction": None,
                    "audit_char_confidence": None,
                    "runtime_audit_prediction": None,
                })
                accepted_by_split[split] += 1
                next_index += 1
                accepted_this_identity = True
                break
            if accepted_by_split[split] >= targets[split]:
                break
            if not accepted_this_identity:
                rejected[f"identity_unusable_{split}"] += 1
        if accepted_by_split[split] < targets[split]:
            raise RuntimeError(
                f"Could not produce requested {targets[split]} {split} samples; made {accepted_by_split[split]}. "
                "Inspect rejection counts and segmentation before lowering quality checks."
            )
        print(
            f"split={split} accepted={accepted_by_split[split]}/{targets[split]} "
            f"unique_plate_ids={accepted_by_split[split]} source_frames_tried={source_attempts[split]}",
            flush=True,
        )

    char_predictions = predict_crops(audit_model, audit_char_crops, device)
    no_count_predictions = predict_crops(audit_model, audit_no_count_crops, device)
    cursor = 0
    runtime_cursor = 0
    audit_mismatches = []
    for record, (start, end), (runtime_start, runtime_end) in zip(records, audit_char_ranges, audit_no_count_ranges):
        local_predictions = char_predictions[start:end]
        if local_predictions and all(item is not None for item in local_predictions):
            predicted_text = "".join(item[0] for item in local_predictions)
            record["audit_prediction"] = predicted_text
            record["audit_char_confidence"] = [round(item[1], 5) for item in local_predictions]
            record["audit_matches_annotation"] = predicted_text == record["text"]
            if predicted_text != record["text"]:
                audit_mismatches.append({
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "source": record["source"],
                    "plate_crop": record["plate_crop"],
                    "annotation_text": record["text"],
                    "audit_ocr_text": predicted_text,
                    "audit_status": "review_image; annotation retained as label",
                })
        else:
            record["audit_matches_annotation"] = None
            record["audit_status"] = "no_independent_audit_model"
        runtime_predictions = no_count_predictions[runtime_start:runtime_end]
        if runtime_predictions and all(item is not None for item in runtime_predictions):
            record["runtime_audit_prediction"] = "".join(item[0] for item in runtime_predictions)
        record["annotation_source"] = "ground_truth text encoded in source filename"
        record["annotation_is_overwritten_by_ocr"] = False
        cursor = end
        runtime_cursor = runtime_end

    split_labels = defaultdict(set)
    for record in records:
        split_labels[record["split"]].add(record["text"])
    assert not (split_labels["train"] & split_labels["val"])
    assert not (split_labels["train"] & split_labels["test"])
    assert not (split_labels["val"] & split_labels["test"])

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output / "labels_and_ocr_audit.csv", [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "source": row["source"],
            "crop": row["plate_crop"],
            "original_filename_text": row["source_filename_text"],
            "ground_truth_from_filename": row["text"],
            "manual_visual_correction_applied": row["label_corrected_after_visual_review"],
            "previous_model_ocr_audit": row["audit_prediction"] or "",
            "audit_matches_ground_truth": row.get("audit_matches_annotation"),
            "runtime_no_known_count_audit": row["runtime_audit_prediction"] or "",
            "human_review_recommended": row.get("audit_matches_annotation") is not True,
        }
        for row in records
    ])
    write_csv(args.output / "review" / "ocr_mismatches.csv", audit_mismatches)
    corrections = [
        {"source_filename_label": source_text, "curated_label": corrected_text,
         "source_filenames": ", ".join(sorted(path.name for path, original in original_labels.items() if original == source_text)),
         "verification": "visually checked against the original full-frame plate crop"}
        for source_text, corrected_text in sorted(VISUALLY_CONFIRMED_LABEL_CORRECTIONS.items())
    ]
    write_csv(args.output / "review" / "label_corrections.csv", corrections)
    save_preview_sheets(args.output, records)

    summary = {
        "source_directory": str(args.source),
        "output_directory": str(args.output),
        "source_one_row_images": sum(1 for _ in args.source.glob("*.jpg")),
        "parsed_unique_plate_labels": len(identities),
        "bad_filename_count": bad_names,
        "seed": args.seed,
        "split_is_grouped_by_exact_plate_label": True,
        "visually_corrected_plate_ids_reserved_for_training": sorted(corrected_ids),
        "visually_confirmed_source_label_corrections": corrections,
        "visually_corrected_samples_by_split": dict(Counter(
            row["split"] for row in records if row["label_corrected_after_visual_review"]
        )),
        "requested_and_collected_plate_images": dict(accepted_by_split),
        "segmentable_plate_images": {
            split: sum(row["split"] == split and row["ground_truth_segmentation_ok"] for row in records)
            for split in ("train", "val", "test")
        },
        "test_sampling_protocol": "500 identities randomly assigned to test and one frame randomly selected before segmentation; every usable annotated ROI is retained, including segmentation failures.",
        "source_frames_tried_by_split": dict(source_attempts),
        "rejected_candidates": dict(rejected),
        "character_crops": len(audit_char_targets),
        "character_count_distribution_by_split": {
            split: dict(Counter(len(row["text"]) for row in records if row["split"] == split))
            for split in ("train", "val", "test")
        },
        "previous_model_audit": {
            "checkpoint": str(args.audit_model) if audit_model is not None else None,
            "is_independent_evaluation": False,
            "note": "Used only to flag label/image disagreements; this checkpoint has seen the broader source dataset, so its predictions are not reported as an unbiased score and never overwrite filename labels.",
            "audited_plates": sum(row.get("audit_matches_annotation") is not None for row in records) if audit_model is not None else 0,
            "plate_text_agreements": sum(row.get("audit_matches_annotation") is True for row in records),
            "plate_text_disagreements": len(audit_mismatches),
            "mismatches_csv": "review/ocr_mismatches.csv",
        },
        "labels_are_taken_from": "source filename annotation except visually confirmed corrections listed in review/label_corrections.csv; original files are unchanged",
        "preprocessing": ["annotated plate-box crop", "perspective correction", args.enhancement, "connected-component character segmentation"],
    }
    (args.output / "preparation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
