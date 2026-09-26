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
from torch import nn
from torch.utils.data import DataLoader

from train_independent_structured import FolderChars, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes, run_epoch


MACS_PER_CHARACTER = 418200


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def read_char_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0)


def predict_crop_files(model, device, root: Path, crop_files: list[str]):
    batch = torch.stack([read_char_tensor(root / rel) for rel in crop_files]).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidence, indices = probs.max(1)
    text = "".join(CLASS_NAMES[index] for index in indices.cpu().tolist())
    return text, confidence.cpu().tolist(), indices.cpu().tolist()


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, path)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def class_report(confusion: np.ndarray):
    rows = []
    supported = []
    for index, character in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "character": character,
            "support": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        if support:
            supported.append(rows[-1])
    macro = {
        key: float(np.mean([row[key] for row in supported])) if supported else 0.0
        for key in ("precision", "recall", "f1")
    }
    return rows, macro


def evaluate_split(model, device, root: Path, records: list[dict], split: str):
    records = [record for record in records if record["split"] == split]
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    by_length = defaultdict(lambda: {"plates": 0, "segmentable_plates": 0, "exact": 0, "correct_chars": 0, "characters": 0})
    plate_rows = []
    exact = correct_chars = total_chars = 0
    segmentable_plates = 0
    char_confidence = []
    runtime_exact = runtime_attempted_exact = runtime_attempts = 0
    runtime_length_mismatch = runtime_edit_distance = 0
    runtime_skipped = Counter()
    runtime_detected_counts = Counter()

    model.eval()
    for record in records:
        target = record["text"]
        segmentable = bool(record.get("ground_truth_segmentation_ok", True) and record["crop_files"])
        if segmentable:
            predicted, confidences, indices = predict_crop_files(model, device, root, record["crop_files"])
            segmentable_plates += 1
        else:
            predicted, confidences, indices = "", [], []
        n_correct = sum(expected == actual for expected, actual in zip(target, predicted))
        is_exact = target == predicted
        exact += int(is_exact)
        if segmentable:
            correct_chars += n_correct
            total_chars += len(target)
        char_confidence.extend(confidences)
        for position, (expected, actual, pred_index) in enumerate(zip(target, predicted, indices), 1):
            expected_index = CLASS_NAMES.index(expected)
            confusion[expected_index, pred_index] += 1
            position_stats[str(position)]["total"] += 1
            position_stats[str(position)]["correct"] += int(expected == actual)
        length_key = str(len(target))
        length = by_length[length_key]
        length["plates"] += 1
        length["exact"] += int(is_exact)
        length["segmentable_plates"] += int(segmentable)
        if segmentable:
            length["correct_chars"] += n_correct
            length["characters"] += len(target)

        corrected_roi = record.get("plate_crop")
        runtime_prediction = ""
        runtime_status = "segmentation_failed"
        runtime_count = None
        if corrected_roi:
            image = cv2.imread(str(root / corrected_roi))
            if image is not None:
                corrected, _, _ = perspective_correct(image)
                segmented = segment_characters(corrected, expected_count=None, enhancement=record["enhancement"])
                if segmented is not None:
                    runtime_crops, runtime_rows, runtime_count = segmented
                    if runtime_rows != 1:
                        runtime_status = "wrong_row_count"
                    elif runtime_count < 1:
                        runtime_status = "no_characters"
                    elif runtime_count not in (7, 8, 9, 10):
                        runtime_detected_counts[str(runtime_count)] += 1
                        runtime_length_mismatch += int(runtime_count != len(target))
                        runtime_status = "unsupported_character_count"
                    else:
                        runtime_detected_counts[str(runtime_count)] += 1
                        runtime_length_mismatch += int(runtime_count != len(target))
                        tensors = []
                        for crop in runtime_crops:
                            pixels = crop.astype(np.float32) / 255.0
                            tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
                        with torch.no_grad():
                            logits = model(torch.stack(tensors).to(device))
                        runtime_prediction = "".join(CLASS_NAMES[index] for index in logits.argmax(1).cpu().tolist())
                        runtime_attempts += 1
                        runtime_edit_distance += edit_distance(target, runtime_prediction)
                        runtime_exact += int(runtime_prediction == target)
                        runtime_attempted_exact += int(runtime_prediction == target)
                        runtime_status = "attempted"
                else:
                    runtime_status = "segmentation_failed"
        if runtime_status != "attempted":
            runtime_skipped[runtime_status] += 1
        plate_rows.append({
            "sample_id": record["sample_id"],
            "source": record["source"],
            "expected": target,
            "ocr_with_ground_truth_segmentation": predicted,
            "correct_characters": n_correct,
            "total_characters": len(target),
            "character_accuracy": n_correct / max(1, len(target)),
            "exact_match_with_ground_truth_segmentation": is_exact,
            "runtime_without_known_count": runtime_prediction,
            "runtime_status": runtime_status,
            "runtime_detected_character_count": runtime_count,
            "runtime_exact_match": runtime_prediction == target,
            "macs_for_ground_truth_character_count": MACS_PER_CHARACTER * len(target),
            "ground_truth_segmentation_ok": segmentable,
        })

    class_rows, macro = class_report(confusion)
    n_plates = len(records)
    position_report = {
        position: {
            **stats,
            "accuracy": stats["correct"] / max(1, stats["total"]),
        }
        for position, stats in sorted(position_stats.items(), key=lambda item: int(item[0]))
    }
    length_report = {
        length: {
            **stats,
            "exact_accuracy": stats["exact"] / max(1, stats["plates"]),
            "exact_accuracy_among_segmentable": stats["exact"] / max(1, stats["segmentable_plates"]),
            "character_accuracy": stats["correct_chars"] / max(1, stats["characters"]),
        }
        for length, stats in sorted(by_length.items(), key=lambda item: int(item[0]))
    }
    summary = {
        "split": split,
        "plates": n_plates,
        "ground_truth_segmentable_plates": segmentable_plates,
        "ground_truth_segmentation_coverage": segmentable_plates / max(1, n_plates),
        "character_count": total_chars,
        "character_count_all_plate_labels": sum(len(record["text"]) for record in records),
        "correct_characters": correct_chars,
        "character_accuracy_conditional_on_ground_truth_segmentation": correct_chars / max(1, total_chars),
        "character_accuracy_conditional_wilson_95_ci": wilson_interval(correct_chars, total_chars),
        "exact_plates": exact,
        "plate_exact_accuracy_all_test_plates": exact / max(1, n_plates),
        "plate_exact_accuracy_conditional_on_ground_truth_segmentation": exact / max(1, segmentable_plates),
        "plate_exact_wilson_95_ci_all_test_plates": wilson_interval(exact, n_plates),
        "mean_max_softmax_confidence_per_character": float(np.mean(char_confidence)) if char_confidence else None,
        "macro_precision_recall_f1_over_characters_with_support": macro,
        "per_character_position": position_report,
        "by_label_length": length_report,
        "per_character_class": class_rows,
        "runtime_no_known_count": {
            "ground_truth_roi_not_full_frame": True,
            "attempted": runtime_attempts,
            "skipped": n_plates - runtime_attempts,
            "coverage": runtime_attempts / max(1, n_plates),
            "skip_reasons": dict(runtime_skipped),
            "detected_character_count_histogram": dict(runtime_detected_counts),
            "exact_plates_all_including_skips": runtime_exact,
            "exact_accuracy_all_including_skips": runtime_exact / max(1, n_plates),
            "exact_accuracy_among_attempted": runtime_attempted_exact / max(1, runtime_attempts),
            "character_count_mismatch": runtime_length_mismatch,
            "sum_levenshtein_distance": runtime_edit_distance,
            "character_error_rate_by_levenshtein": runtime_edit_distance / max(1, sum(len(record["text"]) for record in records)),
        },
        "macs_per_character": MACS_PER_CHARACTER,
        "total_character_macs": MACS_PER_CHARACTER * total_chars,
        "average_macs_per_plate": MACS_PER_CHARACTER * total_chars / max(1, n_plates),
    }
    return summary, plate_rows, confusion


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train dense LeNet-5 from scratch on exactly 1,000 one-row plates.")
    parser.add_argument("--data", type=Path, default=Path("data/one_row_1000_curated_v4"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_dense_one_row_1000.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/one_row_1000_metrics.json"))
    parser.add_argument("--test-csv", type=Path, default=Path("artifacts/one_row_1000_test_predictions.csv"))
    parser.add_argument("--history-csv", type=Path, default=Path("artifacts/one_row_1000_training_history.csv"))
    parser.add_argument("--evaluate-only", action="store_true", help="reuse the trained checkpoint and recompute metrics")
    args = parser.parse_args()

    manifest_path = args.data / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run prepare_one_row_1000.py first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation_metrics_path = args.data / "preparation_metrics.json"
    preparation_metrics = (
        json.loads(preparation_metrics_path.read_text(encoding="utf-8"))
        if preparation_metrics_path.is_file()
        else None
    )
    train_records = [row for row in manifest if row["split"] == "train"]
    val_records = [row for row in manifest if row["split"] == "val"]
    test_records = [row for row in manifest if row["split"] == "test"]
    if len(train_records) != 1000:
        raise RuntimeError(f"Expected exactly 1,000 train plates; found {len(train_records)}")
    if not val_records or not test_records:
        raise RuntimeError("Separate validation and test splits are required.")
    split_labels = {
        split: {row["text"] for row in manifest if row["split"] == split}
        for split in ("train", "val", "test")
    }
    if any(split_labels[a] & split_labels[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("A plate label occurs in multiple splits; refusing a leaked evaluation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_set = FolderChars(args.data / "chars", "train", augment=True)
    val_set = FolderChars(args.data / "chars", "val", augment=False)
    history = []
    if args.evaluate_only and args.history_csv.is_file():
        with args.history_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                history.append({
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_character_accuracy": float(row["train_character_accuracy"]),
                    "validation_loss": float(row["validation_loss"]),
                    "validation_character_accuracy": float(row["validation_character_accuracy"]),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        if not args.output.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.output}")
        best_model, checkpoint = load_checkpoint(args.output, device)
        best_val = checkpoint.get("validation_character_accuracy", 0.0)
        print(f"evaluation_only=1 checkpoint={args.output}", flush=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = LeNet5(len(CLASS_NAMES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        best_val = -1.0
        print(
            f"device={device} training_plates={len(train_records)} train_characters={len(train_set)} "
            f"validation_plates={len(val_records)} validation_characters={len(val_set)} test_plates={len(test_records)} "
            f"architecture=LeNet5_dense pruning=none",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
            val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_character_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_character_accuracy": val_accuracy,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_accuracy > best_val:
                best_val = val_accuracy
                torch.save({
                    "model": model.state_dict(),
                    "classes": CLASS_NAMES,
                    "arch": "LeNet5_dense_no_pruning",
                    "training_plates": len(train_records),
                    "training_character_crops": len(train_set),
                    "validation_character_accuracy": best_val,
                    "enhancement": manifest[0]["enhancement"],
                    "seed": args.seed,
                }, args.output)

    if not args.evaluate_only:
        best_model, checkpoint = load_checkpoint(args.output, device)
    val_summary, _, _ = evaluate_split(best_model, device, args.data, manifest, "val")
    test_summary, test_rows, confusion = evaluate_split(best_model, device, args.data, manifest, "test")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "class_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "dataset": str(args.data),
        "source": (
            "train(1) one-row car plates plus deduplicated VNLP mirror"
            if preparation_metrics and preparation_metrics.get("dataset") == "front-facing, one-row Vietnamese car plates"
            else "train(1)/detection/one_row"
        ),
        "label_source": "source filename annotation except eight manually visually confirmed corrections; OCR audit never overwrote labels",
        "split_grouping": "unique exact plate label assigned to exactly one split",
        "architecture": "LeNet-5 dense, trained from random initialization; no pruning",
        "preprocessing": manifest[0]["enhancement"],
        "preparation": preparation_metrics,
        "training": {
            "epochs_requested": args.epochs,
            "evaluation_only": args.evaluate_only,
            "best_validation_character_accuracy": best_val,
            "train_plates": len(train_records),
            "train_characters": len(train_set),
            "validation_plates": len(val_records),
            "validation_characters": len(val_set),
            "test_plates": len(test_records),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint": str(args.output),
        },
        "validation": val_summary,
        "test": test_summary,
        "interpretation": {
            "ground_truth_segmentation_scores": "Character crops are generated with annotated label length; character accuracy is conditional on segmentable test ROIs, while exact plate accuracy uses all test ROIs and counts segmentation failures as incorrect.",
            "runtime_no_known_count_scores": "Test plate identities and frames are sampled before checking segmentation; no expected count is passed, and skips count against accuracy. The input is still a ground-truth plate ROI, not a full frame.",
            "population_scope": "One-row car plate ROIs passing the frontalness and legibility filters in the preparation metrics; results do not estimate performance on oblique/blurred rejected plates or full-frame detection.",
            "test_is_identity_disjoint": True,
            "uncertainty_interval": "Wilson 95% interval for binary character and exact-plate rates.",
        },
        "epoch_history": history,
        "checkpoint_validation_character_accuracy": checkpoint.get("validation_character_accuracy"),
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.test_csv, test_rows)
    confusion_path = args.metrics.with_name(args.metrics.stem + "_confusion_matrix.csv")
    labels = ["true/pred", *CLASS_NAMES]
    with confusion_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        for character, row in zip(CLASS_NAMES, confusion):
            writer.writerow([character, *row.tolist()])
    history_path = args.history_csv
    if history or not args.evaluate_only:
        write_csv(history_path, history)

    summary = {
        "model": metrics["architecture"],
        "class_count": len(CLASS_NAMES),
        "classes": "".join(CLASS_NAMES),
        "training_plates": len(train_records),
        "training_character_crops": len(train_set),
        "validation_plates": len(val_records),
        "test_plates": len(test_records),
        "validation_char_accuracy": val_summary["character_accuracy_conditional_on_ground_truth_segmentation"],
        "test_char_accuracy_conditional_on_segmentable_rois": test_summary["character_accuracy_conditional_on_ground_truth_segmentation"],
        "test_char_accuracy_conditional_95_ci": test_summary["character_accuracy_conditional_wilson_95_ci"],
        "test_ground_truth_segmentation_coverage": test_summary["ground_truth_segmentation_coverage"],
        "test_exact_plate_accuracy_all_rois_including_segmentation_failures": test_summary["plate_exact_accuracy_all_test_plates"],
        "test_exact_plate_95_ci_all_rois": test_summary["plate_exact_wilson_95_ci_all_test_plates"],
        "runtime_roi_exact_all_including_skips": test_summary["runtime_no_known_count"]["exact_accuracy_all_including_skips"],
        "runtime_roi_coverage": test_summary["runtime_no_known_count"]["coverage"],
        "macs_per_character": MACS_PER_CHARACTER,
        "metrics": str(args.metrics),
        "test_csv": str(args.test_csv),
        "confusion_matrix": str(confusion_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
