from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from train_independent_structured import (
    FolderChars,
    LeNet5Structured50,
    transfer_dense_to_structured,
)
from train_lenet5 import CLASS_NAMES, LeNet5, run_epoch


def model_macs(channels: tuple[int, int, int, int], class_count: int) -> int:
    c1, c2, h1, h2 = channels
    return c1 * 25 * 28 * 28 + c1 * c2 * 25 * 10 * 10 + c2 * 25 * h1 + h1 * h2 + h2 * class_count


def read_plate_validation_batches(data_root: Path, records: list[dict]):
    batches = []
    for record in records:
        tensors = []
        for relative in record["crop_files"]:
            with Image.open(data_root / relative) as image:
                pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
        batches.append((record, torch.stack(tensors)))
    return batches


def evaluate_plates(model, samples, device: torch.device):
    model.eval()
    class_index = {character: index for index, character in enumerate(CLASS_NAMES)}
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
    by_length = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
    correct_characters = total_characters = exact_plates = 0
    with torch.inference_mode():
        for record, inputs in samples:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu().tolist()
            predicted_text = "".join(CLASS_NAMES[index] for index in predictions)
            target = record["text"]
            matches = sum(expected == actual for expected, actual in zip(target, predicted_text))
            is_exact = predicted_text == target
            correct_characters += matches
            total_characters += len(target)
            exact_plates += int(is_exact)
            type_stats = by_type[record["type"]]
            type_stats["plates"] += 1
            type_stats["exact"] += int(is_exact)
            type_stats["characters"] += len(target)
            type_stats["correct"] += matches
            length_stats = by_length[str(len(target))]
            length_stats["plates"] += 1
            length_stats["exact"] += int(is_exact)
            length_stats["characters"] += len(target)
            length_stats["correct"] += matches
            for position, (expected, actual) in enumerate(zip(target, predicted_text), start=1):
                confusion[class_index[expected], class_index[actual]] += 1
                position_stats[str(position)]["total"] += 1
                position_stats[str(position)]["correct"] += int(expected == actual)

    def finalize(groups):
        return {
            key: {
                **stats,
                "character_accuracy": stats["correct"] / max(1, stats["characters"]),
                "exact_plate_accuracy": stats["exact"] / max(1, stats["plates"]),
            }
            for key, stats in sorted(groups.items())
        }

    class_report = []
    for index, character in enumerate(CLASS_NAMES):
        support = int(confusion[index].sum())
        correct = int(confusion[index, index])
        class_report.append({
            "character": character,
            "support": support,
            "correct": correct,
            "accuracy": correct / max(1, support),
        })
    return {
        "plates": len(samples),
        "characters": total_characters,
        "correct_characters": correct_characters,
        "character_accuracy": correct_characters / max(1, total_characters),
        "exact_plates": exact_plates,
        "exact_plate_accuracy": exact_plates / max(1, len(samples)),
        "by_plate_type": finalize(by_type),
        "by_label_length": finalize(by_length),
        "by_character_position": {
            position: {
                **stats,
                "accuracy": stats["correct"] / max(1, stats["total"]),
            }
            for position, stats in sorted(position_stats.items(), key=lambda item: int(item[0]))
        },
        "per_character": class_report,
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(path: Path, model, arch: str, epoch: int, val_accuracy: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = [6, 16, 120, 84] if arch == "LeNet5_dense" else list(LeNet5Structured50.channels)
    torch.save({
        "model": model.state_dict(),
        "classes": CLASS_NAMES,
        "arch": arch,
        "channels": channels,
        "epoch": epoch,
        "val_acc": val_accuracy,
        "macs_per_character": model_macs(tuple(channels), len(CLASS_NAMES)),
        "enhancement": args.enhancement,
        "seed": args.seed,
        "training_data": str(args.data),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Matched dense vs. 50%-channel-structured LeNet-5 training on one train(1) split.")
    parser.add_argument("--data", type=Path, default=Path("data/independent_chars_train1_structured_channel30_v1"))
    parser.add_argument("--preparation-metrics", type=Path, default=Path("artifacts/channel30_matched_structured_prepare.json"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/channel30_matched_structured_metrics.json"))
    parser.add_argument("--history-csv", type=Path, default=Path("artifacts/channel30_matched_structured_history.csv"))
    parser.add_argument("--confusion-csv", type=Path, default=Path("artifacts/channel30_matched_structured_confusion.csv"))
    parser.add_argument("--dense-output", type=Path, default=Path("artifacts/lenet5_dense_channel30_matched_v1.pt"))
    parser.add_argument("--structured-output", type=Path, default=Path("artifacts/lenet5_structured50_channel30_matched_v1.pt"))
    parser.add_argument("--epochs", type=int, default=10, help="Total dataset passes for both branches.")
    parser.add_argument("--prune-after-epoch", type=int, default=5, help="Transfer selected channels after this dense epoch; fine-tune for remaining epochs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--enhancement", choices=("none", "clahe", "clahe_sharp"), default="clahe_sharp")
    args = parser.parse_args()

    if not 1 <= args.prune_after_epoch < args.epochs:
        raise ValueError("prune-after-epoch must be between epoch 1 and epochs-1.")
    manifest_path = args.data / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared train(1) manifest not found: {manifest_path}")
    for output_path in (args.dense_output, args.structured_output):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(args.threads, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = FolderChars(args.data, "train", augment=True)
    val_set = FolderChars(args.data, "val", augment=False)
    if not train_set or not val_set:
        raise RuntimeError(f"Train or validation character split is empty: {args.data}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dense = LeNet5(len(CLASS_NAMES)).to(device)
    dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    structured = None
    structured_optimizer = None
    best_dense = -1.0
    best_structured = -1.0
    pruned_immediate_val = None
    history = []
    start_time = time.perf_counter()
    print(
        f"device={device} classes={len(CLASS_NAMES)} train_characters={len(train_set)} "
        f"val_characters={len(val_set)} epochs={args.epochs} prune_after={args.prune_after_epoch} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        dense_train_loss, dense_train_acc = run_epoch(dense, train_loader, loss_fn, dense_optimizer, device, True)
        dense_val_loss, dense_val_acc = run_epoch(dense, val_loader, loss_fn, dense_optimizer, device, False)
        if dense_val_acc > best_dense:
            best_dense = dense_val_acc
            save_checkpoint(args.dense_output, dense, "LeNet5_dense", epoch, dense_val_acc, args)

        if epoch == args.prune_after_epoch:
            structured = LeNet5Structured50(len(CLASS_NAMES)).to(device)
            transfer_dense_to_structured(dense, structured)
            structured_optimizer = torch.optim.AdamW(
                structured.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            _, pruned_immediate_val = run_epoch(structured, val_loader, loss_fn, None, device, False)

        structured_train_loss = structured_train_acc = structured_val_loss = structured_val_acc = None
        if epoch > args.prune_after_epoch:
            assert structured is not None and structured_optimizer is not None
            structured_train_loss, structured_train_acc = run_epoch(
                structured, train_loader, loss_fn, structured_optimizer, device, True
            )
            structured_val_loss, structured_val_acc = run_epoch(
                structured, val_loader, loss_fn, structured_optimizer, device, False
            )
            if structured_val_acc > best_structured:
                best_structured = structured_val_acc
                save_checkpoint(
                    args.structured_output,
                    structured,
                    "LeNet5Structured50",
                    epoch,
                    structured_val_acc,
                    args,
                )

        row = {
            "epoch": epoch,
            "dense_train_loss": dense_train_loss,
            "dense_train_character_accuracy": dense_train_acc,
            "dense_val_loss": dense_val_loss,
            "dense_val_character_accuracy": dense_val_acc,
            "structured_train_loss": structured_train_loss,
            "structured_train_character_accuracy": structured_train_acc,
            "structured_val_loss": structured_val_loss,
            "structured_val_character_accuracy": structured_val_acc,
            "structured_immediate_post_prune_val_accuracy": pruned_immediate_val if epoch == args.prune_after_epoch else None,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    elapsed_seconds = time.perf_counter() - start_time
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_records = [record for record in manifest if record["split"] == "val"]
    plate_samples = read_plate_validation_batches(args.data, val_records)
    dense_checkpoint = torch.load(args.dense_output, map_location=device, weights_only=False)
    dense.load_state_dict(dense_checkpoint["model"])
    dense_plate_metrics = evaluate_plates(dense, plate_samples, device)
    structured_checkpoint = torch.load(args.structured_output, map_location=device, weights_only=False)
    structured.load_state_dict(structured_checkpoint["model"])
    structured_plate_metrics = evaluate_plates(structured, plate_samples, device)

    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.history_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(history[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    args.confusion_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.confusion_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for character, row in zip(CLASS_NAMES, structured_plate_metrics["confusion_matrix"]):
            writer.writerow([character, *row])

    prepare_metrics_path = args.preparation_metrics
    prepare_metrics = json.loads(prepare_metrics_path.read_text(encoding="utf-8")) if prepare_metrics_path.exists() else None
    dense_params = sum(parameter.numel() for parameter in dense.parameters())
    structured_params = sum(parameter.numel() for parameter in structured.parameters())
    dense_macs = model_macs((6, 16, 120, 84), len(CLASS_NAMES))
    structured_macs = model_macs(LeNet5Structured50.channels, len(CLASS_NAMES))
    result = {
        "experiment": "train(1), 30 classes, matched total data passes, dense baseline vs. 50%-channel structured pruning",
        "source": "data/OCR/OCR/images/train(1)/detection",
        "data_root": args.data.as_posix(),
        "class_order": CLASS_NAMES,
        "preparation": prepare_metrics,
        "split": {
            "train_character_crops": len(train_set),
            "validation_character_crops": len(val_set),
            "validation_plates": len(val_records),
            "sample_predictions_not_exported": True,
        },
        "training": {
            "device": str(device),
            "epochs_total_per_branch": args.epochs,
            "dense_epochs_before_prune": args.prune_after_epoch,
            "structured_finetune_epochs": args.epochs - args.prune_after_epoch,
            "same_dataset_passes": True,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate_both_branches": args.learning_rate,
            "weight_decay_both_branches": args.weight_decay,
            "augmentation_and_preprocessing": "shared train(1) prepared crops; random ±6-degree crop rotation in training; clahe_sharp at crop preparation",
            "seed": args.seed,
            "elapsed_training_seconds": elapsed_seconds,
            "structured_accuracy_immediately_after_channel_prune": pruned_immediate_val,
        },
        "models": {
            "dense": {
                "checkpoint": args.dense_output.as_posix(),
                "channels": [6, 16, 120, 84],
                "parameters": dense_params,
                "macs_per_character": dense_macs,
                "best_validation_crop_character_accuracy": dense_checkpoint["val_acc"],
                "best_epoch": dense_checkpoint["epoch"],
                "validation_plate_metrics": dense_plate_metrics,
            },
            "structured_channel_50": {
                "checkpoint": args.structured_output.as_posix(),
                "channels": list(LeNet5Structured50.channels),
                "parameters": structured_params,
                "macs_per_character": structured_macs,
                "mac_reduction_fraction": 1.0 - structured_macs / dense_macs,
                "parameter_reduction_fraction": 1.0 - structured_params / dense_params,
                "best_validation_crop_character_accuracy": structured_checkpoint["val_acc"],
                "best_epoch": structured_checkpoint["epoch"],
                "validation_plate_metrics": structured_plate_metrics,
            },
        },
        "evaluation_scope": "OCR on accepted validation plate crops after preparation segmentation that used filename ground-truth character length; this is not a no-known-count runtime or full-frame camera metric.",
        "history_csv": args.history_csv.as_posix(),
        "structured_confusion_csv": args.confusion_csv.as_posix(),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dense_character_accuracy": dense_plate_metrics["character_accuracy"],
        "structured_character_accuracy": structured_plate_metrics["character_accuracy"],
        "dense_exact_plate_accuracy": dense_plate_metrics["exact_plate_accuracy"],
        "structured_exact_plate_accuracy": structured_plate_metrics["exact_plate_accuracy"],
        "dense_macs_per_character": dense_macs,
        "structured_macs_per_character": structured_macs,
        "metrics": args.metrics.as_posix(),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
