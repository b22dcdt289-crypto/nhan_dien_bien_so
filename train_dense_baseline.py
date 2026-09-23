from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from train_independent_structured import FolderChars, LeNet5, run_epoch
from train_lenet5 import CLASS_NAMES


DENSE_MACS_PER_CHARACTER = 418704


def load_model(path: Path, device: torch.device) -> LeNet5:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_plate(model: nn.Module, device: torch.device, data_root: Path, record: dict) -> str:
    tensors = []
    for relative in record["crop_files"]:
        with Image.open(data_root / relative) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
    with torch.no_grad():
        logits = model(torch.stack(tensors).to(device))
    return "".join(CLASS_NAMES[index] for index in logits.argmax(1).cpu().tolist())


def evaluate_detailed(model: nn.Module, device: torch.device, data_root: Path, checkpoint: Path) -> dict:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest if record["split"] == "val"]
    details = []
    aggregate = defaultdict(lambda: {"plates": 0, "exact": 0, "chars": 0, "correct_chars": 0, "macs": 0})
    for record in records:
        expected = record["text"]
        predicted = predict_plate(model, device, data_root, record)
        correct_chars = sum(a == b for a, b in zip(expected, predicted))
        exact = predicted == expected
        plate_type = record["type"]
        length_key = str(len(expected))
        for key in ("all", plate_type, f"length_{length_key}"):
            stats = aggregate[key]
            stats["plates"] += 1
            stats["exact"] += int(exact)
            stats["chars"] += len(expected)
            stats["correct_chars"] += correct_chars
            stats["macs"] += DENSE_MACS_PER_CHARACTER * len(expected)
        details.append(
            {
                "source": record["source"],
                "type": plate_type,
                "expected": expected,
                "predicted": predicted,
                "characters": len(expected),
                "correct_characters": correct_chars,
                "character_accuracy": correct_chars / max(1, len(expected)),
                "exact": exact,
                "row_count": record["row_count"],
                "perspective_corrected": record["perspective_corrected"],
                "angle_deg": record["angle_deg"],
                "blur_score": record["blur_score"],
                "macs_this_plate": DENSE_MACS_PER_CHARACTER * len(expected),
            }
        )

    def finish(stats: dict) -> dict:
        return {
            "plates": stats["plates"],
            "exact_plates": stats["exact"],
            "plate_exact_accuracy": stats["exact"] / max(1, stats["plates"]),
            "total_characters": stats["chars"],
            "correct_characters": stats["correct_chars"],
            "character_accuracy": stats["correct_chars"] / max(1, stats["chars"]),
            "total_macs": stats["macs"],
            "average_macs_per_plate": stats["macs"] / max(1, stats["plates"]),
        }

    return {
        "checkpoint": str(checkpoint),
        "model": "LeNet5_dense_no_pruning",
        "macs_per_character": DENSE_MACS_PER_CHARACTER,
        "groups": {key: finish(value) for key, value in sorted(aggregate.items())},
        "details": details,
    }


def write_csv(path: Path, details: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(details[0].keys()) if details else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a dense, non-pruned LeNet-5 baseline")
    parser.add_argument("--data", type=Path, default=Path("data/independent_chars_train1"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_dense_baseline_train1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/dense_baseline_detailed_metrics.json"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/dense_baseline_plate_results.csv"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not args.evaluate_only:
        random.seed(42)
        torch.manual_seed(42)
        train_set = FolderChars(args.data, "train", augment=True)
        val_set = FolderChars(args.data, "val", augment=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = LeNet5(len(CLASS_NAMES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        best_val = -1.0
        history = []
        print(f"device={device} train_crops={len(train_set)} val_crops={len(val_set)}", flush=True)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_acc >= best_val:
                best_val = val_acc
                torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "val_acc": val_acc, "arch": "LeNet5"}, args.output)
    else:
        history = []
        best_val = torch.load(args.output, map_location="cpu", weights_only=False).get("val_acc")

    model = load_model(args.output, device)
    evaluation = evaluate_detailed(model, device, args.data, args.output)
    evaluation["training"] = {
        "epochs": args.epochs if not args.evaluate_only else None,
        "best_validation_character_accuracy": best_val,
        "history": history,
    }

    prepare_path = Path("artifacts/independent_pipeline_metrics.json")
    if prepare_path.exists():
        prepare = json.loads(prepare_path.read_text(encoding="utf-8")).get("prepare", {})
        source_types = prepare.get("source_types", {})
        accepted_types = prepare.get("accepted_types", {})
        skip_by_type = {}
        for key, source_count in source_types.items():
            accepted_count = accepted_types.get(key, 0)
            skip_by_type[key] = {
                "source_images": source_count,
                "accepted_images": accepted_count,
                "skipped_images": source_count - accepted_count,
                "skip_rate": (source_count - accepted_count) / max(1, source_count),
            }
        evaluation["preparation_by_type"] = skip_by_type

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.csv, evaluation["details"])
    summary = dict(evaluation)
    summary.pop("details", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
