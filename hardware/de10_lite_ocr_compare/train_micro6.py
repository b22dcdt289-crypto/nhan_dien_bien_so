"""Matched six-plate, from-scratch dense/Conv2-pruned OCR experiment.

Only two plates from each of three groups contribute training labels. Evaluation
uses different plate texts, split into development and untouched test sets.
This measures the *data-scarcity failure mode*, not field OCR accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, compact_teacher_to_student
from train_lenet5 import CLASS_NAMES, LeNet5, run_epoch
from train_matched_channel_pruning import model_macs
from hardware.de10_lite_ocr_compare.train_compare import bench_forward

DATA = ROOT / "data/independent_chars_train1_structured_channel30_v1"
GROUPS = ("one_row", "two_row_car", "two_row_motorcycle")
CLASS_INDEX = {character: index for index, character in enumerate(CLASS_NAMES)}
# Visual audit: one source filed under two_rows actually shows a one-line car
# plate. Hash relative paths so public code does not disclose plate identities.
AUDITED_SOURCE_REPLACEMENT_SHA256 = {
    "84a039b493b48f6c9a7ffdc058c704e74326b64c7e1495ed7511a219892b833c":
    "2f09e0f2d12a2541fb2acc62b1fef4eeb772d73513a90ffa603e522ab07553bc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_key(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def choose_six(records: list[dict], seed: int) -> list[dict]:
    chosen: list[dict] = []
    used_texts: set[str] = set()
    for group in GROUPS:
        candidates = sorted(
            (record for record in records
             if record["split"] == "train" and record["type"] == group
             and record["angle_deg"] <= 3 and record["blur_score"] >= 300
             and len(record["text"]) == len(record["crop_files"])),
            key=lambda record: record["source"],
        )
        rng = random.Random(f"{seed}:{group}")
        rng.shuffle(candidates)
        for record in candidates:
            if record["text"] in used_texts:
                continue
            if any(character not in CLASS_INDEX for character in record["text"]):
                continue
            chosen.append(record)
            used_texts.add(record["text"])
            if sum(item["type"] == group for item in chosen) == 2:
                break
        else:
            raise RuntimeError(f"Not enough distinct eligible plates: {group}")
    by_source_key = {source_key(record["source"]): record
                     for record in records if record["split"] == "train"}
    for index, record in enumerate(chosen):
        replacement_key = AUDITED_SOURCE_REPLACEMENT_SHA256.get(source_key(record["source"]))
        if replacement_key:
            replacement = by_source_key[replacement_key]
            if replacement["type"] != record["type"] or replacement["text"] in used_texts:
                raise ValueError("Audited replacement is inconsistent")
            used_texts.remove(record["text"])
            used_texts.add(replacement["text"])
            chosen[index] = replacement
    return chosen


def split_evaluation(records: list[dict], excluded_texts: set[str], seed: int) -> tuple[list[dict], list[dict]]:
    dev, test = [], []
    for record in records:
        if record["split"] != "val" or record["text"] in excluded_texts:
            continue
        digest = hashlib.sha256(f"{seed}:holdout:{record['text']}".encode()).digest()
        (dev if digest[0] & 1 else test).append(record)
    if not dev or not test:
        raise RuntimeError("Empty development/test partition")
    if {r["text"] for r in dev} & {r["text"] for r in test}:
        raise AssertionError("Plate identity leaked between development and test")
    return dev, test


class PlateCrops(Dataset):
    def __init__(self, records: list[dict], augment: bool):
        self.augment = augment
        self.cache = []
        for record in records:
            for character, relative in zip(record["text"], record["crop_files"]):
                with Image.open(DATA / relative) as image:
                    pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
                if pixels.shape != (32, 32):
                    raise ValueError(f"Expected 32x32 crop: {relative}")
                self.cache.append((pixels, CLASS_INDEX[character]))

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int):
        pixels, label = self.cache[index]
        image = Image.fromarray(pixels)
        if self.augment and random.random() < 0.55:
            image = image.rotate(random.uniform(-6, 6), resample=Image.Resampling.BILINEAR, fillcolor=0)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(array).unsqueeze(0), label


class EvalCrops:
    def __init__(self, records: list[dict]):
        self.records = records
        self.spans = []
        pixels = []
        labels = []
        for record in records:
            start = len(labels)
            for character, relative in zip(record["text"], record["crop_files"]):
                with Image.open(DATA / relative) as image:
                    array = np.asarray(image.convert("L"), dtype=np.uint8)
                if array.shape != (32, 32):
                    raise ValueError(f"Expected 32x32 crop: {relative}")
                pixels.append(array.copy())
                labels.append(CLASS_INDEX[character])
            self.spans.append((start, len(labels)))
        self.pixels = torch.from_numpy(np.stack(pixels).astype(np.float32) / 127.5 - 1).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def evaluate(self, model: nn.Module) -> dict:
        model.eval()
        predictions: list[int] = []
        loss_total = 0.0
        with torch.inference_mode():
            for start in range(0, len(self.labels), 256):
                end = min(start + 256, len(self.labels))
                logits = model(self.pixels[start:end])
                loss_total += nn.functional.cross_entropy(logits, self.labels[start:end], reduction="sum").item()
                predictions.extend(logits.argmax(1).tolist())
        pred = np.asarray(predictions)
        true = self.labels.numpy()
        confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        np.add.at(confusion, (true, pred), 1)
        type_counts = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
        exact = 0
        for record, (start, end) in zip(self.records, self.spans):
            matches = int((pred[start:end] == true[start:end]).sum())
            plate_exact = int(matches == end - start)
            exact += plate_exact
            group = type_counts[record["type"]]
            group["plates"] += 1
            group["exact"] += plate_exact
            group["characters"] += end - start
            group["correct"] += matches
        by_type = {
            group: {**counts,
                    "character_accuracy": counts["correct"] / counts["characters"],
                    "exact_plate_accuracy": counts["exact"] / counts["plates"]}
            for group, counts in sorted(type_counts.items())
        }
        return {
            "plates": len(self.records), "characters": len(self.labels),
            "correct_characters": int((pred == true).sum()),
            "character_accuracy": float((pred == true).mean()),
            "exact_plates": exact, "exact_plate_accuracy": exact / len(self.records),
            "cross_entropy": loss_total / len(self.labels),
            "by_plate_type": by_type,
            "confusion_matrix": confusion.tolist(),
        }


def seen_unseen_accuracy(metrics: dict, seen: set[str]) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"])
    result = {}
    for key, indices in (
        ("seen", [CLASS_INDEX[c] for c in CLASS_NAMES if c in seen]),
        ("unseen", [CLASS_INDEX[c] for c in CLASS_NAMES if c not in seen]),
    ):
        total = int(matrix[indices].sum())
        correct = int(sum(matrix[i, i] for i in indices))
        result[key] = {"correct": correct, "total": total,
                       "accuracy": correct / total if total else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/compare_micro6_verified_20260930")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--prune-after", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.prune_after < args.epochs or args.eval_every < 1:
        raise ValueError("Invalid training schedule")
    args.out.mkdir(parents=True, exist_ok=True)
    protected = [args.out / name for name in ("dense.pt", "conv2_pruned.pt", "history.csv", "metrics.json")]
    if not args.select_only and any(path.exists() for path in protected):
        raise FileExistsError("Refusing to overwrite existing experiment results")
    manifest_path = DATA / "manifest.json"
    all_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    chosen = choose_six(all_records, args.seed)
    selected_texts = {record["text"] for record in chosen}
    dev_records, test_records = split_evaluation(all_records, selected_texts, args.seed)
    selection_file = args.out / "selection_private.json"
    selection = {"seed": args.seed, "data_manifest_sha256": sha256(manifest_path),
                 "selected": chosen, "dev_plate_count": len(dev_records),
                 "test_plate_count": len(test_records)}
    if selection_file.exists():
        if json.loads(selection_file.read_text(encoding="utf-8")) != selection:
            raise FileExistsError("Existing selection differs from deterministic selection")
    else:
        selection_file.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    class_counts = Counter("".join(record["text"] for record in chosen))
    print(json.dumps({"event": "selected", "groups": dict(Counter(r["type"] for r in chosen)),
                      "train_characters": sum(class_counts.values()),
                      "represented_classes": len(class_counts),
                      "missing_classes": "".join(c for c in CLASS_NAMES if c not in class_counts),
                      "dev_plates": len(dev_records), "test_plates": len(test_records),
                      "selection_private": str(selection_file)}, ensure_ascii=False), flush=True)
    if args.select_only:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start_total = time.perf_counter()
    train_set = PlateCrops(chosen, augment=True)
    dev = EvalCrops(dev_records)
    test = EvalCrops(test_records)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dense = LeNet5(len(CLASS_NAMES))
    pruned = None
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=5e-4, weight_decay=1e-5)
    pruned_opt = None
    loss_fn = nn.CrossEntropyLoss()
    best = {"dense": (-1.0, 0), "conv2_pruned": (-1.0, 0)}
    history = []
    selected_channels = None
    post_prune = None
    for epoch in range(1, args.epochs + 1):
        for name in (["dense"] if epoch <= args.prune_after else ["dense", "conv2_pruned"]):
            model = dense if name == "dense" else pruned
            optimizer = dense_opt if name == "dense" else pruned_opt
            assert model is not None and optimizer is not None
            random.seed(args.seed + 1000 * epoch)
            torch.manual_seed(args.seed + 1000 * epoch)
            before = time.perf_counter()
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, torch.device("cpu"), True)
            train_seconds = time.perf_counter() - before
            eval_result = None
            eval_seconds = 0.0
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                before = time.perf_counter()
                eval_result = dev.evaluate(model)
                eval_seconds = time.perf_counter() - before
                if epoch > args.prune_after and eval_result["character_accuracy"] > best[name][0]:
                    best[name] = (eval_result["character_accuracy"], epoch)
                    torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                                "arch": name, "epoch": epoch, "seed": args.seed,
                                "dev_character_accuracy": eval_result["character_accuracy"],
                                "six_plate_training_only": True}, args.out / (name + ".pt"))
            row = {"epoch": epoch, "branch": name,
                   "phase": "shared_pretraining" if epoch <= args.prune_after else "continuation",
                   "train_loss": train_loss, "train_character_accuracy": train_acc,
                   "dev_loss": eval_result["cross_entropy"] if eval_result else "",
                   "dev_character_accuracy": eval_result["character_accuracy"] if eval_result else "",
                   "dev_exact_plate_accuracy": eval_result["exact_plate_accuracy"] if eval_result else "",
                   "train_seconds": train_seconds, "dev_seconds": eval_seconds}
            history.append(row)
            if eval_result or epoch == 1:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch == args.prune_after:
            pruned = LeNet5Conv2Pruned(len(CLASS_NAMES))
            selected_channels = compact_teacher_to_student(dense, pruned)
            pruned.features[0].weight.requires_grad_(True)
            pruned.features[0].bias.requires_grad_(True)
            dense_opt = torch.optim.AdamW(dense.parameters(), lr=5e-4, weight_decay=1e-5)
            pruned_opt = torch.optim.AdamW(pruned.parameters(), lr=5e-4, weight_decay=1e-5)
            post_prune = dev.evaluate(pruned)
            print(json.dumps({"event": "prune", "epoch": epoch,
                              "selected_conv2_channels": selected_channels,
                              "immediate_dev_character_accuracy": post_prune["character_accuracy"]}), flush=True)

    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final_models = {}
    for name, model in (("dense", dense), ("conv2_pruned", pruned)):
        assert model is not None
        checkpoint_path = args.out / (name + ".pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        train_eval = EvalCrops(chosen).evaluate(model)
        dev_eval = dev.evaluate(model)
        test_eval = test.evaluate(model)
        channels = (6, 16, 120, 84) if name == "dense" else (6, 8, 120, 84)
        final_models[name] = {"checkpoint_sha256": sha256(checkpoint_path),
                              "best_epoch": best[name][1],
                              "parameters": sum(parameter.numel() for parameter in model.parameters()),
                              "macs_per_character": model_macs(channels, len(CLASS_NAMES)),
                              "train": train_eval, "development": dev_eval, "test": test_eval,
                              "seen_unseen_test": seen_unseen_accuracy(test_eval, set(class_counts))}
    cpu_forward = bench_forward({"dense": dense, "conv2_pruned": pruned}, 4, 500)
    for name in final_models:
        final_models[name]["cpu_forward"] = cpu_forward[name]
    metrics = {
        "experiment": "Six source plates only, two per type, from-scratch matched dense versus 50%-Conv2-channel-pruned LeNet-5",
        "source_data_manifest_sha256": sha256(manifest_path),
        "seed": args.seed, "epochs_per_branch": args.epochs,
        "shared_epochs": args.prune_after, "continuation_epochs": args.epochs - args.prune_after,
        "batch_size": args.batch_size, "lr": 5e-4, "weight_decay": 1e-5,
        "augmentation": "55% random rotation uniformly [-6,6] degrees per crop, train only",
        "split": "six train plates; validation source split further divided by plate text hash, no selected text in dev/test",
        "train_plates_by_type": dict(Counter(record["type"] for record in chosen)),
        "train_plate_count": len(chosen), "train_character_count": sum(class_counts.values()),
        "train_class_counts": {character: class_counts.get(character, 0) for character in CLASS_NAMES},
        "represented_class_count": len(class_counts),
        "missing_classes": [character for character in CLASS_NAMES if character not in class_counts],
        "dev_plate_count": len(dev_records), "test_plate_count": len(test_records),
        "dev_text_hashes_sha256": hashlib.sha256("|".join(sorted({r["text"] for r in dev_records})).encode()).hexdigest(),
        "test_text_hashes_sha256": hashlib.sha256("|".join(sorted({r["text"] for r in test_records})).encode()).hexdigest(),
        "post_prune_dev_character_accuracy": post_prune["character_accuracy"] if post_prune else None,
        "selected_conv2_channel_indices": selected_channels,
        "wall_seconds": time.perf_counter() - start_total,
        "models": final_models,
        "limitations": [
            "Only two source plates per type train the classifier; missing character classes receive no positive examples.",
            "Crops are label-informed and plate texts derive from source filenames; this is not end-to-end plate recognition.",
            "Development and test plate texts are disjoint, but the original source has possible repeated identities across its train/validation frames.",
            "No FPGA inference of these six-plate-trained checkpoints is claimed by this script.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "completed", "out": str(args.out),
                      "test": {name: {"character": values["test"]["character_accuracy"],
                                      "plate": values["test"]["exact_plate_accuracy"]}
                               for name, values in final_models.items()},
                      "wall_seconds": metrics["wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
