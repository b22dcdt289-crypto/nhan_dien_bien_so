"""Matched dense/Conv2-channel-pruned LeNet-5 on 3 generated plate layouts.

Stage 1 uses generated plates only. Stage 2 mixes generated train plates with
real train-split character crops; selection uses disjoint real validation
identities, and evaluation uses held-out real identities. Image localization
is not learned or implemented by this character classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torch import nn

from prepare_two_row_frontal_dense import identity_split
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, compact_teacher_to_student
from train_independent_structured import parse_source_name
from train_lenet5 import CLASS_NAMES, CLASS_TO_INDEX, LeNet5
from lenet5_k3 import LeNet5K3, LeNet5K3Pruned, compact_k3, model_size_k3
from train_synthetic_car_compare import (
    as_tensors, bench, epoch_batches, evaluate_crops, model_size, plate_metrics,
    sha, synthetic_segment, train_pass,
)


LAYOUTS = ("one_row_car", "two_row_car", "two_row_motorcycle")
TYPE = {"one_row_car": "one_row", "two_row_car": "two_row_car",
        "two_row_motorcycle": "two_row_motorcycle"}


def generated_plates(car_root: Path, motorcycle_root: Path, seed: int):
    splits = {key: [] for key in ("train", "val", "test")}
    for root, layouts in ((car_root, LAYOUTS[:2]), (motorcycle_root, LAYOUTS[2:])):
        rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
        for layout in layouts:
            group = [row for row in rows if row["layout"] == layout]
            if len(group) != 1000 or len({row["label"] for row in group}) != 1000:
                raise RuntimeError(f"Expected 1000 unique {layout} plates")
            group.sort(key=lambda row: hashlib.sha256(f"{seed}|{row['label']}".encode()).hexdigest())
            for index, row in enumerate(group):
                split = "train" if index < 800 else "val" if index < 900 else "test"
                gray = cv2.imread(str(root / row["file"]), cv2.IMREAD_GRAYSCALE)
                crops = synthetic_segment(gray, layout) if gray is not None else None
                if crops is None or len(crops) != len(row["label"]):
                    raise RuntimeError(f"Cannot segment generated image: {root / row['file']}")
                if identity_split(row["label"]) != "train":
                    raise RuntimeError("Generated source labels must be in original train identity split")
                splits[split].append({"record": {"label": row["label"], "layout": layout,
                                                 "file": row["file"], "root": str(root)}, "crops": crops})
    identities = {key: {p["record"]["label"] for p in plates} for key, plates in splits.items()}
    if any(identities[a] & identities[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Generated identity leakage between splits")
    return splits


def real_plates(root: Path, layout: str, split: str, count: int, seed: int,
                exclude: set[str]) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    unique = {}
    for record in manifest:
        text = record["text"]
        if (identity_split(text) != split or record["type"] != TYPE[layout]
                or record["row_count"] != (1 if layout == "one_row_car" else 2)
                or len(text) != (9 if layout == "two_row_motorcycle" else 8)
                or len(record["crop_files"]) != len(text) or text in exclude
                or any(char not in CLASS_TO_INDEX for char in text)):
            continue
        previous = unique.get(text)
        if previous is None or record["blur_score"] > previous["blur_score"]:
            unique[text] = record
    ranked = sorted(unique.values(), key=lambda row: hashlib.sha256(
        f"{seed}|{layout}|{split}|{row['text']}".encode()).hexdigest())
    if len(ranked) < count:
        raise RuntimeError(f"Only {len(ranked)} {layout} {split} real identities, need {count}")
    plates = []
    for record in ranked[:count]:
        crops = []
        for filename in record["crop_files"]:
            with Image.open(root / filename) as image:
                crop = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if crop.shape != (32, 32):
                crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            crops.append(crop)
        plates.append({"record": {"label": record["text"], "layout": layout,
                                  "source": record["source"]}, "crops": crops})
    return plates


def score_presegmented(model: nn.Module, plates: list[dict]) -> dict:
    x, y = as_tensors(plates)
    crop, ids, confidences = evaluate_crops(model, x, y)
    by_layout = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    offset = 0
    for plate in plates:
        truth = plate["record"]["label"]
        pred = "".join(CLASS_NAMES[i] for i in ids[offset:offset + len(truth)])
        offset += len(truth)
        row = by_layout[plate["record"]["layout"]]
        row["plates"] += 1
        row["exact"] += int(pred == truth)
        row["characters"] += len(truth)
        row["correct"] += sum(a == b for a, b in zip(truth, pred))
        for a, b in zip(truth, pred):
            confusion[CLASS_TO_INDEX[a], CLASS_TO_INDEX[b]] += 1
    total = {key: sum(row[key] for row in by_layout.values())
             for key in ("plates", "exact", "characters", "correct")}
    for row in [*by_layout.values(), total]:
        row["character_accuracy"] = row["correct"] / row["characters"]
        row["plate_exact_accuracy"] = row["exact"] / row["plates"]
    return {"total": total, "by_layout": dict(by_layout), "crop": crop,
            "confidence_median": float(np.median(confidences)), "confusion_matrix": confusion.tolist()}


def annotated_box_plates(plates: list[dict], source_root: Path) -> list[dict]:
    result = []
    for plate in plates:
        source = source_root / plate["record"]["source"]
        parsed = parse_source_name(source)
        if parsed is None:
            continue
        text, x1, y1, x2, y2, _ = parsed
        if text != plate["record"]["label"]:
            continue
        result.append({"path": source, "text": text, "box": (x1, y1, x2, y2),
                       "layout": plate["record"]["layout"]})
    return result


def save_checkpoint(path: Path, model: nn.Module, name: str, epoch: int, seed: int,
                    manifests: dict[str, str], stage: str, kernel_size: int) -> None:
    torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                "channels": [6, 16 if name == "dense" else 8, 120, 84],
                "kernel_size": kernel_size, "epoch": epoch, "seed": seed,
                "stage": stage, "manifests_sha256": manifests}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car", type=Path, default=Path("data/car_frontal_synthetic_2x1000_v1"))
    parser.add_argument("--motorcycle", type=Path, default=Path("data/motorcycle_frontal_synthetic_1000_v2"))
    parser.add_argument("--prepared", type=Path, default=Path("data/independent_chars_train1_structured_channel30_v1"))
    parser.add_argument("--photo-root", type=Path, default=Path("data/OCR/OCR/images/train(1)/detection"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/three_layout_dense_pruned_20261009"))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--prune-after", type=int, default=4)
    parser.add_argument("--mixed-epochs", type=int, default=10)
    parser.add_argument("--real-train-per-layout", type=int, default=600)
    parser.add_argument("--real-val-per-layout", type=int, default=50)
    parser.add_argument("--real-test-per-layout", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, choices=(3, 5), default=5,
                        help="3x3 valid-convolution ablation or the existing 5x5 baseline")
    args = parser.parse_args()
    if not 1 <= args.prune_after < args.pretrain_epochs:
        raise ValueError("Invalid prune-after")
    for name in ("dense.pt", "pruned.pt", "metrics.json"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    wall_start = time.perf_counter()
    synth = generated_plates(args.car, args.motorcycle, args.seed)
    excluded = {p["record"]["label"] for split in ("val", "test") for p in synth[split]}
    real_train = sum((real_plates(args.prepared, layout, "train", args.real_train_per_layout,
                                  args.seed, excluded) for layout in LAYOUTS), [])
    real_val = sum((real_plates(args.prepared, layout, "val", args.real_val_per_layout,
                                args.seed, set()) for layout in LAYOUTS), [])
    real_test = sum((real_plates(args.prepared, layout, "test", args.real_test_per_layout,
                                 args.seed, set()) for layout in LAYOUTS), [])
    ids = {"train": {p["record"]["label"] for p in synth["train"] + real_train},
           "val": {p["record"]["label"] for p in synth["val"] + real_val},
           "test": {p["record"]["label"] for p in synth["test"] + real_test}}
    overlaps = {f"{a}_{b}": len(ids[a] & ids[b]) for a, b in
                (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise RuntimeError(f"Identity leakage: {overlaps}")
    print(json.dumps({"event": "data", "synthetic_split": {k: len(v) for k, v in synth.items()},
                      "real_train": len(real_train), "real_val": len(real_val),
                      "real_test": len(real_test), "identity_overlap": overlaps}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {"cars": sha(args.car / "manifest.csv"), "motorcycles": sha(args.motorcycle / "manifest.csv"),
                 "real_prepared": sha(args.prepared / "manifest.json")}
    x_syn, y_syn = as_tensors(synth["train"])
    x_mixed, y_mixed = as_tensors(synth["train"] + real_train)
    x_val, y_val = as_tensors(real_val)
    dense = LeNet5K3(len(CLASS_NAMES)) if args.kernel_size == 3 else LeNet5(len(CLASS_NAMES))
    pruned = None
    opt_dense = torch.optim.AdamW(dense.parameters(), lr=7e-4, weight_decay=1e-5)
    opt_pruned = None
    history = []
    selected = None
    for epoch in range(1, args.pretrain_epochs + 1):
        batches = epoch_batches(x_syn, y_syn, epoch, args.seed, args.batch_size)
        for name, model, optimizer in (("dense", dense, opt_dense), ("pruned", pruned, opt_pruned)):
            if model is None:
                continue
            train = train_pass(model, batches, optimizer, nn.CrossEntropyLoss())
            val, _, _ = evaluate_crops(model, x_val, y_val)
            record = {"stage": "generated_only", "epoch": epoch, "model": name,
                      "train_loss": train["loss"], "train_char_accuracy": train["accuracy"],
                      "real_val_char_accuracy": val["accuracy"], "train_seconds": train["seconds"]}
            history.append(record)
            print(json.dumps(record), flush=True)
        if epoch == args.prune_after:
            pruned = (LeNet5K3Pruned(len(CLASS_NAMES)) if args.kernel_size == 3
                      else LeNet5Conv2Pruned(len(CLASS_NAMES)))
            selected = (compact_k3(dense, pruned) if args.kernel_size == 3
                        else compact_teacher_to_student(dense, pruned))
            pruned.features[0].weight.requires_grad_(True)
            pruned.features[0].bias.requires_grad_(True)
            opt_dense = torch.optim.AdamW(dense.parameters(), lr=7e-4, weight_decay=1e-5)
            opt_pruned = torch.optim.AdamW(pruned.parameters(), lr=7e-4, weight_decay=1e-5)
            print(json.dumps({"event": "pruned_conv2", "channels": "16->8", "indices": selected}), flush=True)
    models = {"dense": dense, "pruned": pruned}
    pre_mixed = {name: {"generated_test": score_presegmented(model, synth["test"]),
                        "real_test": score_presegmented(model, real_test)}
                 for name, model in models.items()}
    best = {name: (-1.0, -1.0, 0) for name in models}
    for epoch in range(1, args.mixed_epochs + 1):
        batches = epoch_batches(x_mixed, y_mixed, 100 + epoch, args.seed, args.batch_size)
        for name, model in models.items():
            optimizer = opt_dense if name == "dense" else opt_pruned
            train = train_pass(model, batches, optimizer, nn.CrossEntropyLoss())
            val = score_presegmented(model, real_val)["total"]
            record = {"stage": "generated_plus_real", "epoch": epoch, "model": name,
                      "train_loss": train["loss"], "train_char_accuracy": train["accuracy"],
                      "real_val_char_accuracy": val["character_accuracy"],
                      "real_val_plate_accuracy": val["plate_exact_accuracy"],
                      "train_seconds": train["seconds"]}
            history.append(record)
            print(json.dumps(record), flush=True)
            candidate = (val["character_accuracy"], val["plate_exact_accuracy"], epoch)
            if candidate[:2] > best[name][:2]:
                best[name] = candidate
                save_checkpoint(args.output / f"{name}.pt", model, name, epoch, args.seed,
                                manifests, "generated_plus_real", args.kernel_size)
    output = {"run": {"seed": args.seed, "kernel_size": args.kernel_size,
                       "pretrain_epochs": args.pretrain_epochs,
                       "prune_after": args.prune_after, "mixed_epochs": args.mixed_epochs,
                       "synthetic_split_plates": {k: len(v) for k, v in synth.items()},
                       "real_train_plates": len(real_train), "real_val_plates": len(real_val),
                       "real_test_plates": len(real_test), "identity_overlap": overlaps,
                       "manifest_sha256": manifests, "pruned_conv2_indices": selected,
                       "train_wall_seconds_before_final_evaluation": time.perf_counter() - wall_start,
                       "process_rss_bytes": psutil.Process().memory_info().rss,
                       "real_test_scope": "Prepared glyphs and separately annotated-box plate-to-text; no automatic full-frame localization."},
              "models": {}}
    boxes = annotated_box_plates(real_test, args.photo_root)
    for name, model in models.items():
        checkpoint = torch.load(args.output / f"{name}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        final_synth = score_presegmented(model, synth["test"])
        final_real = score_presegmented(model, real_test)
        end_to_end = plate_metrics(model, boxes, synthetic=False) if boxes else None
        output["models"][name] = {"pre_mixed": pre_mixed[name],
                                  "selected_real_val_character_accuracy": best[name][0],
                                  "selected_real_val_plate_accuracy": best[name][1],
                                  "best_mixed_epoch": best[name][2],
                                  "generated_test": final_synth, "real_presegmented_test": final_real,
                                  "real_annotated_box_test": end_to_end,
                                  "size": (model_size_k3(model) if args.kernel_size == 3 else
                                           model_size(model, (6, 16 if name == "dense" else 8, 120, 84))),
                                  "cpu_forward": bench(model, repeats=100),
                                  "checkpoint": str(args.output / f"{name}.pt"),
                                  "checkpoint_sha256": sha(args.output / f"{name}.pt")}
        print(json.dumps({"event": "final_evaluation", "model": name,
                          "synthetic_plate": final_synth["total"]["plate_exact_accuracy"],
                          "real_presegmented_plate": final_real["total"]["plate_exact_accuracy"],
                          "real_annotated_box_plate": None if end_to_end is None else end_to_end["total"]["plate_exact_accuracy"]}), flush=True)
        for domain, result in (("generated", final_synth), ("real", final_real)):
            with (args.output / f"{name}_{domain}_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["true/pred", *CLASS_NAMES])
                for char, row in zip(CLASS_NAMES, result["confusion_matrix"]):
                    writer.writerow([char, *row])
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "metrics": str(args.output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
