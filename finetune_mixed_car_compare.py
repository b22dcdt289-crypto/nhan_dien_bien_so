"""Fair dense/pruned continuation using synthetic cars plus real train(1) crops.

The first four shared epochs are the saved synthetic-only dense checkpoint.
Both branches then see the same mixed batches, with best epoch chosen on a
real identity-disjoint validation split. Real camera-frame localization is
outside scope; source bounding boxes are supplied for the end-to-end check.
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
from train_lenet5 import CLASS_NAMES, CLASS_TO_INDEX, LeNet5
from train_synthetic_car_compare import (
    DATA, REAL, PATTERN, as_tensors, bench, edit_distance, epoch_batches,
    evaluate_crops, load_synthetic, model_size, plate_metrics, real_holdout,
    sha, train_pass,
)


PREPARED = Path("data/independent_chars_train1_structured_channel30_v1")
PRETRAINED = Path("artifacts/synthetic_car_dense_pruned_20261001/dense.pt")
OUT = Path("artifacts/mixed_car_dense_pruned_20261001")


def prepared_cars(root: Path, split: str, per_layout: int, seed: int,
                  exclude: set[str]) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for kind, layout, row_count in (("one_row", "one_row_car", 1),
                                    ("two_row_car", "two_row_car", 2)):
        unique = {}
        for record in manifest:
            text = record["text"]
            if (record["type"] != kind or record["row_count"] != row_count
                    or len(record["crop_files"]) != 8 or not PATTERN.fullmatch(text)
                    or identity_split(text) != split or text in exclude):
                continue
            previous = unique.get(text)
            if previous is None or record["blur_score"] > previous["blur_score"]:
                unique[text] = record
        ranked = sorted(unique.values(), key=lambda row:
                        hashlib.sha256(f"{seed}|{split}|{layout}|{row['text']}".encode()).hexdigest())
        if len(ranked) < per_layout:
            raise RuntimeError(f"Only {len(ranked)} prepared real {kind} {split} identities")
        count = 0
        for record in ranked:
            crops = []
            for filename in record["crop_files"]:
                path = root / filename
                with Image.open(path) as image:
                    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
                if array.shape != (32, 32):
                    array = cv2.resize(array, (32, 32), interpolation=cv2.INTER_AREA)
                crops.append(array)
            result.append({"record": {"label": record["text"], "layout": layout,
                                      "source": record["source"]}, "crops": crops})
            count += 1
            if count == per_layout:
                break
    return result


def prepared_plate_metrics(model: nn.Module, plates: list[dict]) -> dict:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    by_layout = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
    per_class = Counter()
    for plate in plates:
        truth = plate["record"]["label"]
        inputs = torch.from_numpy(np.stack(plate["crops"]).astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
        model.eval()
        with torch.inference_mode():
            pred = model(inputs).argmax(dim=1).tolist()
        text = "".join(CLASS_NAMES[index] for index in pred)
        group = by_layout[plate["record"]["layout"]]
        group["plates"] += 1
        group["exact"] += int(text == truth)
        group["characters"] += len(truth)
        group["correct"] += sum(a == b for a, b in zip(truth, text))
        for a, b in zip(truth, text):
            confusion[CLASS_TO_INDEX[a], CLASS_TO_INDEX[b]] += 1
            per_class[a] += 1
    total = {key: sum(group[key] for group in by_layout.values()) for key in ("plates", "exact", "characters", "correct")}
    total["character_accuracy"] = total["correct"] / total["characters"]
    total["plate_exact_accuracy"] = total["exact"] / total["plates"]
    for group in by_layout.values():
        group["character_accuracy"] = group["correct"] / group["characters"]
        group["plate_exact_accuracy"] = group["exact"] / group["plates"]
    return {"total": total, "by_layout": dict(by_layout), "per_class_support": dict(per_class),
            "confusion_matrix": confusion.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--real", type=Path, default=REAL)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--real-train-per-layout", type=int, default=1000)
    parser.add_argument("--real-val-per-layout", type=int, default=200)
    parser.add_argument("--real-test-per-layout", type=int, default=200)
    parser.add_argument("--external-test-per-layout", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    for path in (args.output / "dense.pt", args.output / "pruned.pt", args.output / "metrics.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    synthetic, synthetic_rows = load_synthetic(args.data, 20261001)
    heldout_synthetic = {row["label"] for split in ("val", "test") for row in synthetic_rows[split]}
    real_train = prepared_cars(args.prepared, "train", args.real_train_per_layout, args.seed, heldout_synthetic)
    real_val = prepared_cars(args.prepared, "val", args.real_val_per_layout, args.seed, set())
    real_test = prepared_cars(args.prepared, "test", args.real_test_per_layout, args.seed, set())
    identities = {
        "train": {plate["record"]["label"] for plate in real_train}
                 | {plate["record"]["label"] for plate in synthetic["train"]},
        "val": {plate["record"]["label"] for plate in real_val}
               | {plate["record"]["label"] for plate in synthetic["val"]},
        "test": {plate["record"]["label"] for plate in real_test}
                | {plate["record"]["label"] for plate in synthetic["test"]},
    }
    overlaps = {"train_val": len(identities["train"] & identities["val"]),
                "train_test": len(identities["train"] & identities["test"]),
                "val_test": len(identities["val"] & identities["test"])}
    if any(overlaps.values()):
        raise RuntimeError(f"Identity leakage: {overlaps}")
    print(json.dumps({"event": "data", "synthetic_train": len(synthetic["train"]),
                      "real_train": len(real_train), "real_val": len(real_val),
                      "real_test_prepared": len(real_test), "identity_overlap": overlaps}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    x_train, y_train = as_tensors(synthetic["train"] + real_train)
    x_val, y_val = as_tensors(real_val)
    x_synth_test, y_synth_test = as_tensors(synthetic["test"])
    x_real_test, y_real_test = as_tensors(real_test)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if checkpoint.get("classes") != CLASS_NAMES:
        raise RuntimeError("Synthetic pretrained checkpoint has wrong class order")
    dense = LeNet5(len(CLASS_NAMES))
    dense.load_state_dict(checkpoint["model"])
    pruned = LeNet5Conv2Pruned(len(CLASS_NAMES))
    selected = compact_teacher_to_student(dense, pruned)
    pruned.features[0].weight.requires_grad_(True)
    pruned.features[0].bias.requires_grad_(True)
    models = {"dense": dense, "pruned": pruned}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
                  for name, model in models.items()}
    best = {name: (-1.0, float("inf"), 0) for name in models}
    history = []
    start_all = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        batches = epoch_batches(x_train, y_train, epoch, args.seed, args.batch_size)
        for name, model in models.items():
            train = train_pass(model, batches, optimizers[name], nn.CrossEntropyLoss())
            val, _pred, _confidence = evaluate_crops(model, x_val, y_val)
            record = {"epoch": epoch, "model": name,
                      "train_loss": train["loss"], "train_char_accuracy": train["accuracy"],
                      "real_val_loss": val["loss"], "real_val_char_accuracy": val["accuracy"],
                      "train_seconds": train["seconds"], "real_val_seconds": val["seconds"]}
            history.append(record)
            print(json.dumps(record), flush=True)
            if (val["accuracy"], -val["loss"]) > (best[name][0], -best[name][1]):
                best[name] = (val["accuracy"], val["loss"], epoch)
                torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                            "channels": [6, 16 if name == "dense" else 8, 120, 84],
                            "epoch": epoch, "seed": args.seed,
                            "synthetic_manifest_sha256": sha(args.data / "manifest.csv"),
                            "real_manifest_sha256": sha(args.prepared / "manifest.json")},
                           args.output / f"{name}.pt")
    for name, model in models.items():
        model.load_state_dict(torch.load(args.output / f"{name}.pt", map_location="cpu", weights_only=False)["model"])
    excluded = identities["train"] | identities["val"] | identities["test"]
    external_one = real_holdout(args.real, "one_row_car", args.external_test_per_layout, excluded, args.seed)
    external_two = real_holdout(args.real, "two_row_car", args.external_test_per_layout,
                                excluded | {row["text"] for row in external_one}, args.seed)
    external = external_one + external_two
    output = {
        "run": {"seed": args.seed, "continuation_epochs_per_branch": args.epochs,
                "source_checkpoint": str(args.pretrained), "source_checkpoint_sha256": sha(args.pretrained),
                "synthetic_manifest_sha256": sha(args.data / "manifest.csv"),
                "real_prepared_manifest_sha256": sha(args.prepared / "manifest.json"),
                "data": {"synthetic_train_plates": len(synthetic["train"]), "real_train_plates": len(real_train),
                         "real_validation_plates": len(real_val), "synthetic_test_plates": len(synthetic["test"]),
                         "real_prepared_test_plates": len(real_test), "real_external_test_plates": len(external),
                         "identity_overlap": overlaps},
                "train_wall_seconds_before_evaluation": time.perf_counter() - start_all,
                "selected_conv2_indices": selected,
                "peak_process_rss_bytes_at_report": psutil.Process().memory_info().rss,
                "batch_size": args.batch_size, "cpu_threads": args.threads,
                "real_external_scope": "Photo -> annotated plate bbox -> homography -> segmentation without label count -> OCR; not full-frame automatic detection."},
        "models": {},
    }
    for name, model in models.items():
        synth_crop, _, _ = evaluate_crops(model, x_synth_test, y_synth_test)
        real_crop, _, _ = evaluate_crops(model, x_real_test, y_real_test)
        prepared = prepared_plate_metrics(model, real_test)
        end_to_end = plate_metrics(model, external, synthetic=False)
        size = model_size(model, (6, 16 if name == "dense" else 8, 120, 84))
        output["models"][name] = {
            "best_real_val_char_accuracy": best[name][0], "best_epoch": best[name][2],
            "synthetic_test_crop": synth_crop, "real_test_crop": real_crop,
            "real_test_presegmented_plate": prepared,
            "real_external_annotated_box_to_text": end_to_end,
            "model_size": size, "cpu_forward": bench(model),
            "checkpoint": str(args.output / f"{name}.pt"),
            "checkpoint_sha256": sha(args.output / f"{name}.pt"),
        }
        print(json.dumps({"event": "evaluate", "model": name,
                          "prepared_plate_accuracy": prepared["total"]["plate_exact_accuracy"],
                          "external_plate_accuracy": end_to_end["total"]["plate_exact_accuracy"],
                          "external_segmentation_success": end_to_end["total"]["segmentation_success"]}), flush=True)
        for domain, confusion in (("real_presegmented", prepared["confusion_matrix"]),
                                  ("real_external", end_to_end["confusion_matrix_aligned_only"])):
            with (args.output / f"{name}_{domain}_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["true/pred", *CLASS_NAMES])
                for char, row in zip(CLASS_NAMES, confusion):
                    writer.writerow([char, *row])
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "metrics": str(args.output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
