"""Matched LeNet-5 dense vs 50% Conv2-filter pruning on frontal rendered cars.

Train/validation/test identities are disjoint. With --synthetic-only, no real
photos are read. Optional real-photo evaluation uses held-out train(1) source
identities and annotated plate boxes without the ground-truth character count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

from prepare_two_row_frontal_dense import identity_split
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, compact_teacher_to_student
from train_independent_structured import parse_source_name, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES, CLASS_TO_INDEX, LeNet5
from train_matched_channel_pruning import model_macs


DATA = Path("data/car_frontal_synthetic_2x1000_v1")
REAL = Path("data/OCR/OCR/images/train(1)/detection")
OUT = Path("artifacts/synthetic_car_dense_pruned_20261001")
PATTERN = re.compile(r"^[0-9]{2}[ABCDEFGHKLMNPSTUVXYZ][0-9]{5}$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_segment(gray: np.ndarray, layout: str) -> list[np.ndarray] | None:
    """Find glyph components without consulting the label or expected count."""
    height, width = gray.shape
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((gray < 115).astype(np.uint8), 8)
    boxes = []
    min_height = 21 if layout == "two_row_car" else 25
    for x, y, w, h, area in stats[1:count]:
        if x < 6 or y < 7 or x + w > width - 6 or y + h > height - 7:
            continue
        if h < min_height or area < 35:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    if layout == "two_row_car":
        upper = sorted((box for box in boxes if box[1] + box[3] / 2 < height / 2), key=lambda box: box[0])
        lower = sorted((box for box in boxes if box[1] + box[3] / 2 >= height / 2), key=lambda box: box[0])
        ordered = upper + lower
        if len(upper) != 3 or len(lower) != 5:
            return None
    else:
        ordered = sorted(boxes, key=lambda box: box[0])
        if len(ordered) != 8:
            return None
    crops = []
    for x, y, w, h in ordered:
        patch = gray[max(0, y - 3):min(height, y + h + 3), max(0, x - 3):min(width, x + w + 3)]
        crops.append(np.asarray(ImageOps.pad(Image.fromarray(patch), (32, 32), color=255), dtype=np.uint8))
    return crops


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    for layout in ("one_row_car", "two_row_car"):
        group = [row for row in rows if row["layout"] == layout]
        group.sort(key=lambda row: hashlib.sha256(f"{seed}|{row['label']}".encode()).hexdigest())
        if len(group) != 1000:
            raise ValueError(f"Expected 1000 {layout} plates, found {len(group)}")
        result["train"].extend(group[:800])
        result["val"].extend(group[800:900])
        result["test"].extend(group[900:])
    return result


def load_synthetic(data: Path, seed: int):
    manifest = list(csv.DictReader((data / "manifest.csv").open(encoding="utf-8")))
    if len(manifest) != 2000 or len({row["label"] for row in manifest}) != 2000:
        raise ValueError("Synthetic manifest must contain 2000 distinct labels")
    groups = split_rows(manifest, seed)
    loaded = {}
    failures = Counter()
    failure_examples = []
    for split, records in groups.items():
        plate_rows = []
        for row in records:
            gray = cv2.imread(str(data / row["file"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failures[(split, "unreadable")] += 1
                continue
            crops = synthetic_segment(gray, row["layout"])
            if crops is None:
                failures[(split, "segmentation")] += 1
                if len(failure_examples) < 20:
                    failure_examples.append(row["file"])
                continue
            plate_rows.append({"record": row, "crops": crops})
        loaded[split] = plate_rows
    if failures:
        raise RuntimeError(f"Generated-image segmentation failures: {dict(failures)}; examples={failure_examples}")
    return loaded, groups


def as_tensors(plates: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    images = np.stack([crop for plate in plates for crop in plate["crops"]]).astype(np.float32)
    labels = np.asarray([CLASS_TO_INDEX[char] for plate in plates for char in plate["record"]["label"]], dtype=np.int64)
    x = torch.from_numpy(images / 127.5 - 1.0).unsqueeze(1)
    y = torch.from_numpy(labels)
    return x, y


def epoch_batches(x: torch.Tensor, y: torch.Tensor, epoch: int, seed: int, batch: int):
    generator = torch.Generator().manual_seed(seed + epoch * 101)
    indices = torch.randperm(len(y), generator=generator)
    result = []
    for part in indices.split(batch):
        images = x[part].clone()
        # The same augmentation tensor is seen by the matched branches.
        angles = (torch.rand(len(part), generator=generator) * 8 - 4) * math.pi / 180
        theta = torch.zeros(len(part), 2, 3)
        theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angles), -torch.sin(angles)
        theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angles), torch.cos(angles)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        images = F.grid_sample(images, grid, padding_mode="border", align_corners=False)
        brightness = (torch.rand(len(part), 1, 1, 1, generator=generator) - 0.5) * 0.24
        images = (images + brightness + torch.randn(images.shape, generator=generator) * 0.055).clamp(-1, 1)
        result.append((images, y[part]))
    return result


def train_pass(model: nn.Module, batches, optimizer: torch.optim.Optimizer, loss_fn) -> dict:
    model.train()
    n = correct = 0
    loss_total = 0.0
    start = time.perf_counter()
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n += len(y)
        correct += int((logits.argmax(1) == y).sum())
        loss_total += loss.detach().item() * len(y)
    return {"loss": loss_total / n, "accuracy": correct / n, "seconds": time.perf_counter() - start}


def evaluate_crops(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 512):
    model.eval()
    predictions = []
    confidences = []
    n = correct = 0
    loss_total = 0.0
    start = time.perf_counter()
    with torch.inference_mode():
        for start_index in range(0, len(y), batch):
            xb, yb = x[start_index:start_index + batch], y[start_index:start_index + batch]
            logits = model(xb)
            loss_total += float(F.cross_entropy(logits, yb, reduction="sum"))
            confidence, pred = logits.softmax(dim=1).max(dim=1)
            predictions.extend(pred.tolist())
            confidences.extend(confidence.tolist())
            n += len(yb)
            correct += int((pred == yb).sum())
    return {"loss": loss_total / n, "accuracy": correct / n,
            "seconds": time.perf_counter() - start}, predictions, confidences


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, character in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (character != other)))
        previous = current
    return previous[-1]


def plate_metrics(model: nn.Module, plates: list[dict], synthetic: bool, limit: int = 0,
                  data_root: Path = DATA) -> dict:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    by_layout = defaultdict(lambda: {"plates": 0, "exact": 0, "correct_chars_aligned": 0, "aligned_chars": 0,
                                    "segmentation_success": 0, "edit_errors": 0, "chars": 0})
    failure = Counter()
    times = []
    confidences = []
    for plate in plates[:limit or None]:
        start = time.perf_counter()
        if synthetic:
            row = plate["record"]
            gray = cv2.imread(str(data_root / row["file"]), cv2.IMREAD_GRAYSCALE)
            crops = synthetic_segment(gray, row["layout"]) if gray is not None else None
            row_count = 2 if row["layout"] == "two_row_car" else 1
            layout = row["layout"]
            truth = row["label"]
        else:
            row = plate
            photo = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
            gray = None
            if photo is not None:
                x1, y1, x2, y2 = row["box"]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(photo.shape[1], x2), min(photo.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    gray, _applied, _angle = perspective_correct(photo[y1:y2, x1:x2])
            result = segment_characters(gray, expected_count=None, enhancement="clahe_sharp") if gray is not None else None
            crops, row_count, _count = result if result is not None else (None, 0, 0)
            layout = row["layout"]
            truth = row["text"]
        stats = by_layout[layout]
        stats["plates"] += 1
        stats["chars"] += len(truth)
        expected_rows = 2 if layout == "two_row_car" else 1
        if crops is None or row_count != expected_rows:
            failure["segmentation_or_row"] += 1
            pred_text = ""
        else:
            stats["segmentation_success"] += 1
            if len(crops) != len(truth):
                failure["character_count"] += 1
            inputs = torch.from_numpy(np.stack(crops).astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
            model.eval()
            with torch.inference_mode():
                prob = model(inputs).softmax(dim=1)
            pred = prob.argmax(dim=1).tolist()
            pred_text = "".join(CLASS_NAMES[index] for index in pred)
            confidences.extend(prob.max(dim=1).values.tolist())
            if len(pred_text) == len(truth):
                stats["aligned_chars"] += len(truth)
                stats["correct_chars_aligned"] += sum(a == b for a, b in zip(truth, pred_text))
                for a, b in zip(truth, pred_text):
                    matrix[CLASS_TO_INDEX[a], CLASS_TO_INDEX[b]] += 1
        stats["exact"] += int(pred_text == truth)
        stats["edit_errors"] += edit_distance(truth, pred_text)
        times.append((time.perf_counter() - start) * 1000)
    total = {key: sum(group[key] for group in by_layout.values()) for key in next(iter(by_layout.values()))}
    total["character_accuracy_on_aligned"] = total["correct_chars_aligned"] / max(1, total["aligned_chars"])
    total["plate_exact_accuracy"] = total["exact"] / max(1, total["plates"])
    total["character_error_rate_edit_distance"] = total["edit_errors"] / max(1, total["chars"])
    for group in by_layout.values():
        group["character_accuracy_on_aligned"] = group["correct_chars_aligned"] / max(1, group["aligned_chars"])
        group["plate_exact_accuracy"] = group["exact"] / max(1, group["plates"])
        group["character_error_rate_edit_distance"] = group["edit_errors"] / max(1, group["chars"])
    return {"total": total, "by_layout": dict(by_layout), "failures": dict(failure),
            "end_to_end_from_plate_or_annotated_box_ms": {
                "median": statistics.median(times), "p95": sorted(times)[int(0.95 * (len(times) - 1))],
                "samples": len(times)},
            "prediction_confidence_median": statistics.median(confidences) if confidences else None,
            "confusion_matrix_aligned_only": matrix.tolist()}


def real_holdout(root: Path, layout: str, count: int, exclude: set[str], seed: int) -> list[dict]:
    folder = "one_row" if layout == "one_row_car" else "two_rows"
    by_text = {}
    for path in (root / folder).glob("*.jpg"):
        parsed = parse_source_name(path)
        if parsed is None:
            continue
        text, x1, y1, x2, y2, _type = parsed
        if not PATTERN.fullmatch(text) or identity_split(text) != "test" or text in exclude:
            continue
        by_text.setdefault(text, {"path": path, "text": text, "box": (x1, y1, x2, y2), "layout": layout})
    ranked = sorted(by_text.values(), key=lambda row: hashlib.sha256(f"{seed}|{layout}|{row['text']}".encode()).hexdigest())
    if len(ranked) < count:
        raise RuntimeError(f"Only {len(ranked)} held-out {layout} real identities")
    return ranked[:count]


def model_size(model: nn.Module, channels: tuple[int, int, int, int]) -> dict:
    weights = sum(module.weight.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    biases = sum(module.bias.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    layers = {name: {"weight_shape": list(module.weight.shape), "weights": module.weight.numel(),
                     "biases": module.bias.numel(), "macs": macs}
              for (name, module), macs in zip(
                  ((name, module) for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))),
                  (117600, 15000 * channels[1], 3000 * channels[1], 10080, 2520))}
    return {"weights": weights, "biases": biases, "parameters": weights + biases,
            "fp32_bytes": 4 * (weights + biases), "int8_weights_int32_bias_bytes": weights + 4 * biases,
            "macs_per_character": model_macs(channels, len(CLASS_NAMES)),
            "macs_per_8_character_plate": 8 * model_macs(channels, len(CLASS_NAMES)),
            "layers": layers}


def bench(model: nn.Module, repeats: int = 300) -> dict:
    model.eval()
    output = {}
    with torch.inference_mode():
        for batch in (1, 8):
            x = torch.randn(batch, 1, 32, 32)
            for _ in range(50):
                model(x)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                model(x)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            output[str(batch)] = {"median_ms": statistics.median(samples),
                                  "p95_ms": sorted(samples)[int(0.95 * (repeats - 1))],
                                  "repeats": repeats}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--real", type=Path, default=REAL)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--prune-after", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-test-per-layout", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Train, validate and test only on the 2,000 rendered plates; do not read real photos.")
    args = parser.parse_args()
    if not 1 <= args.prune_after < args.epochs:
        raise ValueError("prune-after must be between 1 and epochs-1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    plates, source_rows = load_synthetic(args.data, args.seed)
    print(json.dumps({"event": "audit", "train_plates": len(plates["train"]),
                      "val_plates": len(plates["val"]), "test_plates": len(plates["test"]),
                      "characters_per_plate": 8}), flush=True)
    if args.audit_only:
        return
    for path in (args.output / "dense.pt", args.output / "pruned.pt", args.output / "metrics.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    x_train, y_train = as_tensors(plates["train"])
    x_val, y_val = as_tensors(plates["val"])
    x_test, y_test = as_tensors(plates["test"])
    dense = LeNet5(len(CLASS_NAMES))
    student = None
    opt_dense = torch.optim.AdamW(dense.parameters(), lr=7e-4, weight_decay=1e-5)
    opt_student = None
    loss_fn = nn.CrossEntropyLoss()
    best = {"dense": (-1.0, float("inf"), 0), "pruned": (-1.0, float("inf"), 0)}
    history = []
    selected_channels = None
    wall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        batches = epoch_batches(x_train, y_train, epoch, args.seed, args.batch_size)
        for name, model, optimizer in (("dense", dense, opt_dense), ("pruned", student, opt_student)):
            if model is None or optimizer is None:
                continue
            train = train_pass(model, batches, optimizer, loss_fn)
            val, _pred, _confidence = evaluate_crops(model, x_val, y_val)
            record = {"epoch": epoch, "model": name, "train_loss": train["loss"],
                      "train_char_accuracy": train["accuracy"], "val_loss": val["loss"],
                      "val_char_accuracy": val["accuracy"], "train_seconds": train["seconds"],
                      "val_seconds": val["seconds"]}
            history.append(record)
            print(json.dumps(record), flush=True)
            if (val["accuracy"], -val["loss"]) > (best[name][0], -best[name][1]):
                best[name] = (val["accuracy"], val["loss"], epoch)
                torch.save({"model": model.state_dict(), "classes": CLASS_NAMES,
                            "channels": [6, 16 if name == "dense" else 8, 120, 84],
                            "epoch": epoch, "seed": args.seed, "data_manifest_sha256": sha(args.data / "manifest.csv")},
                           args.output / f"{name}.pt")
        if epoch == args.prune_after:
            student = LeNet5Conv2Pruned(len(CLASS_NAMES))
            selected_channels = compact_teacher_to_student(dense, student)
            student.features[0].weight.requires_grad_(True)
            student.features[0].bias.requires_grad_(True)
            # Both branches start a fresh continuation optimizer after shared pretraining.
            opt_dense = torch.optim.AdamW(dense.parameters(), lr=7e-4, weight_decay=1e-5)
            opt_student = torch.optim.AdamW(student.parameters(), lr=7e-4, weight_decay=1e-5)
            print(json.dumps({"event": "structured_prune", "epoch": epoch,
                              "conv2_out_channels": "16->8", "selected_indices": selected_channels}), flush=True)
    models = {"dense": LeNet5(len(CLASS_NAMES)), "pruned": LeNet5Conv2Pruned(len(CLASS_NAMES))}
    for name, model in models.items():
        ckpt = torch.load(args.output / f"{name}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    real = []
    if not args.synthetic_only:
        train_labels = {row["label"] for row in source_rows["train"]}
        val_labels = {row["label"] for row in source_rows["val"]}
        test_labels = {row["label"] for row in source_rows["test"]}
        synthetic_labels = train_labels | val_labels | test_labels
        real_one = real_holdout(args.real, "one_row_car", args.real_test_per_layout, synthetic_labels, args.seed)
        real_two = real_holdout(
            args.real, "two_row_car", args.real_test_per_layout,
            synthetic_labels | {row["text"] for row in real_one}, args.seed,
        )
        real = real_one + real_two
        if len({row["text"] for row in real}) != len(real):
            raise RuntimeError("Real test plate identities overlap between layouts")
    result = {"run": {"seed": args.seed, "epochs": args.epochs, "prune_after": args.prune_after,
                      "batch_size": args.batch_size, "torch_version": torch.__version__,
                      "cpu_threads": args.threads, "elapsed_seconds": time.perf_counter() - wall_start,
                      "peak_process_rss_bytes": psutil.Process().memory_info().rss,
                      "data_manifest_sha256": sha(args.data / "manifest.csv"),
                      "split_plates": {key: len(value) for key, value in plates.items()},
                      "real_external_test_plates": len(real), "selected_conv2_indices": selected_channels,
                      "synthetic_only": args.synthetic_only,
                      "real_test_note": "Not run; no real photos read." if args.synthetic_only else
                                        "Source filename bounding box used; no automatic full-image detector."},
              "models": {}}
    for name, model in models.items():
        crop, _pred, _confidence = evaluate_crops(model, x_test, y_test)
        synthetic = plate_metrics(model, plates["test"], synthetic=True, data_root=args.data)
        external = None if args.synthetic_only else plate_metrics(model, real, synthetic=False)
        channels = (6, 16 if name == "dense" else 8, 120, 84)
        result["models"][name] = {
            "best_val_character_accuracy": best[name][0], "best_epoch": best[name][2],
            "synthetic_test_crop": crop,
            "synthetic_plate_to_text": synthetic,
            "real_annotated_box_to_text": external,
            "model_size": model_size(model, channels),
            "cpu_forward": bench(model),
            "checkpoint": str(args.output / f"{name}.pt"),
            "checkpoint_sha256": sha(args.output / f"{name}.pt"),
        }
        print(json.dumps({"event": "evaluate", "model": name,
                          "synthetic_plate_accuracy": synthetic["total"]["plate_exact_accuracy"],
                          "real_plate_accuracy": None if external is None else external["total"]["plate_exact_accuracy"],
                          "real_segmentation_success": None if external is None else external["total"]["segmentation_success"]}), flush=True)
        domains = (("synthetic", synthetic),) if external is None else (("synthetic", synthetic), ("real", external))
        for domain, metrics in domains:
            with (args.output / f"{name}_{domain}_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["true/pred", *CLASS_NAMES])
                for char, line in zip(CLASS_NAMES, metrics["confusion_matrix_aligned_only"]):
                    writer.writerow([char, *line])
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "metrics": str(args.output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
