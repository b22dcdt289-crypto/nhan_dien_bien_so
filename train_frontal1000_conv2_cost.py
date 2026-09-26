from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, count_parameters, load_checkpoint
from train_independent_structured import perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES
from train_one_row_1000 import evaluate_split, plate_format_indices


MACS_PER_CHAR = 274_200


class ManifestCharacters(Dataset):
    def __init__(self, root: Path, records: list[dict], augment: bool = False):
        self.root = root
        self.augment = augment
        self.items = []
        self.events = Counter()
        char_to_idx = {char: index for index, char in enumerate(CLASS_NAMES)}
        for record in records:
            if not record.get("ground_truth_segmentation_ok"):
                continue
            if len(record["crop_files"]) != len(record["text"]):
                raise ValueError(f"Bad character crops in {record['sample_id']}")
            self.items.extend((path, char_to_idx[char]) for path, char in zip(record["crop_files"], record["text"]))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _augment_geometry(gray: np.ndarray, category: str) -> np.ndarray:
        h, w = gray.shape
        center = (w / 2.0, h / 2.0)
        if category == "rotation":
            matrix = cv2.getRotationMatrix2D(center, random.uniform(-3.5, 3.5), 1.0)
            return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if category == "affine":
            angle = random.uniform(-1.5, 1.5)
            scale = random.uniform(0.92, 1.08)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            matrix[0, 1] += random.uniform(-0.035, 0.035)
            matrix[0, 2] += random.uniform(-1.0, 1.0)
            matrix[1, 2] += random.uniform(-1.0, 1.0)
            return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if category == "perspective":
            src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            jitter = 0.9
            dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return gray

    @staticmethod
    def _augment_photo(gray: np.ndarray, category: str) -> np.ndarray:
        if category == "blur":
            sigma = random.uniform(0.25, 0.8)
            return cv2.GaussianBlur(gray, (3, 3), sigma)
        if category == "noise":
            noise = np.random.normal(0.0, random.uniform(2.0, 8.0), gray.shape)
            return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if category == "brightness_contrast":
            alpha = random.uniform(0.82, 1.18)
            beta = random.uniform(-16.0, 16.0)
            return np.clip(alpha * gray.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return gray

    def __getitem__(self, index):
        relative, target = self.items[index]
        with Image.open(self.root / relative) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        if gray.shape != (32, 32):
            gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        if self.augment:
            r = random.random()
            geometry = "rotation" if r < 0.25 else "affine" if r < 0.37 else "perspective" if r < 0.45 else "none"
            r = random.random()
            photo = "blur" if r < 0.10 else "noise" if r < 0.32 else "brightness_contrast" if r < 0.55 else "none"
            gray = self._augment_geometry(gray, geometry)
            gray = self._augment_photo(gray, photo)
            self.events[f"geometry:{geometry}"] += 1
            self.events[f"photometric:{photo}"] += 1
        tensor = torch.from_numpy((gray.astype(np.float32) / 127.5 - 1.0)).unsqueeze(0)
        return tensor, target


def load_prior_cost(path: Path) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    confusion = np.asarray(report["models"]["conv2_pruned_student"]["validation"]["confusion_matrix"], dtype=np.float64)
    if confusion.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"Expected a {len(CLASS_NAMES)}x{len(CLASS_NAMES)} previous-model confusion matrix")
    costs = np.zeros_like(confusion, dtype=np.float32)
    alpha = 1.0
    for true_idx in range(len(CLASS_NAMES)):
        off_diag = confusion[true_idx].copy()
        off_diag[true_idx] = 0
        denominator = float(off_diag.sum() + alpha * (len(CLASS_NAMES) - 1))
        for predicted_idx in range(len(CLASS_NAMES)):
            if predicted_idx != true_idx:
                costs[true_idx, predicted_idx] = 1.0 + 2.0 * (off_diag[predicted_idx] + alpha) / denominator
    return costs, report


def weighted_ohem_loss(logits, targets, class_weights, confusion_cost, penalty_weight: float, hard_fraction: float):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    probabilities = torch.softmax(logits, dim=1)
    expected_penalty = (probabilities * confusion_cost[targets]).sum(dim=1)
    per_sample = ce + penalty_weight * expected_penalty
    k = max(1, int(math.ceil(len(per_sample) * hard_fraction)))
    hardest = torch.topk(per_sample.detach(), k=k, largest=True).indices
    # All samples contribute; the hardest quartile receives one additional equal-weight term.
    loss = 0.5 * per_sample.mean() + 0.5 * per_sample[hardest].mean()
    return loss, per_sample.detach(), k


def class_counts(dataset: ManifestCharacters) -> dict:
    counts = Counter(CLASS_NAMES[target] for _, target in dataset.items)
    values = {char: int(counts[char]) for char in CLASS_NAMES}
    nonzero = [value for value in values.values() if value]
    return {
        "counts": values,
        "total_characters": sum(values.values()),
        "classes_with_zero_support": [char for char, count in values.items() if count == 0],
        "classes_with_support": len(nonzero),
        "max_class_count": max(nonzero, default=0),
        "min_nonzero_class_count": min(nonzero, default=0),
        "class_balance_ratio_max_over_min_nonzero": max(nonzero) / min(nonzero) if nonzero else None,
    }


def weight_distributions(model: nn.Module) -> dict:
    report = {}
    for name, parameter in model.named_parameters():
        values = parameter.detach().float().cpu().numpy().reshape(-1)
        report[name] = {
            "elements": int(values.size),
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "mean_absolute": float(np.abs(values).mean()),
            "zero_fraction": float(np.mean(values == 0)),
            "quantiles_01_50_99": [float(x) for x in np.quantile(values, [0.01, 0.5, 0.99])],
        }
    return report


def activation_distributions(model: nn.Module, dataset: ManifestCharacters, device: torch.device, limit: int = 256) -> dict:
    names = ("features.0", "features.1", "features.2", "features.3", "features.4", "features.5", "classifier.0", "classifier.1", "classifier.2", "classifier.3", "classifier.4")
    tensors = []
    for path, _ in dataset.items[:limit]:
        with Image.open(dataset.root / path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
        if pixels.shape != (32, 32):
            pixels = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA)
        tensors.append(torch.from_numpy((pixels / 127.5 - 1.0)).unsqueeze(0))
    if not tensors:
        return {"sample_glyphs": 0, "layers": {}}
    batch = torch.stack(tensors).to(device)
    captured = {}
    handles = []
    modules = dict(model.named_modules())
    for name in names:
        if name in modules:
            handles.append(modules[name].register_forward_hook(
                lambda _module, _inputs, output, layer=name: captured.__setitem__(layer, output.detach().float().cpu().numpy())
            ))
    model.eval()
    with torch.inference_mode():
        model(batch)
    for handle in handles:
        handle.remove()
    report = {}
    for name, array in captured.items():
        values = array.reshape(-1)
        report[name] = {
            "shape_per_glyph": list(array.shape[1:]),
            "elements_per_glyph": int(array[0].size),
            "sampled_values": int(values.size),
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "zero_fraction": float(np.mean(values == 0)),
            "quantiles_01_50_99": [float(x) for x in np.quantile(values, [0.01, 0.5, 0.99])],
        }
    return {"sample_glyphs": len(tensors), "source_split": "validation character crops", "layers": report}


def confusion_from_rows(rows: list[dict], field: str) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for row in rows:
        expected, predicted = row["expected"], row[field]
        for truth, guess in zip(expected, predicted):
            matrix[CLASS_NAMES.index(truth), CLASS_NAMES.index(guess)] += 1
    return matrix


def runtime_aligned_char_metrics(model, root: Path, records: list[dict], device: torch.device) -> tuple[dict, np.ndarray]:
    """Character metrics for real predicted segments only when detected count matches the label length."""
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    aligned_plates = aligned_chars = correct_chars = exact_plates = 0
    confidences, correct_conf, wrong_conf = [], [], []
    detected_count_matches = 0
    model.eval()
    with torch.inference_mode():
        for record in records:
            image = cv2.imread(str(root / record["plate_crop"]))
            if image is None:
                continue
            gray, _, _ = perspective_correct(image)
            segments = segment_characters(gray, expected_count=None, enhancement=record["enhancement"])
            if segments is None:
                continue
            crops, row_count, detected_count = segments
            if row_count != 1:
                continue
            detected_count_matches += int(detected_count == len(record["text"]))
            if detected_count != len(record["text"]):
                continue
            tensors = [torch.from_numpy((crop.astype(np.float32) / 127.5 - 1.0)).unsqueeze(0) for crop in crops]
            logits = model(torch.stack(tensors).to(device))
            probabilities = torch.softmax(logits, dim=1)
            conf = probabilities.max(1).values.cpu().numpy()
            predicted = plate_format_indices(logits, len(record["text"]))
            predicted_text = "".join(CLASS_NAMES[index] for index in predicted.cpu().tolist())
            aligned_plates += 1
            exact_plates += int(predicted_text == record["text"])
            aligned_chars += len(record["text"])
            for position, (truth, guess) in enumerate(zip(record["text"], predicted_text)):
                ti, gi = CLASS_NAMES.index(truth), CLASS_NAMES.index(guess)
                matrix[ti, gi] += 1
                correct = truth == guess
                correct_chars += int(correct)
                confidences.append(float(conf[position]))
                (correct_conf if correct else wrong_conf).append(float(conf[position]))
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.000001)]
    confidence = {
        "glyphs": len(confidences), "mean": float(np.mean(confidences)) if confidences else None,
        "median": float(np.median(confidences)) if confidences else None,
        "p10": float(np.quantile(confidences, .10)) if confidences else None,
        "p90": float(np.quantile(confidences, .90)) if confidences else None,
        "mean_on_correct": float(np.mean(correct_conf)) if correct_conf else None,
        "mean_on_wrong": float(np.mean(wrong_conf)) if wrong_conf else None,
        "correct_count": len(correct_conf), "wrong_count": len(wrong_conf),
        "histogram": {f"{a:.0%}-{min(b,1.0):.0%}": int(sum(a <= x < b for x in confidences)) for a,b in bins},
        "condition": "predicted single-row segmentation count must equal labeled character count; confidence is uncalibrated",
    }
    return ({
        "plates_with_single_row_and_correct_detected_count": aligned_plates,
        "plates_with_single_row_detected_count_equal_label": detected_count_matches,
        "character_count": aligned_chars, "correct_characters": correct_chars,
        "character_accuracy_conditional_on_runtime_count_match": correct_chars / aligned_chars if aligned_chars else None,
        "exact_plates_conditional_on_runtime_count_match": exact_plates,
        "plate_exact_accuracy_conditional_on_runtime_count_match": exact_plates / aligned_plates if aligned_plates else None,
        "coverage_all_test_plates": aligned_plates / max(1, len(records)),
        "confusion_matrix": matrix.tolist(), "confidence": confidence,
    }, matrix)


def save_confusion(path: Path, matrix: np.ndarray):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for char, row in zip(CLASS_NAMES, matrix):
            writer.writerow([char, *map(int, row)])


def confidence_report(model, dataset: ManifestCharacters, device: torch.device, batch_size: int = 256) -> dict:
    if not dataset.items:
        return {"glyphs": 0, "note": "No aligned ground-truth character crops in this split."}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    confidences, correct_conf, wrong_conf = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            values, preds = probs.max(dim=1)
            values = values.cpu().numpy(); matches = (preds.cpu() == targets).numpy()
            confidences.extend(values.tolist())
            correct_conf.extend(values[matches].tolist())
            wrong_conf.extend(values[~matches].tolist())
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.000001)]
    return {
        "glyphs": len(confidences),
        "mean": float(np.mean(confidences)), "median": float(np.median(confidences)),
        "p10": float(np.quantile(confidences, 0.10)), "p90": float(np.quantile(confidences, 0.90)),
        "mean_on_correct": float(np.mean(correct_conf)) if correct_conf else None,
        "mean_on_wrong": float(np.mean(wrong_conf)) if wrong_conf else None,
        "correct_count": len(correct_conf), "wrong_count": len(wrong_conf),
        "histogram": {
            f"{low:.0%}-{min(high, 1.0):.0%}": int(sum(low <= value < high for value in confidences))
            for low, high in bins
        },
        "source_split": "test aligned character crops only; reports empty if segmentation did not yield labelled glyphs",
    }


def benchmark(old: nn.Module, new: nn.Module, device: torch.device, repeats: int = 500, warmup: int = 100) -> dict:
    torch.set_num_threads(1)
    old_cpu, new_cpu = old.to("cpu").eval(), new.to("cpu").eval()
    summaries = {}
    for batch in (1, 8):
        x = torch.randn(batch, 1, 32, 32)
        with torch.inference_mode():
            for model in (old_cpu, new_cpu):
                for _ in range(warmup):
                    model(x)
            samples = {"previous_conv2_student": [], "frontal1000_finetuned": []}
            names = list(samples)
            for i in range(repeats):
                random.shuffle(names)
                for name in names:
                    model = old_cpu if name == "previous_conv2_student" else new_cpu
                    start = time.perf_counter_ns(); model(x); elapsed = time.perf_counter_ns() - start
                    samples[name].append(elapsed / 1e6)
        summaries[str(batch)] = {}
        for name, values in samples.items():
            summaries[str(batch)][name] = {
                "median_ms_per_forward": float(statistics.median(values)),
                "p95_ms_per_forward": float(np.quantile(values, 0.95)),
                "glyphs_per_second_median_based": batch * 1000.0 / statistics.median(values),
            }
    old.to(device); new.to(device)
    return {
        "method": "Interleaved PyTorch CPU forward-only benchmark; 1 CPU thread; synthetic normalized N×1×32×32 glyph tensors",
        "warmup_per_model_batch": warmup, "repeats_per_model_batch": repeats,
        "torch_version": torch.__version__, "cpu": platform.processor(), "batches": summaries,
        "warning": "Software CPU latency; not FPGA latency/fMAX/cycle measurement.",
    }


def markdown_report(path: Path, metrics: dict):
    old = metrics["models"]["previous_conv2_student"]
    new = metrics["models"]["frontal1000_finetuned"]
    data = metrics["data"]
    train_classes = data["class_balance"]["train"]
    val_confidence = new["validation"].get("mean_max_softmax_confidence_per_character")
    penalty_values = np.asarray(metrics["confusion_cost"]["matrix"], dtype=np.float32)
    off_diagonal_costs = penalty_values[~np.eye(len(CLASS_NAMES), dtype=bool)]
    lines = [
        "# 1,000 near-frontal one-row plates — Conv2-only structured-pruned LeNet-5",
        "",
        "## Dataset and protocol",
        "",
        f"- Source: `{data['source']}`; train/validation/test plates: {data['plate_counts']}.",
        f"- Test set: {data['test_protocol']}",
        f"- Preparation audit: scanned {data['preparation_quality']['scanned_unique_images']:,} frames; {data['preparation_quality']['eligible_frames_after_quality_filters']:,} passed image gates ({data['preparation_quality']['unique_eligible_plate_identities']:,} unique labels, {data['preparation_quality']['unique_trainable_plate_identities']:,} segmentable 8-character identities).",
        "- ROI selection filters: width×height ≥120×28 px, aspect ratio 3.7–5.8, Laplacian blur ≥80, grayscale contrast std ≥25, roll angle ≤4°. Images are real crops enhanced with CLAHE + mild sharpening, not AI super-resolution.",
        f"- Train characters: {train_classes['total_characters']:,}; supported classes: {train_classes['classes_with_support']}/30; balance max/min(nonzero): {train_classes['class_balance_ratio_max_over_min_nonzero']:.3f}.",
        f"- Source-label QA: {data['visually_verified_training_label_corrections']} labels were corrected only after visually checking the plate crops; substitutions by position are in the JSON. Uncertain disagreements were not auto-corrected.",
        "- The test identities are absent from the previous model's manifest. Runtime character accuracy is reported only where predicted segmentation count aligns with the label; end-to-end plate exact counts skips and count mismatches as failures.",
        "",
        "## Accuracy on the same unseen test plates",
        "",
        "| Model | Runtime character accuracy (count aligned) | Full-plate exact (end-to-end; skips wrong) | Runtime segmentation coverage | Reference segmentation coverage | MAC/char | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("previous_conv2_student", "Previous Conv2 student"), ("frontal1000_finetuned", "Fine-tuned on 1,000")):
        s = metrics["models"][key]["test"]
        runtime_chars = metrics["models"][key]["runtime_character_metrics"]
        char_accuracy = runtime_chars["character_accuracy_conditional_on_runtime_count_match"]
        char_accuracy_text = f"{char_accuracy:.3%}" if char_accuracy is not None else "N/A"
        lines.append(
            f"| {label} | {char_accuracy_text} "
            f"({runtime_chars['character_count']:,} chars; "
            f"{runtime_chars['plates_with_single_row_and_correct_detected_count']}/{s['plates']} plates) | "
            f"{s['runtime_no_known_count']['exact_accuracy_all_including_skips']:.3%} "
            f"({s['runtime_no_known_count']['exact_plates_all_including_skips']}/{s['plates']}) | "
            f"{s['runtime_no_known_count']['coverage']:.1%} | "
            f"{s['ground_truth_segmentation_coverage']:.1%} | {MACS_PER_CHAR:,} | {metrics['parameter_counts_for_report'][key]:,} |"
        )
    lines += [
        "",
        "Character accuracy on raw new test plates is only defined for plates where the automatic segmenter returns a single row and exactly 8 glyphs; coverage is shown above. The 300-plate validation split below has reference character crops, but its identities occur in the previous model's manifest, so treat it as a tuning/diagnostic set, not an independent test.",
        f"- The new test segmenter produced exactly 8 aligned glyphs on {new['runtime_character_metrics']['plates_with_single_row_and_correct_detected_count']}/{new['test']['plates']} plates; therefore unseen-test character accuracy and confidence are **not measurable** here (N/A, not 0% classifier accuracy). Full-plate exact remains 0/500 because count mismatches/skips fail end-to-end.",
        f"- Test segmentation counts (previous/adapted share this weight-independent stage): `{json.dumps(new['test']['runtime_no_known_count']['detected_character_count_histogram'])}`; attempted {new['test']['runtime_no_known_count']['attempted']}/500, skipped {new['test']['runtime_no_known_count']['skipped']}/500. The dominant failure is outputting 7 instead of 8 glyphs.",
        f"- Validation mean max-softmax confidence: previous {old['validation']['mean_max_softmax_confidence_per_character']:.3%}, adapted {val_confidence:.3%}; not calibrated and validation identities were seen by the prior model.",
        f"- Focused confusion submatrix source: `{metrics['confusion_cost']['focused_test_submatrix_source']}`. Off-diagonal confusion-cost M range: {float(off_diagonal_costs.min()):.3f}–{float(off_diagonal_costs.max()):.3f}; full M is exported separately.",
        "",
        "| Model | Validation character accuracy (reference crops) | Validation exact plate (reference crops) | Reference segmentation coverage |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("previous_conv2_student", "Previous Conv2 student"), ("frontal1000_finetuned", "Fine-tuned on 1,000")):
        s = metrics["models"][key]["validation"]
        lines.append(f"| {label} | {s['character_accuracy_conditional_on_ground_truth_segmentation']:.3%} ({s['character_count']:,} chars) | {s['plate_exact_accuracy_all_test_plates']:.3%} ({s['exact_plates']}/{s['plates']}) | {s['ground_truth_segmentation_coverage']:.1%} |")
    lines += [
        "",
        "## Training, augmentation, and hard examples",
        "",
        f"- Cost-sensitive loss: weighted cross-entropy + 0.25 × expected confusion-cost penalty, with M derived only from the previous model's validation confusion matrix and Laplace smoothing.",
        f"- OHEM: top {metrics['training']['hard_fraction']:.0%} hardest examples add an equally weighted auxiliary mean; discarded samples: {metrics['training']['hard_samples_discarded_rate']:.1%}.",
        f"- Augmentation observed over {metrics['training']['sample_presentations']:,} sample presentations: `{json.dumps(metrics['training']['augmentation_observed_rates'], ensure_ascii=False)}` (fractions per training glyph presentation).",
        f"- Test confidence: N/A because no test plate produced an aligned 8-glyph sequence; see validation confidence above and raw distributions in JSON.",
        "",
        "## Compute and resource estimates",
        "",
        f"- LeNet-5 Conv2-only topology (Conv1=6 fixed, Conv2=8): **{MACS_PER_CHAR:,} MAC/character**, **{MACS_PER_CHAR*8:,} MAC/8-character plate**, **38,198 parameters**.",
        f"- FP32 raw weights: {metrics['resources']['fp32_parameter_bytes']:,} bytes; INT8 raw-weight estimate: {metrics['resources']['int8_parameter_bytes_estimate_only']:,} bytes (not an exported/validated quantized model).",
        f"- CPU software latency: see benchmark in JSON; the 8-character batch p50 is {metrics['latency']['batches']['8']['frontal1000_finetuned']['median_ms_per_forward']:.4f} ms (not FPGA timing).",
        f"- Interleaved CPU batch-8 p50: previous {metrics['latency']['batches']['8']['previous_conv2_student']['median_ms_per_forward']:.4f} ms → adapted {metrics['latency']['batches']['8']['frontal1000_finetuned']['median_ms_per_forward']:.4f} ms; measured delta {metrics['latency']['batches']['8']['frontal1000_finetuned']['median_ms_per_forward']/max(1e-12,metrics['latency']['batches']['8']['previous_conv2_student']['median_ms_per_forward'])-1:+.1%}. Same topology/MAC, so this is not a structural speedup.",
        f"- At the existing SDC target 50 MHz and an assumed single MAC/cycle: {MACS_PER_CHAR:,} cycles/character = {MACS_PER_CHAR/50_000_000*1000:.3f} ms; 8 characters = {MACS_PER_CHAR*8:,} cycles = {MACS_PER_CHAR*8/50_000_000*1000:.3f} ms. This is an operation-count estimate, not a measured design result.",
        "- Quartus synthesis/STA and AI RTL are not available in this environment; ALM/DSP/BRAM, achieved fMAX, pipeline cycles, and hardware latency are therefore **not measured**. Current DE10-Lite top is a heartbeat demo, not this classifier.",
        "",
        "## Interpretation",
        "",
        "Fine-tuning changes weights, not the structured topology: MACs and parameter count remain the same as the previous Conv2-pruned student. Any measured CPU latency delta is benchmark variation/weight-dependent runtime noise, not a structural speedup. The model itself classifies a pre-cropped glyph; the current classical segmentation stage is part of end-to-end coverage and is counted explicitly.",
        "",
        f"Full aggregated metrics: `{metrics['artifact_paths']['json']}`. Confusion submatrix: `artifacts/frontal1000_confusion_submatrix.csv`; full confusion-cost matrix M: `artifacts/frontal1000_confusion_cost_M.csv`; class/augmentation distributions: `artifacts/frontal1000_class_distribution.csv` and `artifacts/frontal1000_augmentation_distribution.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Conv2-only structured-pruned LeNet-5 on a curated 1,000-plate set.")
    parser.add_argument("--data", type=Path, default=Path("data/train1_frontal_1000_unseen_v3"))
    parser.add_argument("--initial", type=Path, default=Path("artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt"))
    parser.add_argument("--prior-metrics", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_distill_metrics.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_conv2_frontal1000_cost_v1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/frontal1000_conv2_cost_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/frontal1000_conv2_cost_report.md"))
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--confusion-penalty", type=float, default=0.25)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((args.data / "preparation_metrics.json").read_text(encoding="utf-8"))
    correction_path = args.data / "review" / "verified_label_corrections.json"
    verified_corrections = json.loads(correction_path.read_text(encoding="utf-8")) if correction_path.is_file() else []
    correction_map = {item["source_file"]: item["corrected_text"] for item in verified_corrections}
    applied_corrections = []
    for record in manifest:
        proposed = correction_map.get(Path(record["source"]).name)
        if record["split"] == "train" and proposed is not None:
            if len(proposed) != len(record["text"]) or any(char not in CLASS_NAMES for char in proposed):
                raise ValueError(f"Invalid visually checked label correction for {record['sample_id']}")
            before = record["text"]
            record["text"] = proposed
            record["label_correction_applied"] = True
            changed = [(index + 1, old, new) for index, (old, new) in enumerate(zip(before, proposed)) if old != new]
            applied_corrections.extend(changed)
    if len(applied_corrections) != len(verified_corrections):
        raise RuntimeError("Not all visually checked label corrections matched exactly one selected training record")
    by_split = {split: [r for r in manifest if r["split"] == split] for split in ("train", "val", "test")}
    if len(by_split["train"]) != 1000 or len(by_split["val"]) != 300 or len(by_split["test"]) != 500:
        raise RuntimeError(f"Unexpected plate counts: { {k: len(v) for k, v in by_split.items()} }")
    train_ids = {r["text"] for r in by_split["train"]}; val_ids = {r["text"] for r in by_split["val"]}; test_ids = {r["text"] for r in by_split["test"]}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Identity overlap across train/validation/test")
    if len(train_ids) != len(by_split["train"]) or len(val_ids) != len(by_split["val"]) or len(test_ids) != len(by_split["test"]):
        raise RuntimeError("Duplicate plate identities appeared after applying verified label corrections")
    if not prep.get("test_identities_absent_from_prior_model_dataset"):
        raise RuntimeError("The selected test identities overlap the prior model dataset")

    train_set = ManifestCharacters(args.data, by_split["train"], augment=True)
    val_chars = ManifestCharacters(args.data, by_split["val"], augment=False)
    test_chars = ManifestCharacters(args.data, by_split["test"], augment=False)
    if len(train_set) == 0 or len(val_chars) == 0:
        raise RuntimeError("No aligned training/validation glyph crops")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_chars, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
    prior_checkpoint = load_checkpoint(model, args.initial, device)
    model._report_name = "frontal1000_finetuned"
    conv1 = model.features[0]
    conv1.weight.requires_grad_(False); conv1.bias.requires_grad_(False)
    initial_conv1 = {name: value.detach().clone() for name, value in conv1.state_dict().items()}
    cost_np, prior_report = load_prior_cost(args.prior_metrics)
    cost_matrix = torch.tensor(cost_np, dtype=torch.float32, device=device)
    counts = Counter(target for _, target in train_set.items)
    max_count = max(counts.values())
    raw_weights = torch.tensor([math.sqrt(max_count / max(1, counts.get(i, 0))) for i in range(len(CLASS_NAMES))], dtype=torch.float32)
    class_weights = (raw_weights / raw_weights.mean()).clamp(0.5, 3.0).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    prior_model = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
    load_checkpoint(prior_model, args.initial, device)
    prior_model._report_name = "previous_conv2_student"
    prior_model.eval()
    best_score = -1.0; best_char = -1.0; best_state = None; history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = total_correct = total_seen = hard_selected = 0
        started = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss, per_sample, k = weighted_ohem_loss(logits, labels, class_weights, cost_matrix, args.confusion_penalty, args.hard_fraction)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            total_seen += len(labels); hard_selected += k
        scheduler.step()
        model.eval(); val_loss = val_correct = val_seen = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
                val_correct += int((logits.argmax(1) == labels).sum())
                val_seen += len(labels)
        val_summary, _, _ = evaluate_split(model, device, args.data, manifest, "val")
        runtime_acc = val_summary["runtime_no_known_count"]["exact_accuracy_all_including_skips"]
        char_acc = val_summary["character_accuracy_conditional_on_ground_truth_segmentation"]
        score = runtime_acc + 1e-6 * char_acc
        row = {
            "epoch": epoch, "train_loss": total_loss / max(1, total_seen), "train_top1_accuracy": total_correct / max(1, total_seen),
            "val_loss": val_loss / max(1, val_seen), "val_top1_accuracy": val_correct / max(1, val_seen),
            "val_plate_runtime_exact_accuracy_all_including_skips": runtime_acc,
            "val_character_accuracy_on_aligned_gt_crops": char_acc,
            "hard_top_fraction_selected": hard_selected / max(1, total_seen),
            "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, best_char = score, char_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "model": best_state, "classes": CLASS_NAMES,
                "arch": "LeNet5Conv2Pruned", "channels": [6, 8, 120, 84],
                "epoch": epoch, "val_runtime_plate_exact_accuracy": runtime_acc,
                "val_character_accuracy": char_acc, "macs_per_character": MACS_PER_CHAR,
                "params": count_parameters(model), "train_plates": 1000,
                "source_dataset": str(args.data), "initialized_from": str(args.initial),
                "structured_pruning": "Conv1 fixed at 6 channels; Conv2 retained 8 channels",
            }, args.output)

    if best_state is None:
        raise RuntimeError("No best checkpoint selected")
    model.load_state_dict(best_state); model.eval()
    if any(not torch.equal(initial_conv1[name], value.detach()) for name, value in model.features[0].state_dict().items()):
        raise AssertionError("Conv1 changed although it was frozen")
    print("stage=test_evaluation", flush=True)
    prior_summary, prior_rows, prior_conf = evaluate_split(prior_model, device, args.data, manifest, "test")
    final_summary, final_rows, final_conf = evaluate_split(model, device, args.data, manifest, "test")
    prior_val_summary, _, prior_val_conf = evaluate_split(prior_model, device, args.data, manifest, "val")
    final_val_summary, _, final_val_conf = evaluate_split(model, device, args.data, manifest, "val")
    prior_runtime_chars, prior_runtime_conf = runtime_aligned_char_metrics(prior_model, args.data, by_split["test"], device)
    final_runtime_chars, final_runtime_conf = runtime_aligned_char_metrics(model, args.data, by_split["test"], device)
    for summary in (prior_summary, final_summary):
        summary["macs_per_character"] = MACS_PER_CHAR
        summary["total_character_macs"] = MACS_PER_CHAR * summary["character_count"]
        summary["average_macs_per_plate"] = MACS_PER_CHAR * summary["character_count_all_plate_labels"] / max(1, summary["plates"])
        if summary["character_count"] == 0:
            summary["character_accuracy_conditional_on_ground_truth_segmentation"] = None
            summary["raw_character_accuracy_conditional_on_ground_truth_segmentation"] = None
            summary["plate_exact_accuracy_conditional_on_ground_truth_segmentation"] = None
            summary["plate_exact_accuracy_all_test_plates"] = None
            summary["raw_plate_exact_accuracy_all_test_plates"] = None
            summary["mean_max_softmax_confidence_per_character"] = None
            summary["character_metric_status"] = "not measurable: no ground-truth aligned character crops in this split"
    raw_prior = confusion_from_rows(prior_rows, "raw_ocr_with_ground_truth_segmentation")
    raw_final = confusion_from_rows(final_rows, "raw_ocr_with_ground_truth_segmentation")
    out_dir = args.metrics.parent
    save_confusion(out_dir / "frontal1000_test_previous_confusion.csv", prior_conf)
    save_confusion(out_dir / "frontal1000_test_finetuned_confusion.csv", final_conf)
    save_confusion(out_dir / "frontal1000_test_previous_raw_confusion.csv", raw_prior)
    save_confusion(out_dir / "frontal1000_test_finetuned_raw_confusion.csv", raw_final)
    save_confusion(out_dir / "frontal1000_test_previous_runtime_aligned_confusion.csv", prior_runtime_conf)
    save_confusion(out_dir / "frontal1000_test_finetuned_runtime_aligned_confusion.csv", final_runtime_conf)
    save_confusion(out_dir / "frontal1000_validation_previous_confusion.csv", prior_val_conf)
    save_confusion(out_dir / "frontal1000_validation_finetuned_confusion.csv", final_val_conf)
    with (out_dir / "frontal1000_confusion_cost_M.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["true\\pred", *CLASS_NAMES])
        for char, row in zip(CLASS_NAMES, cost_np): writer.writerow([char, *[f"{x:.6f}" for x in row]])
    submatrix_source = final_runtime_conf if int(final_runtime_conf.sum()) else final_val_conf
    submatrix_source_name = "unseen_test_runtime_count_aligned" if int(final_runtime_conf.sum()) else "validation_reference_crops_fallback"
    pair_errors = []
    for i, true_char in enumerate(CLASS_NAMES):
        for j, pred_char in enumerate(CLASS_NAMES):
            if i != j and submatrix_source[i, j] > 0:
                pair_errors.append((int(submatrix_source[i, j]), true_char, pred_char))
    focused = sorted({c for _, a, b in sorted(pair_errors, reverse=True)[:4] for c in (a, b)}, key=CLASS_NAMES.index)
    if not focused:
        support = final_val_conf.sum(axis=1)
        focused = [CLASS_NAMES[index] for index in np.argsort(support)[::-1][:8] if support[index] > 0]
    focused_idx = [CLASS_NAMES.index(c) for c in focused]
    with (out_dir / "frontal1000_confusion_submatrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["true\\pred", *focused])
        for char, idx in zip(focused, focused_idx): writer.writerow([char, *[int(submatrix_source[idx, j]) for j in focused_idx]])

    final_confidence = final_runtime_chars["confidence"]
    previous_confidence = prior_runtime_chars["confidence"]
    latency = benchmark(prior_model, model, device)
    validation_activations = {
        "previous_conv2_student": activation_distributions(prior_model, val_chars, device),
        "frontal1000_finetuned": activation_distributions(model, val_chars, device),
    }
    previous_weights = weight_distributions(prior_model)
    final_weights = weight_distributions(model)
    max_acts = max(item["elements_per_glyph"] for item in validation_activations["frontal1000_finetuned"]["layers"].values())
    parameters = count_parameters(model)
    max_chars = max((len(r["text"]) for r in by_split["test"]), default=8)
    metrics = {
        "experiment": "Frontal 1,000-plate adaptation of prior Conv2-only structured-pruned 30-class LeNet-5; cost-sensitive confusion penalty and OHEM",
        "data": {
            "source": "data/OCR/OCR/images/train(1)/detection/one_row",
            "prepared_data": str(args.data), "plate_counts": {k: len(v) for k, v in by_split.items()},
            "test_protocol": "500 unseen, eight-character identities absent from previous model manifest; near-frontal image/ROI gates passed. Some do not pass current reference character segmentation and count as end-to-end failures.",
            "preparation_quality": {
                key: prep[key] for key in (
                    "scanned_unique_images", "eligible_frames_after_quality_filters",
                    "unique_eligible_plate_identities", "unique_trainable_plate_identities",
                    "eligible_identities_absent_from_prior_manifest", "rejected_frames_by_reason", "filters",
                )
            },
            "identity_disjoint_across_splits": True,
            "test_identities_absent_from_prior_model_dataset": prep.get("test_identities_absent_from_prior_model_dataset"),
            "test_gt_segmentable_plates": sum(bool(r.get("ground_truth_segmentation_ok")) for r in by_split["test"]),
            "visually_verified_training_label_corrections": len(verified_corrections),
            "verified_label_substitutions_by_position": [
                {"position_1based": position, "source_character": old, "corrected_character": new}
                for position, old, new in applied_corrections
            ],
            "class_balance": {"train": class_counts(train_set), "validation": class_counts(val_chars), "test_segmentable": class_counts(test_chars)},
            "augmentation_config": {
                "geometry_probabilities": {"rotation": 0.25, "affine_scale_shear_translation": 0.12, "perspective_corner_jitter": 0.08, "none": 0.55},
                "photometric_probabilities": {"blur": 0.10, "gaussian_noise": 0.22, "brightness_contrast": 0.23, "none": 0.45},
                "operations_are_mild_glyph_level_perturbations": True,
            },
        },
        "training": {
            "epochs_requested": args.epochs, "epochs_completed": len(history), "best_epoch": int(torch.load(args.output, map_location="cpu", weights_only=False)["epoch"]),
            "batch_size": args.batch_size, "learning_rate_initial": args.learning_rate,
            "optimizer": "AdamW", "loss": "weighted CE + lambda * expected confusion cost; 0.5 all-sample mean + 0.5 top-k hard mean",
            "confusion_penalty_lambda": args.confusion_penalty, "hard_fraction": args.hard_fraction,
            "hard_samples_discarded_rate": 0.0, "hard_fraction_extra_weighted": args.hard_fraction,
            "sample_presentations": sum(len(train_set) for _ in history),
            "augmentation_observed": dict(train_set.events), "augmentation_presentations_per_epoch": len(train_set),
            "augmentation_observed_rates": {
                key: count / max(1, len(train_set) * len(history)) for key, count in train_set.events.items()
            },
            "conv1_frozen": True, "conv1_bitwise_identical_to_initial": True,
            "confusion_matrix_teacher_source": "previous Conv2 model validation only; no test labels used",
            "class_weight_formula": "sqrt(max_class_count/class_count), normalized to mean 1, clipped [0.5,3.0]",
            "history": history,
        },
        "models": {
            "previous_conv2_student": {"checkpoint": str(args.initial), "validation": prior_val_summary, "test": prior_summary, "runtime_character_metrics": prior_runtime_chars, "confidence": previous_confidence,
                                        "weights": previous_weights, "validation_activation_distributions": validation_activations["previous_conv2_student"]},
            "frontal1000_finetuned": {"checkpoint": str(args.output), "validation": final_val_summary, "test": final_summary, "runtime_character_metrics": final_runtime_chars, "confidence": final_confidence,
                                      "weights": final_weights, "validation_activation_distributions": validation_activations["frontal1000_finetuned"]},
        },
        "confusion_cost": {
            "definition": "M[k,k]=0; M[k,j]=1+2*(C[k,j]+1)/(sum_{q!=k}C[k,q]+29), j!=k; C from previous model validation confusion matrix",
            "matrix": cost_np.tolist(), "source_validation_confusion": prior_report["models"]["conv2_pruned_student"]["validation"]["confusion_matrix"],
            "focused_test_submatrix_classes": focused,
            "focused_test_submatrix_source": submatrix_source_name,
            "focused_test_submatrix": submatrix_source[np.ix_(focused_idx, focused_idx)].tolist() if focused_idx else [],
            "most_common_test_confusions": [{"true": a, "predicted": b, "count": count} for count, a, b in sorted(pair_errors, reverse=True)[:20]],
        },
        "architecture": {"model": "LeNet-5 Conv2-only structured channel pruning", "channels": [6, 8, 120, 84], "conv1_fixed": True,
                         "parameters": parameters, "macs_per_character": MACS_PER_CHAR, "macs_per_plate_by_length": {str(n): MACS_PER_CHAR*n for n in sorted({len(r['text']) for r in by_split['test']})}},
        "resources": {
            "fp32_parameter_bytes": parameters * 4, "fp32_parameter_kib": parameters * 4 / 1024,
            "int8_parameter_bytes_estimate_only": parameters, "int8_parameter_kib_estimate_only": parameters / 1024,
            "max_single_layer_activation_elements_per_glyph": max_acts,
            "max_single_activation_tensor_fp32_bytes_per_glyph": max_acts * 4,
            "largest_map_layer": "features.2 (6x14x14) has 1,176 values; features.0 (6x28x28) has 4,704 values and is the peak at 18,816 B/glyph FP32",
            "maximum_test_label_length": max_chars, "max_plate_macs": MACS_PER_CHAR * max_chars,
            "existing_sdc_clock_target_hz": 50_000_000,
            "single_mac_per_cycle_theoretical_cycles_per_glyph": MACS_PER_CHAR,
            "single_mac_per_cycle_theoretical_ms_per_glyph_at_50mhz": MACS_PER_CHAR / 50_000,
            "single_mac_per_cycle_theoretical_cycles_per_max_plate": MACS_PER_CHAR * max_chars,
            "single_mac_per_cycle_theoretical_ms_per_max_plate_at_50mhz": MACS_PER_CHAR * max_chars / 50_000,
            "fpga_resource_and_fmax_measurement": "not available: no AI inference RTL or Quartus compile/STA result",
        },
        "latency": latency,
        "artifact_paths": {
            "checkpoint": str(args.output), "json": str(args.metrics), "report": str(args.report),
            "previous_test_confusion_csv": "artifacts/frontal1000_test_previous_confusion.csv",
            "finetuned_test_confusion_csv": "artifacts/frontal1000_test_finetuned_runtime_aligned_confusion.csv",
            "previous_runtime_aligned_confusion_csv": "artifacts/frontal1000_test_previous_runtime_aligned_confusion.csv",
            "finetuned_runtime_aligned_confusion_csv": "artifacts/frontal1000_test_finetuned_runtime_aligned_confusion.csv",
            "confusion_cost_M_csv": "artifacts/frontal1000_confusion_cost_M.csv",
            "focused_confusion_submatrix_csv": "artifacts/frontal1000_confusion_submatrix.csv",
        },
        "caveats": [
            "This is fine-tuning from the previous Conv2-pruned checkpoint, not training from random initialization.",
            "One of 500 test identities is unique and disjoint from previous train/validation identity manifest; image quality gates applied. Test segmentation skips count as failures for end-to-end exact accuracy.",
            "Character-level test accuracy is conditional on reference segmentation success; report its coverage beside the accuracy.",
            "Prediction confidence uses only aligned reference crops; it is not a calibrated probability of an entire plate being correct.",
            "FPGA clock/resource numbers are not synthesized measurements. 50 MHz values are a single-MAC/cycle lower-level workload estimate only.",
            "Plate IDs and per-plate predictions are not included in the aggregate report.",
            "Seven label corrections confirmed by visually checking the selected training crops are applied through a local ignored file; the original source/preparation manifest is unchanged.",
        ],
    }
    class_balance = metrics["data"]["class_balance"]
    with (out_dir / "frontal1000_class_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "train_count", "train_fraction", "validation_count", "test_aligned_count"])
        for char in CLASS_NAMES:
            train_n = class_balance["train"]["counts"][char]
            val_n = class_balance["validation"]["counts"][char]
            test_n = class_balance["test_segmentable"]["counts"][char]
            writer.writerow([char, train_n, train_n / max(1, class_balance["train"]["total_characters"]), val_n, test_n])
    total_augmented_presentations = max(1, len(train_set) * len(history))
    with (out_dir / "frontal1000_augmentation_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["category", "observed_count", "observed_fraction"])
        for key, count in sorted(train_set.events.items()):
            writer.writerow([key, count, count / total_augmented_presentations])
    metrics["artifact_paths"]["class_distribution_csv"] = "artifacts/frontal1000_class_distribution.csv"
    metrics["artifact_paths"]["augmentation_distribution_csv"] = "artifacts/frontal1000_augmentation_distribution.csv"
    # Keep model instances out of JSON/report; only parameter counts are needed in tables.
    metrics["_models"] = {"previous_conv2_student": prior_model, "frontal1000_finetuned": model}
    metrics["json_path_for_report"] = str(args.metrics)
    # Produce a public-safe shallow view and construct the report from it.
    params = {key: count_parameters(value) for key, value in metrics.pop("_models").items()}
    metrics["parameter_counts_for_report"] = params
    metrics["training"].pop("history", None)
    metrics["training"]["epoch_history"] = history
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # Markdown helper needs only counts; avoid serializing tensors or model objects.
    metrics["_models"] = {k: SimpleCount(v) for k, v in params.items()}
    markdown_report(args.report, metrics)
    del metrics["_models"]
    for name, rows in (("previous", prior_rows), ("finetuned", final_rows)):
        with (out_dir / f"frontal1000_{name}_test_cases_local.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else []); writer.writeheader(); writer.writerows(rows)
    print(f"saved checkpoint={args.output} metrics={args.metrics} report={args.report}", flush=True)


class SimpleCount:
    def __init__(self, count: int): self.count = count
    def __call__(self, _): return self.count


if __name__ == "__main__":
    main()
