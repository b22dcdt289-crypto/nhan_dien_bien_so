from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_cost_sensitive_conv2_distill import count_parameters, load_checkpoint
from train_frontal1000_conv2_cost import (
    activation_distributions,
    benchmark,
    load_prior_cost,
    save_confusion,
    weight_distributions,
    weighted_ohem_loss,
)
from train_lenet5 import CLASS_NAMES, LeNet5
from train_independent_structured import segment_characters


ONE_ROW_DATA = Path("data/train1_frontal_1000_unseen_v3")
TWO_ROW_DATA = Path("data/two_row_frontal_dense_v1")
PHASE1_METRICS = Path("artifacts/frontal1000_pruning_vs_no_pruning_metrics.json")
PHASE1_CHECKPOINT = Path("artifacts/lenet5_dense_frontal1000_cost_v1.pt")
PRIOR_COST_METRICS = Path("artifacts/channel30_cost_hard_conv2_distill_metrics.json")
MACS_PER_GLYPH = 418_200


class CachedGlyphDataset(Dataset):
    """Load aligned 32x32 glyph crops once; augment them on access."""

    def __init__(self, domains: list[tuple[Path, list[dict]]], augment: bool):
        from train_frontal1000_conv2_cost import ManifestCharacters

        self.augment = augment
        self.items: list[tuple[np.ndarray, int]] = []
        self.plates: list[dict] = []
        self.events = Counter()
        class_to_idx = {char: index for index, char in enumerate(CLASS_NAMES)}
        for root, records in domains:
            for record in records:
                files = record.get("crop_files", [])
                text = record.get("text", "")
                if not record.get("ground_truth_segmentation_ok") or len(files) != len(text):
                    continue
                if any(char not in class_to_idx for char in text):
                    continue
                indices = []
                loaded = []
                for relative in files:
                    try:
                        with Image.open(root / relative) as image:
                            pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
                    except (OSError, ValueError):
                        loaded = []
                        break
                    if pixels.shape != (32, 32):
                        pixels = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA)
                    loaded.append(pixels)
                if len(loaded) != len(text):
                    continue
                for pixels, character in zip(loaded, text):
                    indices.append(len(self.items))
                    self.items.append((pixels, class_to_idx[character]))
                self.plates.append({
                    "text": text,
                    "type": record.get("type", "one_row"),
                    "split": record.get("split", "unknown"),
                    "indices": indices,
                })
        self._augment_geometry = ManifestCharacters._augment_geometry
        self._augment_photo = ManifestCharacters._augment_photo

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pixels, target = self.items[index]
        gray = pixels.copy()
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


def apply_verified_one_row_corrections(records: list[dict], data_root: Path):
    correction_path = data_root / "review" / "verified_label_corrections.json"
    corrections = json.loads(correction_path.read_text(encoding="utf-8")) if correction_path.is_file() else []
    by_source = {item["source_file"]: item["corrected_text"] for item in corrections}
    applied = 0
    for record in records:
        proposed = by_source.get(Path(record.get("source", "")).name)
        if record.get("split") == "train" and proposed is not None:
            if len(proposed) != len(record["text"]) or any(character not in CLASS_NAMES for character in proposed):
                raise ValueError("Invalid verified correction in the one-row training split")
            record["text"] = proposed
            applied += 1
    if applied != len(corrections):
        raise RuntimeError("One-row verified label correction count did not match selected train records")
    return len(corrections)


def per_class_report(confusion: np.ndarray):
    rows = []
    for index, character in enumerate(CLASS_NAMES):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        correct = int(confusion[index, index])
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "character": character,
            "support": support,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    supported = [row for row in rows if row["support"]]
    return rows, {
        name: float(np.mean([row[name] for row in supported])) if supported else None
        for name in ("precision", "recall", "f1")
    }


def confidence_histogram(values: list[float]):
    bins = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.000001))
    return {
        f"{low:.0%}-{min(high, 1.0):.0%}": int(sum(low <= value < high for value in values))
        for low, high in bins
    }


def activation_distributions_cached(model, dataset: CachedGlyphDataset, device: torch.device, limit: int = 256):
    names = ("features.0", "features.1", "features.2", "features.3", "features.4", "features.5", "classifier.0", "classifier.1", "classifier.2", "classifier.3", "classifier.4")
    if not dataset.items:
        return {"sample_glyphs": 0, "layers": {}}
    tensors = [
        torch.from_numpy((pixels.astype(np.float32) / 127.5 - 1.0)).unsqueeze(0)
        for pixels, _ in dataset.items[:limit]
    ]
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
    layers = {}
    for name, array in captured.items():
        values = array.reshape(-1)
        layers[name] = {
            "shape_per_glyph": list(array.shape[1:]),
            "elements_per_glyph": int(array[0].size),
            "sampled_values": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "zero_fraction": float(np.mean(values == 0)),
            "quantiles_01_50_99": [float(value) for value in np.quantile(values, [0.01, 0.5, 0.99])],
        }
    return {"sample_glyphs": len(tensors), "source_split": "one-row validation reference crops", "layers": layers}


def evaluate_reference(model, dataset: CachedGlyphDataset, device: torch.device, batch_size: int = 512):
    if not dataset.items:
        return {"plates": 0, "characters": 0, "character_accuracy": None, "plate_exact_accuracy": None}, np.zeros((30, 30), dtype=np.int64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_conf = [], []
    model.eval()
    with torch.inference_mode():
        for images, _targets in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_conf.extend(confidence.cpu().tolist())
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    chars = correct_chars = exact_plates = 0
    by_type = defaultdict(lambda: {"plates": 0, "exact_plates": 0, "characters": 0, "correct_characters": 0})
    by_length = defaultdict(lambda: {"plates": 0, "exact_plates": 0, "characters": 0, "correct_characters": 0})
    char_conf, correct_conf, wrong_conf = [], [], []
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for plate in dataset.plates:
        indices = plate["indices"]
        predicted = "".join(CLASS_NAMES[all_preds[index]] for index in indices)
        truth = plate["text"]
        exact = predicted == truth
        exact_plates += int(exact)
        chars += len(truth)
        by_type[plate["type"]]["plates"] += 1
        by_type[plate["type"]]["exact_plates"] += int(exact)
        by_type[plate["type"]]["characters"] += len(truth)
        by_length[len(truth)]["plates"] += 1
        by_length[len(truth)]["exact_plates"] += int(exact)
        by_length[len(truth)]["characters"] += len(truth)
        for offset, (index, expected) in enumerate(zip(indices, truth)):
            guess = CLASS_NAMES[all_preds[index]]
            is_correct = guess == expected
            correct_chars += int(is_correct)
            by_type[plate["type"]]["correct_characters"] += int(is_correct)
            by_length[len(truth)]["correct_characters"] += int(is_correct)
            confusion[CLASS_NAMES.index(expected), CLASS_NAMES.index(guess)] += 1
            char_conf.append(all_conf[index])
            (correct_conf if is_correct else wrong_conf).append(all_conf[index])
            position_stats[offset + 1]["total"] += 1
            position_stats[offset + 1]["correct"] += int(is_correct)
    per_char, macro = per_class_report(confusion)
    return {
        "plates": len(dataset.plates),
        "characters": chars,
        "correct_characters": correct_chars,
        "character_accuracy": correct_chars / max(1, chars),
        "exact_plates": exact_plates,
        "plate_exact_accuracy": exact_plates / max(1, len(dataset.plates)),
        "macro_precision_recall_f1": macro,
        "per_character": per_char,
        "by_plate_type": {
            key: {
                **value,
                "character_accuracy": value["correct_characters"] / max(1, value["characters"]),
                "plate_exact_accuracy": value["exact_plates"] / max(1, value["plates"]),
            }
            for key, value in by_type.items()
        },
        "by_character_count": {
            str(key): {
                **value,
                "character_accuracy": value["correct_characters"] / max(1, value["characters"]),
                "plate_exact_accuracy": value["exact_plates"] / max(1, value["plates"]),
            }
            for key, value in by_length.items()
        },
        "mean_max_softmax_confidence": float(np.mean(char_conf)) if char_conf else None,
        "mean_confidence_on_correct_characters": float(np.mean(correct_conf)) if correct_conf else None,
        "mean_confidence_on_wrong_characters": float(np.mean(wrong_conf)) if wrong_conf else None,
        "confidence_is_calibrated": False,
        "confidence_histogram": confidence_histogram(char_conf),
        "character_accuracy_by_sequence_position": {
            str(position): {**stats, "accuracy": stats["correct"] / max(1, stats["total"])}
            for position, stats in sorted(position_stats.items())
        },
    }, confusion


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate_runtime(model, root: Path, records: list[dict], device: torch.device):
    attempted = exact = correct_count_plates = aligned_chars = correct_chars = 0
    counts, rows, skip_reasons = Counter(), Counter(), Counter()
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    by_type = defaultdict(lambda: {"plates": 0, "attempted": 0, "exact": 0, "count_aligned": 0})
    by_length = defaultdict(lambda: {"plates": 0, "attempted": 0, "exact": 0, "count_aligned": 0})
    confidences, correct_conf, wrong_conf = [], [], []
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    distance_sum = 0
    model.eval()
    with torch.inference_mode():
        for record in records:
            plate_type, truth = record["type"], record["text"]
            by_type[plate_type]["plates"] += 1
            by_length[len(truth)]["plates"] += 1
            image_path = record.get("plate_crop")
            image = cv2.imread(str(root / image_path), cv2.IMREAD_GRAYSCALE) if image_path else None
            if image is None:
                skip_reasons["missing_or_unreadable_plate_crop"] += 1
                continue
            segmented = segment_characters(image, expected_count=None, enhancement=record.get("enhancement", "clahe_sharp"))
            if segmented is None:
                skip_reasons["segmentation_failed"] += 1
                continue
            crops, row_count, count = segmented
            counts[str(count)] += 1
            rows[str(row_count)] += 1
            if row_count != 2:
                skip_reasons["not_two_rows"] += 1
                continue
            if count not in (7, 8, 9, 10):
                skip_reasons["unsupported_character_count"] += 1
                continue
            batch = torch.stack([
                torch.from_numpy((crop.astype(np.float32) / 127.5 - 1.0)).unsqueeze(0)
                for crop in crops
            ]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            predicted_text = "".join(CLASS_NAMES[index] for index in pred.cpu().tolist())
            predicted_conf = conf.cpu().tolist()
            attempted += 1
            by_type[plate_type]["attempted"] += 1
            by_length[len(truth)]["attempted"] += 1
            exact_match = predicted_text == truth
            exact += int(exact_match)
            by_type[plate_type]["exact"] += int(exact_match)
            by_length[len(truth)]["exact"] += int(exact_match)
            distance_sum += edit_distance(truth, predicted_text)
            if count == len(truth):
                correct_count_plates += 1
                by_type[plate_type]["count_aligned"] += 1
                by_length[len(truth)]["count_aligned"] += 1
                aligned_chars += len(truth)
                for expected, guess, confidence in zip(truth, predicted_text, predicted_conf):
                    is_correct = expected == guess
                    correct_chars += int(is_correct)
                    confusion[CLASS_NAMES.index(expected), CLASS_NAMES.index(guess)] += 1
                    confidences.append(confidence)
                    (correct_conf if is_correct else wrong_conf).append(confidence)
                for position, (expected, guess) in enumerate(zip(truth, predicted_text), start=1):
                    if position > count:
                        break
                    position_stats[position]["total"] += 1
                    position_stats[position]["correct"] += int(expected == guess)
            else:
                skip_reasons["detected_count_mismatch"] += 1
    return {
        "plate_frames": len(records),
        "plates_with_valid_two_row_output": attempted,
        "coverage": attempted / max(1, len(records)),
        "skipped": len(records) - attempted,
        "skip_reasons": dict(skip_reasons),
        "detected_character_count_histogram": dict(counts),
        "detected_row_count_histogram": dict(rows),
        "full_plate_exact_all_frames_skips_fail": exact,
        "full_plate_exact_accuracy_all_frames_skips_fail": exact / max(1, len(records)),
        "full_plate_exact_accuracy_among_valid_two_row_outputs": exact / max(1, attempted),
        "plates_with_count_aligned_characters": correct_count_plates,
        "character_count_aligned": aligned_chars,
        "correct_characters_when_count_aligned": correct_chars,
        "character_accuracy_conditional_on_detected_count_match": correct_chars / max(1, aligned_chars) if aligned_chars else None,
        "count_aligned_plate_coverage": correct_count_plates / max(1, len(records)),
        "sum_levenshtein_distance_on_attempted_outputs": distance_sum,
        "mean_max_softmax_confidence_on_count_aligned_characters": float(np.mean(confidences)) if confidences else None,
        "mean_confidence_on_correct_characters": float(np.mean(correct_conf)) if correct_conf else None,
        "mean_confidence_on_wrong_characters": float(np.mean(wrong_conf)) if wrong_conf else None,
        "confidence_is_calibrated": False,
        "confidence_histogram": confidence_histogram(confidences),
        "character_accuracy_by_sequence_position_when_count_aligned": {
            str(position): {**stats, "accuracy": stats["correct"] / max(1, stats["total"])}
            for position, stats in sorted(position_stats.items())
        },
        "by_plate_type": {
            key: {
                **value,
                "coverage": value["attempted"] / max(1, value["plates"]),
                "full_plate_exact_accuracy_all_frames": value["exact"] / max(1, value["plates"]),
                "full_plate_exact_accuracy_among_valid_outputs": value["exact"] / max(1, value["attempted"]),
            }
            for key, value in by_type.items()
        },
        "by_character_count": {
            str(key): {
                **value,
                "coverage": value["attempted"] / max(1, value["plates"]),
                "full_plate_exact_accuracy_all_frames": value["exact"] / max(1, value["plates"]),
                "full_plate_exact_accuracy_among_valid_outputs": value["exact"] / max(1, value["attempted"]),
            }
            for key, value in by_length.items()
        },
        "confusion_matrix_count_aligned_only": confusion.tolist(),
    }


def write_class_comparison(path: Path, before: dict, after: dict):
    before_rows = {row["character"]: row for row in before["per_character"]}
    after_rows = {row["character"]: row for row in after["per_character"]}
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["character", "test_support", "before_correct", "before_recall", "after_correct", "after_recall", "after_precision", "after_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for character in CLASS_NAMES:
            b, a = before_rows[character], after_rows[character]
            writer.writerow({
                "character": character,
                "test_support": a["support"],
                "before_correct": b["correct"],
                "before_recall": b["recall"],
                "after_correct": a["correct"],
                "after_recall": a["recall"],
                "after_precision": a["precision"],
                "after_f1": a["f1"],
            })


def render_report(path: Path, metrics: dict):
    baseline, trained = metrics["models"]["phase1_dense_before_two_row"], metrics["models"]["dense_after_two_row_finetune"]
    after_reference, after_runtime = trained["two_row_test_reference"], trained["two_row_test_runtime"]
    baseline_test, baseline_runtime = baseline["two_row_test_reference"], baseline["two_row_test_runtime"]
    runtime_cpu = metrics["latency_cpu"]["batches"]
    lines = [
        "# Báo cáo huấn luyện LeNet-5 dense: biển một hàng → biển hai hàng",
        "",
        "## Tóm tắt theo điều kiện đặt ra",
        "",
        f"Pha 1 dùng 1.000 biển ô tô một hàng, đạt {metrics['phase1_gate']['validation_character_accuracy']:.3%} độ chính xác ký tự và {metrics['phase1_gate']['validation_plate_exact_accuracy']:.3%} exact trên crop validation; đạt ngưỡng tối thiểu 95% nên mới chuyển sang pha hai hàng. Mức trên 99% vẫn được tính là đạt. Không dùng pruning.",
        f"Pha 2 fine-tune {metrics['training']['epochs_completed']} epoch; epoch tốt nhất {metrics['training']['best_epoch']} có validation hai hàng {metrics['training']['epoch_history'][metrics['training']['best_epoch']-1]['validation_character_accuracy_two_row']:.3%} ký tự và {metrics['training']['epoch_history'][metrics['training']['best_epoch']-1]['validation_plate_exact_two_row']:.3%} exact biển.",
        "",
        "**Giới hạn pha 1:** số trên là validation có crop tham chiếu, không phải phép đo độc lập cuối cùng. Test 500 biển trước đó có 0 crop ký tự tham chiếu và segmenter không trả chuỗi 8 ký tự cho biển nào; do đó độ chính xác OCR/end-to-end trên test một hàng vẫn chưa được chứng minh.",
        "",
        "## Dữ liệu hai hàng và xử lý ảnh",
        "",
        f"- Nguồn: `{metrics['data']['source']}`; ảnh biển hai hàng ô tô và xe máy được gom theo chuỗi biển trước khi chia tập. Số frame đã lưu: {metrics['data']['frames_saved']:,}; số danh tính biển duy nhất theo train/validation/test: {metrics['data']['unique_plate_identities_by_split']}.",
        f"- Chia nhóm xác định: train {metrics['data']['split_percentages']['train']:.1f}%, validation {metrics['data']['split_percentages']['val']:.1f}%, test {metrics['data']['split_percentages']['test']:.1f}%; giao nhau danh tính = 0 ở cả ba cặp. Đã loại khỏi train {metrics['data']['training_records_excluded_for_cross_domain_heldout_identity']['two_row']:,} frame hai hàng và {metrics['data']['training_records_excluded_for_cross_domain_heldout_identity']['one_row_replay']:,} frame replay một hàng do trùng text biển với miền validation/test còn lại.",
        f"- Cắt ROI theo bbox nguồn → tìm tứ giác và warp phối cảnh khi đủ tin cậy → CLAHE + lọc nhiễu/làm nét nhẹ → phân đoạn và phân loại hàng → sắp từ trái sang phải ở hàng trên, rồi hàng dưới → LeNet-5 nhận dạng từng ký tự.",
        f"- Chỉnh hình học: homography 4 góc {metrics['data']['rectification_method_counts'].get('four_corner_homography', 0):,} ({metrics['data']['rectification_method_counts'].get('four_corner_homography', 0)/max(1, metrics['data']['frames_saved']):.1%}); min-area rotated rectangle {metrics['data']['rectification_method_counts'].get('rotated_rectangle', 0):,}; fallback bbox {metrics['data']['rectification_method_counts'].get('bbox_fallback', 0):,}. Fallback được giữ và tính riêng, không tuyên bố đã chỉnh thẳng.",
        f"- Phân đoạn có số ký tự tham chiếu đúng và ra 2 hàng: {metrics['data']['reference_segmentation_success_by_type']} (theo loại xe). Những biển thất bại vẫn được giữ trong test runtime để tính skip/coverage.",
        f"- Train hai hàng sau lọc identity: {metrics['data']['class_balance']['plates_with_reference_crops']:,} biển có crop tham chiếu, {metrics['data']['class_balance']['characters']:,} crop ký tự; đủ {metrics['data']['class_balance']['classes_with_support']}/30 lớp. Tỉ số lớp nhiều mẫu nhất/ít nhất có mẫu = {metrics['data']['class_balance']['class_balance_ratio_max_over_min_nonzero']:.1f}:1.",
        "",
        "## Nhận dạng ký tự và exact biển trên crop tham chiếu",
        "",
        "| Tập / loại xe | Mô hình trước khi thêm hai hàng | Sau fine-tune hai hàng |",
        "|---|---:|---:|",
        f"| Test hai hàng, ký tự (crop tham chiếu) | {baseline_test['character_accuracy']:.3%} ({baseline_test['correct_characters']:,}/{baseline_test['characters']:,}) | {after_reference['character_accuracy']:.3%} ({after_reference['correct_characters']:,}/{after_reference['characters']:,}) |",
        f"| Test hai hàng, exact chuỗi (crop tham chiếu) | {baseline_test['plate_exact_accuracy']:.3%} ({baseline_test['exact_plates']}/{baseline_test['plates']}) | {after_reference['plate_exact_accuracy']:.3%} ({after_reference['exact_plates']}/{after_reference['plates']}) |",
        f"| Macro precision / recall / F1 | {baseline_test['macro_precision_recall_f1']['precision']:.3%} / {baseline_test['macro_precision_recall_f1']['recall']:.3%} / {baseline_test['macro_precision_recall_f1']['f1']:.3%} | {after_reference['macro_precision_recall_f1']['precision']:.3%} / {after_reference['macro_precision_recall_f1']['recall']:.3%} / {after_reference['macro_precision_recall_f1']['f1']:.3%} |",
    ]
    for plate_type, title in (("two_row_car", "Ô tô hai hàng"), ("two_row_motorcycle", "Xe máy hai hàng")):
        b = baseline_test.get("by_plate_type", {}).get(plate_type)
        a = after_reference.get("by_plate_type", {}).get(plate_type)
        if b and a:
            lines.append(f"| {title}, ký tự / exact crop | {b['character_accuracy']:.3%} / {b['plate_exact_accuracy']:.3%} | {a['character_accuracy']:.3%} / {a['plate_exact_accuracy']:.3%} |")
    lines += [
        "",
        "Đây là accuracy OCR khi đã có crop ký tự được căn chỉnh, không bao gồm lỗi tìm biển/phân đoạn tự do. Tập test được tách theo chuỗi biển nhưng là một phân hoạch mới của nguồn train(1); nó có thể trùng dữ liệu đã dùng để pretrain checkpoint cũ. Vì vậy so sánh trước/sau đo tác dụng fine-tune trên phân hoạch hiện tại, chưa phải đánh giá external-dataset độc lập.",
        "",
        "## Pipeline tự chạy, không biết trước số ký tự",
        "",
        "| Chỉ số trên test thô | Trước fine-tune | Sau fine-tune |",
        "|---|---:|---:|",
        f"| Độ phủ đầu ra hợp lệ 2 hàng | {baseline_runtime['coverage']:.1%} ({baseline_runtime['plates_with_valid_two_row_output']}/{baseline_runtime['plate_frames']}) | {after_runtime['coverage']:.1%} ({after_runtime['plates_with_valid_two_row_output']}/{after_runtime['plate_frames']}) |",
        f"| Exact biển end-to-end, bỏ qua tính là sai | {baseline_runtime['full_plate_exact_accuracy_all_frames_skips_fail']:.3%} ({baseline_runtime['full_plate_exact_all_frames_skips_fail']}/{baseline_runtime['plate_frames']}) | {after_runtime['full_plate_exact_accuracy_all_frames_skips_fail']:.3%} ({after_runtime['full_plate_exact_all_frames_skips_fail']}/{after_runtime['plate_frames']}) |",
        f"| Ký tự đúng, chỉ khi số crop khớp nhãn | {baseline_runtime['character_accuracy_conditional_on_detected_count_match']:.3%} | {after_runtime['character_accuracy_conditional_on_detected_count_match']:.3%} |",
        f"| Chuỗi có số lượng ký tự khớp nhãn | {baseline_runtime['plates_with_count_aligned_characters']}/{baseline_runtime['plate_frames']} | {after_runtime['plates_with_count_aligned_characters']}/{after_runtime['plate_frames']} |",
        "",
        f"Histogram số crop sau phân đoạn của mô hình đã fine-tune: `{json.dumps(after_runtime['detected_character_count_histogram'], ensure_ascii=False)}`; hàng phát hiện: `{json.dumps(after_runtime['detected_row_count_histogram'], ensure_ascii=False)}`. Phần này quyết định exact end-to-end; accuracy CNN cao không bù được cắt thiếu/thừa ký tự.",
        f"- Pipeline không phát output ở {after_runtime['skipped']:,}/{after_runtime['plate_frames']:,} frame ({after_runtime['skipped']/max(1, after_runtime['plate_frames']):.2%}); nguyên nhân: `{json.dumps({key: value for key, value in after_runtime['skip_reasons'].items() if key != 'detected_count_mismatch'}, ensure_ascii=False)}`. Ngoài ra có {after_runtime['skip_reasons'].get('detected_count_mismatch', 0):,} output lệch số lượng ký tự; các trường hợp này không được dùng cho accuracy ký tự căn chỉnh.",
        f"- Character accuracy runtime chỉ trên các chuỗi có số crop khớp nhãn: {after_runtime['character_accuracy_conditional_on_detected_count_match']:.3%} ({after_runtime['correct_characters_when_count_aligned']:,}/{after_runtime['character_count_aligned']:,} ký tự); chỉ {after_runtime['plates_with_count_aligned_characters']:,}/{after_runtime['plate_frames']:,} frame căn chỉnh đủ số lượng.",
        "",
        "## Độ chính xác theo từng lớp ký tự",
        "",
        "Recall ở đây là accuracy trong lớp; support các lớp không có dữ liệu test bằng 0 thì không thể đánh giá. Xem CSV để đối chiếu baseline/fine-tune theo từng lớp.",
        "",
        "| Ký tự | Số mẫu test | Trước: recall | Sau: recall | Sau: precision | Sau: F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    before_rows = {row["character"]: row for row in baseline_test["per_character"]}
    after_rows = {row["character"]: row for row in after_reference["per_character"]}
    for char in CLASS_NAMES:
        b, a = before_rows[char], after_rows[char]
        if a["support"]:
            lines.append(f"| {char} | {a['support']} | {b['recall']:.2%} | {a['recall']:.2%} | {a['precision']:.2%} | {a['f1']:.2%} |")
    lines += [
        "",
        "## Loss, hard-example mining và augmentation",
        "",
        f"- Loss: weighted cross-entropy + λ={metrics['training']['confusion_penalty_lambda']} × kỳ vọng ma trận phạt nhầm lẫn M; OHEM top {metrics['training']['hard_fraction']:.0%} được tăng trọng số, không bỏ phần còn lại.",
        f"- Tối ưu: AdamW, LR đầu {metrics['training']['learning_rate_initial']:.2g}, cosine annealing, batch {metrics['training']['batch_size']}; hoàn thành {metrics['training']['epochs_completed']} epoch, checkpoint chọn theo accuracy ký tự validation trước, exact biển làm tiêu chí phụ; epoch tốt nhất {metrics['training']['best_epoch']}.",
        f"- Replay giữ kiến thức biển một hàng: {metrics['training']['one_row_replay_characters']:,} ký tự trong {metrics['training']['one_row_replay_plates']:,} biển train, trộn cùng ký tự hai hàng để hạn chế quên pha 1.",
        f"- Augmentation quan sát được trong train: `{json.dumps(metrics['training']['augmentation_observed_rates'], ensure_ascii=False)}`; gồm xoay/affine/perspective nhẹ, blur/noise/độ sáng-contrast.",
        "",
        "## Tham số, MAC, bộ nhớ và tốc độ",
        "",
        f"- Dense LeNet-5 (Conv1=6, Conv2=16, FC=120/84, 30 lớp): **{metrics['architecture']['parameters']:,} tham số**, **{MACS_PER_GLYPH:,} MAC/ký tự**.",
        f"- MAC theo độ dài: " + "; ".join(f"{n} ký tự = {v:,}" for n, v in metrics["architecture"]["macs_per_plate_by_length"].items()) + ".",
        f"- Trọng số thô: FP32 {metrics['resources']['fp32_weight_bytes']:,} B ({metrics['resources']['fp32_weight_kib']:.2f} KiB); 8-bit lý thuyết {metrics['resources']['int8_weight_bytes_estimate']:,} B (chưa phải mô hình lượng tử hóa kiểm định). Đỉnh activation riêng lẻ {metrics['resources']['peak_activation_elements_per_glyph']:,} giá trị = {metrics['resources']['peak_activation_fp32_bytes_per_glyph']:,} B FP32/ký tự.",
        f"- CPU PyTorch 1 luồng, p50/p95 batch 1: trước {runtime_cpu['1']['phase1_dense_before_two_row']['median_ms_per_forward']:.4f}/{runtime_cpu['1']['phase1_dense_before_two_row']['p95_ms_per_forward']:.4f} ms, sau {runtime_cpu['1']['dense_after_two_row_finetune']['median_ms_per_forward']:.4f}/{runtime_cpu['1']['dense_after_two_row_finetune']['p95_ms_per_forward']:.4f} ms; batch 8 p50: trước {runtime_cpu['8']['phase1_dense_before_two_row']['median_ms_per_forward']:.4f} ms, sau {runtime_cpu['8']['dense_after_two_row_finetune']['median_ms_per_forward']:.4f} ms. Cùng topology/MAC nên đây không phải tăng tốc kiến trúc.",
        f"- Ước lượng số học nếu DE10-Lite chạy 50 MHz và một MAC/chu kỳ: 8 ký tự {MACS_PER_GLYPH*8/50_000:.3f} ms; 9 ký tự {MACS_PER_GLYPH*9/50_000:.3f} ms; 10 ký tự {MACS_PER_GLYPH*10/50_000:.3f} ms. Chưa có RTL AI/Quartus synthesis nên không có fMAX, DSP/ALM/BRAM hoặc latency FPGA đo thật.",
        "",
        "## Kết luận và giới hạn",
        "",
        "Training fine-tune này tối ưu nhận dạng ký tự có crop tham chiếu trước, sau đó mới nhìn exact biển. Nếu exact/coverage runtime thấp hơn rõ rệt so với crop reference thì nút thắt là ROI, phân đoạn hoặc row-sort chứ không chỉ là LeNet-5. Các biển xe máy/ô tô có số ký tự khác nhau được so khớp theo chuỗi nhãn thực tế; không giả định mọi biển có cùng chiều dài.",
        "",
        "Kết quả có thể bị ảnh hưởng bởi nhãn filename chưa kiểm tra thủ công toàn bộ, nhóm test mới có khả năng đã xuất hiện trong pretraining cũ, và các crop tham chiếu dùng số lượng ký tự nhãn để chọn phân đoạn. Confidence softmax chưa được calibration. Báo cáo không tuyên bố đạt 95–99% trên camera/FPGA nếu số liệu end-to-end chưa đạt.",
        "",
        f"Số liệu đầy đủ JSON: `{metrics['artifact_paths']['metrics_json']}`; báo cáo theo lớp: `{metrics['artifact_paths']['per_class_csv']}`; lịch sử epoch: `{metrics['artifact_paths']['history_csv']}`. Ảnh từng biển/ID không được đưa lên GitHub.",
    ]
    lines.extend([
        "",
        "### Số lượng theo split nguồn (trước lọc replay chéo)",
        "",
        "| Split | Ô tô hai hàng | Xe máy hai hàng | Tổng frame |",
        "|---|---:|---:|---:|",
    ])
    for split in ("train", "val", "test"):
        vehicle_counts = metrics["data"]["plate_counts_by_type_split"][split]
        car_count = vehicle_counts.get("two_row_car", 0)
        moto_count = vehicle_counts.get("two_row_motorcycle", 0)
        lines.append(f"| {split} | {car_count:,} | {moto_count:,} | {car_count + moto_count:,} |")
    lines.extend([
        "",
        "## Phụ lục: runtime theo loại xe/độ dài và confidence",
        "",
        "| Nhóm test thô | Số frame | Độ phủ | Exact, skip tính sai | Exact trong output hợp lệ |",
        "|---|---:|---:|---:|---:|",
    ])
    for plate_type, title in (("two_row_car", "Ô tô hai hàng"), ("two_row_motorcycle", "Xe máy hai hàng")):
        row = after_runtime["by_plate_type"].get(plate_type)
        if row:
            lines.append(f"| {title} | {row['plates']} | {row['coverage']:.2%} | {row['full_plate_exact_accuracy_all_frames']:.2%} ({row['exact']}/{row['plates']}) | {row['full_plate_exact_accuracy_among_valid_outputs']:.2%} |")
    for length, row in sorted(after_runtime["by_character_count"].items(), key=lambda pair: int(pair[0])):
        lines.append(f"| {length} ký tự | {row['plates']} | {row['coverage']:.2%} | {row['full_plate_exact_accuracy_all_frames']:.2%} ({row['exact']}/{row['plates']}) | {row['full_plate_exact_accuracy_among_valid_outputs']:.2%} |")
    lines.extend([
        "",
        f"Confidence softmax trung bình trên crop tham chiếu: {after_reference['mean_max_softmax_confidence']:.4f}; trung bình ký tự đúng {after_reference['mean_confidence_on_correct_characters']:.4f}; ký tự sai {after_reference['mean_confidence_on_wrong_characters']:.4f} nếu có. Confidence chưa calibration nên không được hiểu như xác suất đúng đã hiệu chuẩn.",
        f"Accuracy theo vị trí chuỗi 1-based trên crop tham chiếu: `{json.dumps(after_reference['character_accuracy_by_sequence_position'], ensure_ascii=False)}`.",
        "",
        f"Activation được khảo sát trên {metrics['validation_activation_distributions']['dense_after_two_row_finetune']['sample_glyphs']} glyph validation; số liệu min/max/mean/std/quantile/zero fraction từng layer nằm trong JSON. Ma trận nhầm lẫn test 30×30 và ma trận cost M 30×30 được xuất CSV; tập test không tham gia xây M.",
        "",
        "## Chạy lại từ VS Code",
        "",
        "Trong thư mục gốc repo, dùng Python interpreter `.venv` rồi chạy lần lượt:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -X utf8 -u prepare_two_row_frontal_dense.py",
        ".\\.venv\\Scripts\\python.exe -X utf8 -u train_two_row_dense.py --epochs 12 --batch-size 256",
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Dense no-pruning LeNet-5 fine-tune from frontal one-row cars to two-row cars/motorcycles.")
    parser.add_argument("--one-row-data", type=Path, default=ONE_ROW_DATA)
    parser.add_argument("--two-row-data", type=Path, default=TWO_ROW_DATA)
    parser.add_argument("--phase1-checkpoint", type=Path, default=PHASE1_CHECKPOINT)
    parser.add_argument("--phase1-metrics", type=Path, default=PHASE1_METRICS)
    parser.add_argument("--prior-cost-metrics", type=Path, default=PRIOR_COST_METRICS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--confusion-penalty", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_dense_two_row_car_moto_v1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/two_row_dense_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/bao_cao_lenet5_dense_bien_hai_hang.md"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; architecture=LeNet-5 dense; pruning=disabled", flush=True)

    phase1 = json.loads(args.phase1_metrics.read_text(encoding="utf-8"))
    gate_char = phase1["models"]["unpruned"]["validation"]["character_accuracy_conditional_on_ground_truth_segmentation"]
    gate_plate = phase1["models"]["unpruned"]["validation"]["plate_exact_accuracy_all_test_plates"]
    # Treat the requested 95–99% as a target band with a 95% minimum;
    # a result above 99% must not prevent progression to the next phase.
    gate_char_threshold = gate_char >= 0.95
    gate_plate_threshold = gate_plate >= 0.95
    if not (gate_char_threshold and gate_plate_threshold):
        raise RuntimeError(
            f"Pha một chưa đạt ngưỡng tối thiểu validation 95% cho cả ký tự và biển: char={gate_char:.3%}, exact={gate_plate:.3%}; dừng trước pha hai."
        )
    phase1_gate = {
        "validation_character_accuracy": gate_char,
        "validation_plate_exact_accuracy": gate_plate,
        "character_gate_passed": gate_char_threshold,
        "plate_gate_passed": gate_plate_threshold,
        "validation_is_independent": False,
        "end_to_end_test_exact_accuracy": phase1["models"]["unpruned"]["test"]["runtime_no_known_count"]["exact_accuracy_all_including_skips"],
        "end_to_end_test_character_accuracy": phase1["models"]["unpruned"]["test_runtime_character_metrics"]["character_accuracy_conditional_on_runtime_count_match"],
        "proceeds_to_two_row_phase": True,
    }

    one_records = json.loads((args.one_row_data / "manifest.json").read_text(encoding="utf-8"))
    one_corrections = apply_verified_one_row_corrections(one_records, args.one_row_data)
    two_records = json.loads((args.two_row_data / "manifest.json").read_text(encoding="utf-8"))
    if any(record.get("type") not in ("two_row_car", "two_row_motorcycle") for record in two_records):
        raise RuntimeError("The stage-two manifest contains a non-two-row plate type")
    by_split = {split: [record for record in two_records if record["split"] == split] for split in ("train", "val", "test")}
    original_by_split = {split: list(rows) for split, rows in by_split.items()}
    ids = {split: {record["text"] for record in rows} for split, rows in by_split.items()}
    if any(ids[left] & ids[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Plate identity leakage detected in two-row preparation")
    if sum(record.get("ground_truth_segmentation_ok", False) for record in by_split["train"]) == 0:
        raise RuntimeError("No row-ordered two-row training crops are available")

    # Do not replay an identity from any held-out split in the other domain.
    # The already-trained phase-one checkpoint may have seen train(1), which is
    # disclosed in the report, but this prevents new fine-tuning leakage.
    heldout_ids = {
        record["text"]
        for record in [*by_split["val"], *by_split["test"]]
    } | {
        record["text"]
        for record in one_records
        if record["split"] in ("val", "test")
    }
    two_train_before_identity_filter = list(by_split["train"])
    by_split["train"] = [record for record in by_split["train"] if record["text"] not in heldout_ids]
    two_train_identity_excluded = len(two_train_before_identity_filter) - len(by_split["train"])
    one_train_before_identity_filter = [record for record in one_records if record["split"] == "train"]
    one_train_records = [record for record in one_train_before_identity_filter if record["text"] not in heldout_ids]
    one_train_identity_excluded = len(one_train_before_identity_filter) - len(one_train_records)
    if not any(record.get("ground_truth_segmentation_ok", False) for record in by_split["train"]):
        raise RuntimeError("No two-row training character crops remain after cross-domain identity filtering")
    one_val_records = [record for record in one_records if record["split"] == "val"]
    train_set = CachedGlyphDataset(
        [(args.one_row_data, one_train_records), (args.two_row_data, by_split["train"])], augment=True
    )
    one_val = CachedGlyphDataset([(args.one_row_data, one_val_records)], augment=False)
    two_val = CachedGlyphDataset([(args.two_row_data, by_split["val"])], augment=False)
    two_test_reference = CachedGlyphDataset([(args.two_row_data, by_split["test"])], augment=False)
    if not train_set.items or not two_val.items:
        raise RuntimeError("Training or validation character crops are empty")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = LeNet5(len(CLASS_NAMES)).to(device)
    initial_checkpoint = load_checkpoint(model, args.phase1_checkpoint, device)
    conv1 = model.features[0]
    conv1.weight.requires_grad_(False)
    conv1.bias.requires_grad_(False)
    frozen_conv1 = {name: value.detach().clone() for name, value in conv1.state_dict().items()}
    cost_np, prior_metrics = load_prior_cost(args.prior_cost_metrics)
    cost_matrix = torch.tensor(cost_np, dtype=torch.float32, device=device)
    count_by_class = Counter(target for _, target in train_set.items)
    max_class_count = max(count_by_class.values())
    raw_weights = torch.tensor(
        [math.sqrt(max_class_count / max(1, count_by_class.get(index, 0))) for index in range(len(CLASS_NAMES))],
        dtype=torch.float32,
    )
    class_weights = (raw_weights / raw_weights.mean()).clamp(0.5, 3.0).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_char = -1.0
    best_plate = -1.0
    best_epoch = 0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = hard_selected = 0
        loss_sum = 0.0
        started = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss, _, hard_count = weighted_ohem_loss(
                logits, labels, class_weights, cost_matrix, args.confusion_penalty, args.hard_fraction
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(labels)
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            hard_selected += hard_count
        scheduler.step()
        val_two, _ = evaluate_reference(model, two_val, device)
        val_one, _ = evaluate_reference(model, one_val, device)
        denominator = val_two["characters"] + val_one["characters"]
        numerator = val_two["correct_characters"] + val_one["correct_characters"]
        combined_char = numerator / max(1, denominator)
        combined_exact = (val_two["exact_plates"] + val_one["exact_plates"]) / max(1, val_two["plates"] + val_one["plates"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "train_character_accuracy": correct / max(1, seen),
            "validation_character_accuracy_two_row": val_two["character_accuracy"],
            "validation_plate_exact_two_row": val_two["plate_exact_accuracy"],
            "validation_character_accuracy_one_row_replay": val_one["character_accuracy"],
            "validation_plate_exact_one_row_replay": val_one["plate_exact_accuracy"],
            "validation_character_accuracy_combined": combined_char,
            "validation_plate_exact_combined": combined_exact,
            "hard_examples_selected_fraction": hard_selected / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        # Character accuracy is the primary selection metric; exact-plate rate breaks ties.
        if combined_char > best_char or (math.isclose(combined_char, best_char, abs_tol=1e-12) and combined_exact > best_plate):
            best_char, best_plate, best_epoch = combined_char, combined_exact, epoch
            torch.save(
                {
                    "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                    "classes": CLASS_NAMES,
                    "arch": "LeNet5_dense_no_pruning",
                    "channels": [6, 16, 120, 84],
                    "epoch": epoch,
                    "validation_character_accuracy": combined_char,
                    "validation_plate_exact_accuracy": combined_exact,
                    "macs_per_character": MACS_PER_GLYPH,
                    "params": count_parameters(model),
                    "two_row_train_plates": len(by_split["train"]),
                    "initialized_from": str(args.phase1_checkpoint),
                    "structured_pruning": "none",
                },
                args.output,
            )

    best_state = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    model.eval()
    if any(not torch.equal(frozen_conv1[name], value.detach()) for name, value in model.features[0].state_dict().items()):
        raise AssertionError("Conv1 changed although it was frozen")

    before = LeNet5(len(CLASS_NAMES)).to(device)
    load_checkpoint(before, args.phase1_checkpoint, device)
    before.eval()
    models = {"phase1_dense_before_two_row": before, "dense_after_two_row_finetune": model}
    results = {}
    confusion_matrices = {}
    for model_name, evaluated_model in models.items():
        print(f"Evaluation stage: {model_name}", flush=True)
        one_val_result, one_val_conf = evaluate_reference(evaluated_model, one_val, device)
        two_val_result, two_val_conf = evaluate_reference(evaluated_model, two_val, device)
        two_test_ref_result, two_test_ref_conf = evaluate_reference(evaluated_model, two_test_reference, device)
        raw_test_records = by_split["test"]
        two_test_runtime = evaluate_runtime(evaluated_model, args.two_row_data, raw_test_records, device)
        results[model_name] = {
            "checkpoint": str(args.phase1_checkpoint if model_name.startswith("phase1") else args.output),
            "one_row_validation_reference": one_val_result,
            "two_row_validation_reference": two_val_result,
            "two_row_test_reference": two_test_ref_result,
            "two_row_test_runtime": two_test_runtime,
        }
        confusion_matrices[model_name] = {
            "one_row_validation": one_val_conf,
            "two_row_validation": two_val_conf,
            "two_row_test_reference": two_test_ref_conf,
            "two_row_test_runtime": np.asarray(two_test_runtime["confusion_matrix_count_aligned_only"], dtype=np.int64),
        }

    cpu_benchmark = benchmark(before, model, device, repeats=500, warmup=100)
    cpu_benchmark["batches"] = {
        batch: {
            "phase1_dense_before_two_row": row["previous_conv2_student"],
            "dense_after_two_row_finetune": row["frontal1000_finetuned"],
        }
        for batch, row in cpu_benchmark["batches"].items()
    }

    chars_by_domain = {
        "one_row_train": CachedGlyphDataset([(args.one_row_data, one_train_records)], augment=False),
        "two_row_train": CachedGlyphDataset([(args.two_row_data, by_split["train"])], augment=False),
    }
    train_domain_stats = {}
    for domain, dataset in chars_by_domain.items():
        counts = Counter(CLASS_NAMES[target] for _, target in dataset.items)
        support = [count for count in counts.values() if count]
        train_domain_stats[domain] = {
            "plates_with_reference_crops": len(dataset.plates),
            "characters": len(dataset.items),
            "counts_by_class": {character: int(counts[character]) for character in CLASS_NAMES},
            "classes_with_support": len(support),
            "class_balance_ratio_max_over_min_nonzero": max(support) / min(support) if support else None,
        }

    perspective_by_type = Counter()
    for record in two_records:
        perspective_by_type[(record["type"], "warp" if record.get("perspective_corrected") else "roi_fallback")] += 1
    split_percentages = {split: 100.0 * len(rows) / max(1, len(two_records)) for split, rows in original_by_split.items()}
    rectification_method_counts = Counter(record.get("rectification_method", "unknown") for record in two_records)
    pruned_metrics = json.loads(args.prior_cost_metrics.read_text(encoding="utf-8"))
    parameters = count_parameters(model)
    max_acts = 6 * 28 * 28
    macs_by_length = {str(length): MACS_PER_GLYPH * length for length in range(7, 11)}
    report_json_path = args.metrics.as_posix()
    per_class_csv = "artifacts/two_row_test_per_class_comparison.csv"
    history_csv = "artifacts/two_row_dense_training_history.csv"
    metrics = {
        "experiment": "No-pruning dense LeNet-5 curriculum: frontal one-row 1,000-car gate followed by grouped two-row car/motorcycle fine-tuning",
        "phase1_gate": phase1_gate,
        "data": {
            "source": "data/OCR/OCR/images/train(1)/detection/{two_rows,two_rows_label_xe_may}",
            "prepared_directory": args.two_row_data.as_posix(),
            "frames_saved": len(two_records),
            "frame_counts_by_split": {split: len(rows) for split, rows in original_by_split.items()},
            "unique_plate_identities_by_split": {split: len(ids[split]) for split in by_split},
            "plate_counts_by_type_split": {
                split: dict(Counter(record["type"] for record in rows)) for split, rows in original_by_split.items()
            },
            "split_percentages": split_percentages,
            "training_records_excluded_for_cross_domain_heldout_identity": {
                "two_row": two_train_identity_excluded,
                "one_row_replay": one_train_identity_excluded,
                "unique_heldout_plate_texts": len(heldout_ids),
            },
            "identity_disjoint_across_splits": True,
            "class_balance": train_domain_stats["two_row_train"],
            "one_row_replay_distribution": train_domain_stats["one_row_train"],
            "reference_segmentation_success_by_type": {
                type_name: sum(record.get("ground_truth_segmentation_ok", False) and record["type"] == type_name for record in two_records)
                for type_name in ("two_row_car", "two_row_motorcycle")
            },
            "perspective_correction_applied_fraction": sum(value for (kind, status), value in perspective_by_type.items() if status == "warp") / max(1, len(two_records)),
            "perspective_correction_counts_by_type": {"|".join(key): value for key, value in perspective_by_type.items()},
            "rectification_method_counts": dict(rectification_method_counts),
            "two_row_training_frames_after_cross_domain_identity_filter": len(by_split["train"]),
            "preparation_metrics": json.loads((args.two_row_data / "preparation_metrics.json").read_text(encoding="utf-8")),
            "one_row_verified_training_label_corrections": one_corrections,
        },
        "training": {
            "architecture": "LeNet-5 dense, Conv2=16; no pruning",
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate_initial": args.learning_rate,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "class-weighted cross-entropy + expected confusion-cost penalty; all-sample mean and OHEM top-k mean equally weighted",
            "confusion_penalty_lambda": args.confusion_penalty,
            "hard_fraction": args.hard_fraction,
            "hard_samples_discarded_rate": 0.0,
            "one_row_replay_plates": train_domain_stats["one_row_train"]["plates_with_reference_crops"],
            "two_row_train_frames_after_cross_domain_identity_filter": len(by_split["train"]),
            "cross_domain_heldout_identity_exclusions": {
                "two_row_train_frames": two_train_identity_excluded,
                "one_row_replay_frames": one_train_identity_excluded,
            },
            "one_row_replay_characters": train_domain_stats["one_row_train"]["characters"],
            "class_weight_formula": "sqrt(max_count/class_count), normalized to mean 1, clipped [0.5, 3.0]",
            "conv1_frozen": True,
            "training_character_crops": len(train_set.items),
            "augmentation_observed": dict(train_set.events),
            "augmentation_observed_rates": {key: count / max(1, len(train_set.items) * len(history)) for key, count in train_set.events.items()},
            "epoch_history": history,
            "selection_rule": "maximize combined one-row/two-row validation character accuracy; use plate exact accuracy as tie-breaker",
            "initialized_from_dense_stage1": str(args.phase1_checkpoint),
            "teacher_confusion_source": "previous validation confusion matrix; no current test labels used in loss",
        },
        "models": results,
        "architecture": {
            "model": "LeNet-5 dense OCR classifier, no pruning",
            "channels": [6, 16, 120, 84],
            "parameters": parameters,
            "macs_per_character": MACS_PER_GLYPH,
            "macs_per_plate_by_length": macs_by_length,
            "conv1_frozen": True,
        },
        "resources": {
            "fp32_weight_bytes": parameters * 4,
            "fp32_weight_kib": parameters * 4 / 1024,
            "int8_weight_bytes_estimate": parameters,
            "int8_weight_kib_estimate": parameters / 1024,
            "peak_activation_elements_per_glyph": max_acts,
            "peak_activation_fp32_bytes_per_glyph": max_acts * 4,
            "largest_feature_map": "Conv1 output 6x28x28; Conv2 output 16x10x10",
            "assumed_clock_hz_for_workload_estimate": 50_000_000,
            "single_mac_per_cycle_theoretical_ms_by_length": {length: MACS_PER_GLYPH * length / 50_000 for length in range(7, 11)},
            "rtl_or_quartus_synthesis_available": False,
        },
        "latency_cpu": cpu_benchmark,
        "weight_distributions": {key: weight_distributions(value) for key, value in models.items()},
        "validation_activation_distributions": {
            key: activation_distributions_cached(value, one_val, device)
            for key, value in models.items()
        },
        "confusion_cost": {
            "matrix": load_prior_cost(args.prior_cost_metrics)[0].tolist(),
            "definition": "M[k,k]=0; M[k,j]=1+2*(C[k,j]+1)/(sum_{q!=k}C[k,q]+29); Laplace alpha=1",
            "source_validation_confusion": pruned_metrics["models"]["conv2_pruned_student"]["validation"]["confusion_matrix"],
            "current_two_row_test_not_used_for_cost_matrix": True,
        },
        "artifact_paths": {
            "dense_checkpoint": args.output.as_posix(),
            "metrics_json": report_json_path,
            "report_markdown": args.report.as_posix(),
            "per_class_csv": per_class_csv,
            "history_csv": history_csv,
            "test_confusion_before_csv": "artifacts/two_row_test_reference_before_confusion.csv",
            "test_confusion_after_csv": "artifacts/two_row_test_reference_after_confusion.csv",
            "confusion_cost_M_csv": "artifacts/two_row_confusion_cost_M.csv",
        },
        "caveats": [
            "Pha một 95–99% là validation crop tham chiếu; test one-row end-to-end chưa đo được do segmenter/label crops.",
            "Hai hàng được chia group-disjoint theo text nhưng nguồn train(1) có thể đã xuất hiện trong pretraining checkpoint giai đoạn trước.",
            "Reference crop accuracy dùng độ dài chuỗi nhãn để chọn candidate segmentation; runtime metrics không truyền số ký tự kỳ vọng.",
            "Tên biển từ filename chưa được kiểm tra thủ công toàn bộ; test split frame có thể gồm nhiều ảnh cùng một danh tính, đã nhóm để không rò sang train/val.",
            "INT8 memory and 50MHz timing are theoretical only; no quantization export or FPGA synthesis/STA was performed.",
        ],
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_confusion(args.metrics.parent / "two_row_test_reference_before_confusion.csv", confusion_matrices["phase1_dense_before_two_row"]["two_row_test_reference"])
    save_confusion(args.metrics.parent / "two_row_test_reference_after_confusion.csv", confusion_matrices["dense_after_two_row_finetune"]["two_row_test_reference"])
    with (args.metrics.parent / "two_row_confusion_cost_M.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for char, row in zip(CLASS_NAMES, cost_np):
            writer.writerow([char, *[f"{value:.6f}" for value in row]])
    write_class_comparison(
        args.metrics.parent / "two_row_test_per_class_comparison.csv",
        results["phase1_dense_before_two_row"]["two_row_test_reference"],
        results["dense_after_two_row_finetune"]["two_row_test_reference"],
    )
    with (args.metrics.parent / "two_row_dense_training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]) if history else [])
        writer.writeheader()
        writer.writerows(history)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(args.report, metrics)
    print(f"Saved checkpoint={args.output}, metrics={args.metrics}, report={args.report}", flush=True)


if __name__ == "__main__":
    main()
