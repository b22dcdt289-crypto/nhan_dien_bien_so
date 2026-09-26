from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from train_independent_structured import FolderChars, LeNet5Structured50, top_indices
from train_lenet5 import CLASS_NAMES, LeNet5
from train_matched_channel_pruning import model_macs, read_plate_validation_batches


class LeNet5Conv2Pruned(nn.Module):
    """LeNet with the teacher's Conv1 intact and only Conv2 output channels reduced."""

    conv2_channels = 8

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, self.conv2_channels, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.conv2_channels * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_checkpoint(model: nn.Module, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if list(checkpoint.get("classes", [])) != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order does not match the 30-class experiment: {path}")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def compact_teacher_to_student(teacher: LeNet5, student: LeNet5Conv2Pruned) -> list[int]:
    teacher_conv1, teacher_conv2 = teacher.features[0], teacher.features[3]
    teacher_fc1, teacher_fc2, teacher_fc3 = (
        teacher.classifier[0], teacher.classifier[2], teacher.classifier[4]
    )
    student_conv1, student_conv2 = student.features[0], student.features[3]
    student_fc1, student_fc2, student_fc3 = (
        student.classifier[0], student.classifier[2], student.classifier[4]
    )
    selected = top_indices(teacher_conv2.weight, student.conv2_channels).tolist()
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=teacher_conv2.weight.device)
    spatial_columns = (
        selected_tensor[:, None] * 25 + torch.arange(25, device=selected_tensor.device)[None, :]
    ).reshape(-1)
    with torch.no_grad():
        student_conv1.weight.copy_(teacher_conv1.weight)
        student_conv1.bias.copy_(teacher_conv1.bias)
        student_conv2.weight.copy_(teacher_conv2.weight[selected_tensor])
        student_conv2.bias.copy_(teacher_conv2.bias[selected_tensor])
        student_fc1.weight.copy_(teacher_fc1.weight[:, spatial_columns])
        student_fc1.bias.copy_(teacher_fc1.bias)
        student_fc2.weight.copy_(teacher_fc2.weight)
        student_fc2.bias.copy_(teacher_fc2.bias)
        student_fc3.weight.copy_(teacher_fc3.weight)
        student_fc3.bias.copy_(teacher_fc3.bias)
    student_conv1.weight.requires_grad_(False)
    student_conv1.bias.requires_grad_(False)
    if not torch.equal(student_conv1.weight, teacher_conv1.weight):
        raise AssertionError("Conv1 weights must be copied exactly from teacher.")
    if not torch.equal(student_conv1.bias, teacher_conv1.bias):
        raise AssertionError("Conv1 bias must be copied exactly from teacher.")
    return selected


def blur_bucket(value: float) -> str:
    if value < 80:
        return "<80"
    if value < 200:
        return "80–199"
    if value < 500:
        return "200–499"
    return "≥500"


def angle_bucket(value: float) -> str:
    value = abs(value)
    if value <= 2:
        return "0–2°"
    if value <= 5:
        return ">2–5°"
    if value <= 8:
        return ">5–8°"
    return ">8°"


def analyze_model(model: nn.Module, samples: list, detail_rows: list[dict] | None = None) -> dict:
    model.eval()
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    by_type = defaultdict(lambda: {"plates": 0, "wrong_plates": 0, "characters": 0, "correct_characters": 0, "character_errors": 0})
    by_length = defaultdict(lambda: {"plates": 0, "wrong_plates": 0, "characters": 0, "correct_characters": 0, "character_errors": 0})
    by_position = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    by_blur = defaultdict(lambda: {"plates": 0, "wrong_plates": 0})
    by_angle = defaultdict(lambda: {"plates": 0, "wrong_plates": 0})
    by_perspective = defaultdict(lambda: {"plates": 0, "wrong_plates": 0})
    error_count_hist = Counter()
    substitution_pairs = Counter()
    wrong_prediction_confidences = []
    correct_prediction_confidences = []
    total_characters = correct_characters = exact_plates = 0
    length_mismatch_plates = 0
    with torch.inference_mode():
        for record, inputs in samples:
            logits = model(inputs)
            probabilities = logits.softmax(dim=1)
            confidence, predicted_ids = probabilities.max(dim=1)
            predictions = [CLASS_NAMES[index] for index in predicted_ids.cpu().tolist()]
            predicted_text = "".join(predictions)
            target = record["text"]
            mismatch_positions = [
                index for index, (expected, actual) in enumerate(zip(target, predicted_text), start=1)
                if expected != actual
            ]
            length_mismatch = len(target) != len(predicted_text)
            if length_mismatch:
                length_mismatch_plates += 1
            mismatch_count = len(mismatch_positions) + abs(len(target) - len(predicted_text))
            exact = mismatch_count == 0
            exact_plates += int(exact)
            error_count_hist[str(mismatch_count) if mismatch_count < 3 else "3+"] += 1
            total_characters += len(target)
            correct_characters += len(target) - mismatch_count
            type_stats = by_type[record["type"]]
            type_stats["plates"] += 1
            type_stats["wrong_plates"] += int(not exact)
            type_stats["characters"] += len(target)
            type_stats["correct_characters"] += len(target) - mismatch_count
            type_stats["character_errors"] += mismatch_count
            length_stats = by_length[str(len(target))]
            length_stats["plates"] += 1
            length_stats["wrong_plates"] += int(not exact)
            length_stats["characters"] += len(target)
            length_stats["correct_characters"] += len(target) - mismatch_count
            length_stats["character_errors"] += mismatch_count

            for position, (expected, actual) in enumerate(zip(target, predicted_text), start=1):
                expected_id = CLASS_NAMES.index(expected)
                actual_id = CLASS_NAMES.index(actual)
                confusion[expected_id, actual_id] += 1
                slot = by_position[str(position)]
                slot["total"] += 1
                slot["correct"] += int(expected == actual)
                slot["errors"] += int(expected != actual)
                predicted_conf = float(confidence[position - 1])
                if expected == actual:
                    correct_prediction_confidences.append(predicted_conf)
                else:
                    wrong_prediction_confidences.append(predicted_conf)
                    substitution_pairs[f"{expected}→{actual}"] += 1

            for key, bucket, store in (
                ("blur_score", blur_bucket(float(record.get("blur_score", 0.0))), by_blur),
                ("angle_deg", angle_bucket(float(record.get("angle_deg", 0.0))), by_angle),
                ("perspective_corrected", str(bool(record.get("perspective_corrected", False))), by_perspective),
            ):
                _ = key
                store[bucket]["plates"] += 1
                store[bucket]["wrong_plates"] += int(not exact)

            if not exact and detail_rows is not None:
                wrong_conf = [
                    float(confidence[pos - 1])
                    for pos in mismatch_positions
                    if pos - 1 < len(confidence)
                ]
                substitutions = [
                    f"{target[pos - 1]}>{predicted_text[pos - 1]}"
                    for pos in mismatch_positions
                ]
                detail_rows.append({
                    "model": getattr(model, "_report_name", "model"),
                    "source_local_only": record["source"],
                    "ground_truth_local_only": target,
                    "prediction_local_only": predicted_text,
                    "plate_type": record["type"],
                    "label_length": len(target),
                    "wrong_characters": mismatch_count,
                    "error_positions_1based": ",".join(str(pos) for pos in mismatch_positions),
                    "substitutions": ",".join(substitutions),
                    "mean_wrong_prediction_confidence": float(np.mean(wrong_conf)) if wrong_conf else "",
                    "blur_score": record.get("blur_score", ""),
                    "angle_degrees": record.get("angle_deg", ""),
                    "perspective_corrected": record.get("perspective_corrected", ""),
                    "row_count": record.get("row_count", ""),
                })

    def finalize_groups(groups):
        return {
            key: {
                **values,
                "exact_plate_accuracy": 1 - values["wrong_plates"] / max(1, values["plates"]),
                "plate_error_rate": values["wrong_plates"] / max(1, values["plates"]),
                **(
                    {"character_accuracy": values["correct_characters"] / max(1, values["characters"])}
                    if "characters" in values else {}
                ),
            }
            for key, values in sorted(groups.items())
        }

    per_character = []
    for index, character in enumerate(CLASS_NAMES):
        support = int(confusion[index].sum())
        right = int(confusion[index, index])
        per_character.append({
            "character": character,
            "support": support,
            "correct": right,
            "accuracy": right / max(1, support),
        })

    return {
        "plates": len(samples),
        "characters": total_characters,
        "correct_characters": correct_characters,
        "character_accuracy": correct_characters / max(1, total_characters),
        "exact_plates": exact_plates,
        "wrong_plates": len(samples) - exact_plates,
        "exact_plate_accuracy": exact_plates / max(1, len(samples)),
        "length_mismatch_plates": length_mismatch_plates,
        "error_count_histogram": dict(sorted(error_count_hist.items())),
        "by_plate_type": finalize_groups(by_type),
        "by_label_length": finalize_groups(by_length),
        "by_character_position": {
            key: {**values, "accuracy": values["correct"] / max(1, values["total"])}
            for key, values in sorted(by_position.items(), key=lambda pair: int(pair[0]))
        },
        "by_blur_bucket": finalize_groups(by_blur),
        "by_angle_bucket": finalize_groups(by_angle),
        "by_perspective_corrected": finalize_groups(by_perspective),
        "top_substitution_pairs": [
            {"pair": pair, "count": count}
            for pair, count in substitution_pairs.most_common(20)
        ],
        "mean_confidence_correct_characters": float(np.mean(correct_prediction_confidences)) if correct_prediction_confidences else None,
        "mean_confidence_wrong_predictions": float(np.mean(wrong_prediction_confidences)) if wrong_prediction_confidences else None,
        "per_character": per_character,
        "confusion_matrix": confusion.tolist(),
    }


def class_cost_weights(train_set: FolderChars) -> tuple[torch.Tensor, list[dict]]:
    labels = torch.tensor([label for _, label in train_set.items], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
    raw = torch.sqrt(counts.max() / counts.clamp_min(1))
    weights = (raw / raw.mean()).clamp(0.5, 3.0)
    report = [
        {"character": char, "train_count": int(counts[index]), "loss_weight": float(weights[index])}
        for index, char in enumerate(CLASS_NAMES)
    ]
    return weights, report


def cost_sensitive_ohem(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    hard_fraction: float,
    hard_weight: float,
) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    hard_count = max(1, math.ceil(len(labels) * hard_fraction))
    hardest = per_sample.topk(hard_count).values.mean()
    return (1.0 - hard_weight) * per_sample.mean() + hard_weight * hardest


def evaluate_character_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = correct = 0
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += len(labels)
    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total), "correct": correct, "total": total}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: torch.Tensor,
    hard_fraction: float,
    hard_weight: float,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.0,
    temperature: float = 2.0,
) -> dict:
    model.train()
    if teacher is not None:
        teacher.eval()
    weighted_sum = total_correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        supervised = cost_sensitive_ohem(logits, labels, weights, hard_fraction, hard_weight)
        if teacher is not None and kd_alpha > 0:
            with torch.no_grad():
                teacher_logits = teacher(images)
            kd = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = (1.0 - kd_alpha) * supervised + kd_alpha * kd
        else:
            kd = torch.zeros((), device=device)
            loss = supervised
        loss.backward()
        optimizer.step()
        count = len(labels)
        weighted_sum += float(supervised.detach()) * count
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
        total += count
    return {
        "cost_sensitive_ohem_loss": weighted_sum / max(1, total),
        "train_accuracy": total_correct / max(1, total),
    }


def train_stage(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_plate_samples: list,
    device: torch.device,
    weights: torch.Tensor,
    args,
    output_path: Path,
    teacher: nn.Module | None = None,
) -> tuple[nn.Module, list[dict], dict]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict] = []
    best_key = (-1.0, -1.0)
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_result = train_epoch(
            model, train_loader, optimizer, device, weights, args.hard_fraction,
            args.hard_weight, teacher=teacher, kd_alpha=args.kd_alpha if teacher else 0.0,
            temperature=args.temperature,
        )
        crop_result = evaluate_character_loader(model, val_loader, device)
        plate_result = analyze_model(model, val_plate_samples)
        key = (plate_result["exact_plate_accuracy"], plate_result["character_accuracy"])
        row = {
            "stage": name,
            "epoch": epoch,
            **train_result,
            "validation_crop_loss_unweighted": crop_result["loss"],
            "validation_crop_accuracy": crop_result["accuracy"],
            "validation_plate_character_accuracy": plate_result["character_accuracy"],
            "validation_exact_plate_accuracy": plate_result["exact_plate_accuracy"],
            "epoch_seconds": time.perf_counter() - start,
            "selected_as_best": key > best_key,
        }
        history.append(row)
        print(
            f"{name} epoch={epoch}/{args.epochs} "
            f"train_acc={row['train_accuracy']:.4%} val_char={row['validation_plate_character_accuracy']:.4%} "
            f"val_exact={row['validation_exact_plate_accuracy']:.4%} "
            f"loss={row['cost_sensitive_ohem_loss']:.4f} epoch_s={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_metrics = plate_result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "classes": CLASS_NAMES,
                "arch": name,
                "channels": (
                    [6, 16, 120, 84] if isinstance(model, LeNet5)
                    else [6, LeNet5Conv2Pruned.conv2_channels, 120, 84]
                ),
                "epoch": epoch,
                "val_acc": crop_result["accuracy"],
                "val_plate_character_accuracy": plate_result["character_accuracy"],
                "val_exact_plate_accuracy": plate_result["exact_plate_accuracy"],
                "training_data": str(args.data),
                "loss": "class-cost inverse-sqrt frequency + batch OHEM" + (" + teacher KL distillation" if teacher else ""),
                "seed": args.seed,
            }, output_path)
    if best_metrics is None:
        raise RuntimeError(f"No best checkpoint recorded in stage {name}")
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, history, best_metrics


def benchmark_models(models: dict[str, nn.Module], args, device: torch.device) -> tuple[dict, list[dict]]:
    torch.set_num_threads(1)
    all_rows = []
    summaries = {}
    rng = random.Random(args.seed + 117)
    for batch_size in (1, 7, 8, 9, 10):
        inputs = torch.rand((batch_size, 1, 32, 32), device=device)
        for name, model in models.items():
            model.eval()
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(inputs)
        samples_by_model = {name: [] for name in models}
        names = list(models)
        for _ in range(args.repeats):
            rng.shuffle(names)
            for name in names:
                model = models[name]
                model.eval()
                with torch.inference_mode():
                    start = time.perf_counter_ns()
                    model(inputs)
                    samples_by_model[name].append((time.perf_counter_ns() - start) / 1e6)
        for name, samples_ms in samples_by_model.items():
            summary = {
                "iterations": args.repeats,
                "median_ms_per_forward": float(np.percentile(samples_ms, 50)),
                "p95_ms_per_forward": float(np.percentile(samples_ms, 95)),
                "mean_ms_per_forward": float(statistics.fmean(samples_ms)),
                "characters_per_second_from_median": batch_size / (float(np.percentile(samples_ms, 50)) / 1000.0),
            }
            summaries.setdefault(name, {})[str(batch_size)] = summary
            for iteration, latency in enumerate(samples_ms, start=1):
                all_rows.append({
                    "model": name,
                    "batch_characters": batch_size,
                    "iteration": iteration,
                    "latency_ms": latency,
                })
            print(
                f"benchmark {name} batch={batch_size} p50={summary['median_ms_per_forward']:.5f}ms "
                f"p95={summary['p95_ms_per_forward']:.5f}ms",
                flush=True,
            )
    dense_name = "dense_previous"
    for name, batch_results in summaries.items():
        for batch_size, result in batch_results.items():
            dense_p50 = summaries[dense_name][batch_size]["median_ms_per_forward"]
            result["speedup_vs_previous_dense"] = dense_p50 / result["median_ms_per_forward"]
            result["latency_reduction_vs_previous_dense_percent"] = (
                1.0 - result["median_ms_per_forward"] / dense_p50
            ) * 100.0
    return summaries, all_rows


def write_report(path: Path, metrics: dict):
    previous = metrics["models"]["dense_previous"]
    old_small = metrics["models"]["structured50_previous"]
    teacher = metrics["models"]["cost_sensitive_teacher"]
    student = metrics["models"]["conv2_pruned_student"]

    def pct(value):
        return f"{100 * value:.2f}%"

    lines = [
        "# Cost-sensitive hard mining và Conv2-only pruning/distillation",
        "",
        "## Tóm tắt",
        "",
        "Thí nghiệm tiếp tục từ checkpoint dense 30 lớp đã train trên cùng train(1).",
        "Teacher được fine-tune với class-cost inverse-sqrt và hard-example mining;",
        "sau đó student giữ nguyên, đóng băng toàn bộ Conv1, chỉ giảm Conv2 từ 16 xuống 8 kênh",
        "và học từ teacher bằng supervised loss kết hợp KL knowledge distillation.",
        "",
        "| Mô hình | Accuracy ký tự | Khớp nguyên biển | MAC/ký tự | Tham số |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, model_metrics in metrics["models"].items():
        validation = model_metrics["validation"]
        lines.append(
            f"| {name} | {pct(validation['character_accuracy'])} | "
            f"{pct(validation['exact_plate_accuracy'])} | {model_metrics['macs_per_character']:,} | "
            f"{model_metrics['parameters']:,} |"
        )
    lines += [
        "",
        "### Exact-plate theo nhóm validation",
        "",
        "| Mô hình | Một hàng | Ô tô hai hàng | Xe máy hai hàng |",
        "|---|---:|---:|---:|",
    ]
    for name, model_metrics in metrics["models"].items():
        groups = model_metrics["validation"]["by_plate_type"]
        lines.append(
            f"| {name} | {pct(groups['one_row']['exact_plate_accuracy'])} | "
            f"{pct(groups['two_row_car']['exact_plate_accuracy'])} | "
            f"{pct(groups['two_row_motorcycle']['exact_plate_accuracy'])} |"
        )
    teacher_seconds = sum(row["epoch_seconds"] for row in metrics["training_history"] if row["stage"] == "cost_sensitive_teacher")
    student_seconds = sum(row["epoch_seconds"] for row in metrics["training_history"] if row["stage"] == "conv2_pruned_student")
    lines += [
        "",
        "## Thiết lập train",
        "",
        f"Nguồn train(1); train/validation có {metrics['data']['train_character_crops']:,}/"
        f"{metrics['data']['validation_character_crops']:,} crop ký tự và {metrics['data']['validation_plates']:,} biển validation.",
        f"Teacher fine-tune {metrics['training']['teacher_finetune_epochs']} epoch từ checkpoint dense cũ; student distill "
        f"{metrics['training']['student_distill_epochs']} epoch. Batch={metrics['training']['batch_size']}, "
        f"AdamW lr={metrics['training']['learning_rate']}, weight_decay={metrics['training']['weight_decay']}, "
        f"seed={metrics['training']['seed']}.",
        f"Thời gian epoch tích lũy: teacher {teacher_seconds:.1f}s; student {student_seconds:.1f}s "
        "(không gồm nạp cache/benchmark).",
    ]
    lines += [
        "",
        "## Phân rã lỗi trên validation trước tối ưu",
        "",
        "Mỗi sample là một biển; OCR chỉ phân loại các crop ký tự đã được tạo trước.",
        "Vì vậy mismatch ký tự được đo độc lập với lỗi phát hiện/segmentation khi chạy tự do.",
        f"Bảng cá nhân theo từng biển nằm trong file local-only {Path(metrics['private_error_cases_local_file_not_for_git']).name};",
        "file này có nội dung biển số và không được đẩy lên GitHub.",
        "",
        "| Mô hình cũ | Biển sai | Exact accuracy | Histogram số ký tự sai trên mỗi biển |",
        "|---|---:|---:|---|",
    ]
    for name in ("dense_previous", "structured50_previous"):
        error = metrics["baseline_error_breakdown"][name]
        lines.append(
            f"| {name} | {error['wrong_plates']}/{error['plates']} | "
            f"{pct(error['exact_plate_accuracy'])} | {error['error_count_histogram']} |"
        )
    lines += ["", "Chi tiết theo nhóm (số biển, số biển lỗi, error rate):", ""]
    for name in ("dense_previous", "structured50_previous"):
        error = metrics["baseline_error_breakdown"][name]
        lines += [
            f"### {name}: lỗi theo nhóm",
            "",
            "| Nhóm | Biển | Sai biển | Tỷ lệ sai |",
            "|---|---:|---:|---:|",
        ]
        for kind, groups in (("Loại biển", error["by_plate_type"]), ("Độ dài nhãn", error["by_label_length"])):
            for group, values in groups.items():
                lines.append(
                    f"| {kind}: {group} | {values['plates']} | {values['wrong_plates']} | "
                    f"{pct(values['plate_error_rate'])} |"
                )
        lines += [
            "",
            f"- Confidence trung bình ở ký tự dự đoán đúng: {error['mean_confidence_correct_characters']:.4f}.",
            f"- Confidence trung bình ở ký tự dự đoán sai: {error['mean_confidence_wrong_predictions']:.4f}.",
            f"- Cặp nhầm ký tự thường gặp nhất: {error['top_substitution_pairs'][:10]}.",
            "- Tỷ lệ lỗi theo blur, góc nghiêng, perspective-correction và từng vị trí có trong JSON metrics.",
            "",
        ]
    for field, title in (
        ("by_blur_bucket", "Blur bucket"),
        ("by_angle_bucket", "Góc ước lượng bucket"),
        ("by_perspective_corrected", "Perspective correction"),
    ):
        dense_groups = metrics["baseline_error_breakdown"]["dense_previous"][field]
        structured_groups = metrics["baseline_error_breakdown"]["structured50_previous"][field]
        lines += [
            f"### Error theo {title.lower()}",
            "",
            "| Nhóm | Biển | Dense lỗi | Dense error rate | Structured lỗi | Structured error rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for group, dense_values in dense_groups.items():
            structured_values = structured_groups[group]
            lines.append(
                f"| {group} | {dense_values['plates']} | {dense_values['wrong_plates']} | "
                f"{pct(dense_values['plate_error_rate'])} | {structured_values['wrong_plates']} | "
                f"{pct(structured_values['plate_error_rate'])} |"
            )
        lines.append("")
    lines += [
        "",
        "Các thống kê blur/góc là tương quan trong validation, không chứng minh nguyên nhân lỗi.",
        "Tên/nhãn biển trong CSV cá nhân chỉ hỗ trợ người dùng kiểm tra lại ảnh và segmentation.",
        "",
        "## Loss và distillation",
        "",
        f"- Cost-sensitive: mỗi lớp có trọng số tỷ lệ nghịch căn bậc hai tần suất train,",
        f"  chuẩn hóa về trung bình gần 1 và giới hạn [0,5; 3,0].",
        f"- Hard mining: trong mỗi batch lấy top {metrics['training']['hard_fraction']:.0%} mẫu có weighted CE cao nhất;",
        f"  loss = {1-metrics['training']['hard_weight']:.2f}×mean(all) + {metrics['training']['hard_weight']:.2f}×mean(hardest).",
        f"- Student objective: {1-metrics['training']['kd_alpha']:.2f}×cost-sensitive OHEM +",
        f"  {metrics['training']['kd_alpha']:.2f}×T² KL(student/T, teacher/T), T={metrics['training']['temperature']}.",
        f"- Conv1 weights/bias được copy chính xác từ teacher và frozen; Conv2 output được rút 16→8;",
        "  downstream FC1 input columns được rút tương ứng để tensor thực sự nhỏ lại.",
        "",
        "## MAC và tài nguyên",
        "",
        "| Mô hình | Kênh (Conv1, Conv2, FC1, FC2) | MAC/ký tự | Thay đổi MAC vs dense | Tham số | FP32 weights thô |",
        "|---|---|---:|---:|---:|---:|",
    ]
    dense_macs = previous["macs_per_character"]
    for name, model_metrics in metrics["models"].items():
        fp32_kib = model_metrics["parameters"] * 4 / 1024
        mac_delta = (model_metrics["macs_per_character"] / dense_macs - 1) * 100
        channels = ", ".join(str(value) for value in model_metrics["channels"])
        lines.append(
            f"| {name} | ({channels}) | {model_metrics['macs_per_character']:,} | "
            f"{mac_delta:+.2f}% | {model_metrics['parameters']:,} | {fp32_kib:.2f} KiB |"
        )
    lines += [
        "",
        "### MAC cho toàn chuỗi sau segmentation",
        "",
        "| Số ký tự | Dense cũ | Structured-50 cũ | Teacher | Conv2 student |",
        "|---:|---:|---:|---:|---:|",
    ]
    for length in (7, 8, 9, 10):
        lines.append(
            f"| {length} | "
            + " | ".join(f"{metrics['models'][name]['macs_per_character'] * length:,}" for name in metrics["models"])
            + " |"
        )
    student_vs_dense = (1 - student["macs_per_character"] / dense_macs) * 100
    student_vs_small = (student["macs_per_character"] / old_small["macs_per_character"] - 1) * 100
    lines += [
        "",
        f"Conv2-only student giảm MAC {student_vs_dense:.2f}% so với dense cũ; do giữ nguyên Conv1 và FC hidden widths, "
        f"nó có MAC cao hơn {student_vs_small:.2f}% so với structured-50 cũ.",
        "Công thức student mỗi ký tự: Conv1 6×25×28×28 = 117.600; Conv2 8×6×25×10×10 = 120.000; "
        "FC1 8×25×120 = 24.000; FC2 120×84 = 10.080; FC3 84×30 = 2.520; tổng = 274.200 MAC.",
        "",
        "## Latency CPU",
        "",
        "| Batch ký tự | Dense cũ p50 ms | Structured-50 cũ p50 ms | Teacher p50 ms | Conv2 student p50 ms | Student speedup vs dense | Student speedup vs struct-50 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    speed = metrics["speed_benchmark"]["models"]
    for batch in (1, 7, 8, 9, 10):
        dense_p50 = speed["dense_previous"][str(batch)]["median_ms_per_forward"]
        small_p50 = speed["structured50_previous"][str(batch)]["median_ms_per_forward"]
        student_p50 = speed["conv2_pruned_student"][str(batch)]["median_ms_per_forward"]
        lines.append(
            f"| {batch} | {dense_p50:.5f} | {small_p50:.5f} | "
            f"{speed['cost_sensitive_teacher'][str(batch)]['median_ms_per_forward']:.5f} | "
            f"{student_p50:.5f} | {dense_p50 / student_p50:.3f}× | {small_p50 / student_p50:.3f}× |"
        )
    batch8_dense = speed["dense_previous"]["8"]
    batch8_small = speed["structured50_previous"]["8"]
    batch8_student = speed["conv2_pruned_student"]["8"]
    lines += [
        "",
        f"Batch 8: student latency giảm {(1 - batch8_student['median_ms_per_forward'] / batch8_dense['median_ms_per_forward']) * 100:.2f}% "
        f"so với dense cũ; p95 của dense/student là {batch8_dense['p95_ms_per_forward']:.5f}/"
        f"{batch8_student['p95_ms_per_forward']:.5f} ms.",
        f"MAC ratio dense/student = {dense_macs / student['macs_per_character']:.3f}× nếu workload compute-bound; "
        f"đây chỉ là tỷ lệ phép toán, không phải dự báo timing FPGA. Student có p50 latency cao hơn structured-50 cũ "
        f"{(batch8_student['median_ms_per_forward'] / batch8_small['median_ms_per_forward'] - 1) * 100:.2f}% ở batch 8.",
        "Benchmark dùng 250 lượt warm-up và 2.500 lượt đo mỗi mô hình/batch; thứ tự mô hình được xáo trộn trong từng vòng đo.",
    ]
    lines += [
        "",
        "Forward latency đo PyTorch CPU, 1 thread, 250 warm-up + 2.500 lượt đo, thứ tự model xáo trộn,",
        "cùng tensor N×1×32×32; không gồm đọc ảnh,",
        "perspective correction, phân hàng, segmentation hoặc giải mã. Speedup thực tế phụ thuộc runtime.",
        "",
        "## Giới hạn",
        "",
        "- Validation được chia theo frame chứ không group theo identity; frame gần trùng có thể xuất hiện ở hai split.",
        "- Validation dùng crop và số crop theo nhãn tên file; không phải full-frame end-to-end.",
        "- Cost-sensitive OHEM làm thay đổi objective và có thể cải thiện lớp hiếm nhưng làm giảm accuracy tổng thể;",
        "  cần đọc cả exact-plate và per-class metrics trước khi chọn model.",
        "- CPU benchmark không dự báo trực tiếp DE10-Lite. Cần tổng hợp hai phiên bản trên FPGA",
        "  rồi so ALM/DSP/RAM/fMAX, cycles và timing slack.",
        "",
        "Metrics đầy đủ, class weights, histories, per-character confusion và số liệu benchmark nằm cạnh báo cáo.",
        "Bảng per-plate có tên/nhãn được giữ local-only; aggregate error breakdown không chứa định danh biển.",
        "",
        "## Reproduce",
        "",
        "Lệnh dưới đây cần data đã chuẩn bị và dùng tên output mới vì trainer không ghi đè checkpoint:",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe train_cost_sensitive_conv2_distill.py --epochs 8 --teacher-output artifacts/repro_cost_teacher.pt --student-output artifacts/repro_conv2_student.pt --metrics artifacts/repro_metrics.json --history-csv artifacts/repro_history.csv --report artifacts/repro_report.md --error-private-csv artifacts/repro_local_error_plates.csv --error-aggregate artifacts/repro_error_aggregate.json --speed-json artifacts/repro_speed.json --speed-raw artifacts/repro_speed_raw.csv --student-confusion artifacts/repro_confusion.csv",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark_only(args):
    checkpoint_paths = (
        args.baseline_dense,
        args.baseline_structured,
        args.teacher_output,
        args.student_output,
        args.metrics,
    )
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Benchmark-only mode requires existing checkpoints/metrics: " + ", ".join(missing))
    device = torch.device("cpu")
    dense = LeNet5(len(CLASS_NAMES)).to(device)
    load_checkpoint(dense, args.baseline_dense, device)
    structured = LeNet5Structured50(len(CLASS_NAMES)).to(device)
    load_checkpoint(structured, args.baseline_structured, device)
    teacher = LeNet5(len(CLASS_NAMES)).to(device)
    load_checkpoint(teacher, args.teacher_output, device)
    student = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
    load_checkpoint(student, args.student_output, device)
    models = {
        "dense_previous": dense,
        "structured50_previous": structured,
        "cost_sensitive_teacher": teacher,
        "conv2_pruned_student": student,
    }
    for model in models.values():
        model.eval()
    print("stage=interleaved_cpu_latency_benchmark", flush=True)
    summaries, raw = benchmark_models(models, args, device)
    speed = {
        "method": "Interleaved randomized-order PyTorch CPU forward-only benchmark; random N×1×32×32 tensors",
        "cpu": platform.processor(),
        "torch_version": torch.__version__,
        "threads": 1,
        "warmup_forwards_per_model_and_batch": args.warmup,
        "measured_forwards_per_model_and_batch": args.repeats,
        "batch_sizes_characters": [1, 7, 8, 9, 10],
        "measurement_order": "Within each repeat, model order is shuffled; all four models are measured once before the next repeat.",
        "models": summaries,
    }
    args.speed_json.parent.mkdir(parents=True, exist_ok=True)
    args.speed_json.write_text(json.dumps(speed, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.speed_raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "batch_characters", "iteration", "latency_ms"])
        writer.writeheader()
        writer.writerows(raw)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics["speed_benchmark"] = speed
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, metrics)
    print(json.dumps({
        name: {batch: result["median_ms_per_forward"] for batch, result in batches.items()}
        for name, batches in summaries.items()
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"updated_metrics={args.metrics} report={args.report} speed={args.speed_json}", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Cost-sensitive/OHEM teacher fine-tuning, Conv2-only channel pruning, and KD.")
    parser.add_argument("--data", type=Path, default=Path("data/independent_chars_train1_structured_channel30_v1"))
    parser.add_argument("--baseline-dense", type=Path, default=Path("artifacts/lenet5_dense_channel30_matched_v1.pt"))
    parser.add_argument("--baseline-structured", type=Path, default=Path("artifacts/lenet5_structured50_channel30_matched_v1.pt"))
    parser.add_argument("--teacher-output", type=Path, default=Path("artifacts/lenet5_cost_hard_channel30_teacher_v1.pt"))
    parser.add_argument("--student-output", type=Path, default=Path("artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_distill_metrics.json"))
    parser.add_argument("--history-csv", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_distill_history.csv"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_distill_report.md"))
    parser.add_argument("--error-private-csv", type=Path, default=Path("artifacts/channel30_error_cases_local.csv"))
    parser.add_argument("--error-aggregate", type=Path, default=Path("artifacts/channel30_error_breakdown_before.json"))
    parser.add_argument("--reuse-error-analysis", action="store_true", help="Reuse an already computed local baseline analysis.")
    parser.add_argument("--speed-json", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_speed.json"))
    parser.add_argument("--speed-raw", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_speed_raw.csv"))
    parser.add_argument("--student-confusion", type=Path, default=Path("artifacts/channel30_conv2_student_confusion.csv"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--kd-alpha", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=2500)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--benchmark-only", action="store_true", help="Re-measure the existing checkpoints in interleaved order and refresh report metrics.")
    args = parser.parse_args()

    if args.benchmark_only:
        run_benchmark_only(args)
        return

    output_paths = [
        args.teacher_output, args.student_output, args.metrics, args.history_csv,
        args.report, args.speed_json, args.speed_raw, args.student_confusion,
    ]
    if not args.reuse_error_analysis:
        output_paths += [args.error_private_csv, args.error_aggregate]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite outputs: " + ", ".join(existing))
    if args.reuse_error_analysis and (not args.error_private_csv.exists() or not args.error_aggregate.exists()):
        raise FileNotFoundError("--reuse-error-analysis requires both existing local error-analysis files.")
    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared train(1) manifest not found: {manifest_path}")
    for source_path in (args.baseline_dense, args.baseline_structured):
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    train_set = FolderChars(args.data, "train", augment=True)
    val_set = FolderChars(args.data, "val", augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_records = [record for record in manifest if record["split"] == "val"]
    val_plate_samples = read_plate_validation_batches(args.data, val_records)
    weights, weight_report = class_cost_weights(train_set)
    weights = weights.to(device)

    dense_previous = LeNet5(len(CLASS_NAMES)).to(device)
    dense_ckpt = load_checkpoint(dense_previous, args.baseline_dense, device)
    structured_previous = LeNet5Structured50(len(CLASS_NAMES)).to(device)
    structured_ckpt = load_checkpoint(structured_previous, args.baseline_structured, device)

    if args.reuse_error_analysis:
        print("stage=baseline_error_analysis (reusing previously computed files)", flush=True)
        baseline_errors = json.loads(args.error_aggregate.read_text(encoding="utf-8"))
    else:
        private_error_rows: list[dict] = []
        dense_previous._report_name = "dense_previous"
        structured_previous._report_name = "structured50_previous"
        print("stage=baseline_error_analysis", flush=True)
        baseline_errors = {
            "dense_previous": analyze_model(dense_previous, val_plate_samples, private_error_rows),
            "structured50_previous": analyze_model(structured_previous, val_plate_samples, private_error_rows),
        }
        args.error_private_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.error_private_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            fields = list(private_error_rows[0]) if private_error_rows else ["model", "source_local_only", "ground_truth_local_only", "prediction_local_only"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(private_error_rows)
        args.error_aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.error_aggregate.write_text(json.dumps(baseline_errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "baseline errors: " + json.dumps({
            name: {
                "wrong_plates": values["wrong_plates"],
                "exact_plate_accuracy": values["exact_plate_accuracy"],
                "character_accuracy": values["character_accuracy"],
                "top_substitutions": values["top_substitution_pairs"][:5],
            }
            for name, values in baseline_errors.items()
        }, ensure_ascii=False),
        flush=True,
    )

    teacher = LeNet5(len(CLASS_NAMES)).to(device)
    teacher.load_state_dict(dense_previous.state_dict())
    print("stage=cost_sensitive_ohem_teacher_finetune", flush=True)
    teacher, teacher_history, teacher_validation = train_stage(
        "cost_sensitive_teacher", teacher, train_loader, val_loader, val_plate_samples,
        device, weights, args, args.teacher_output,
    )

    student = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
    selected_conv2 = compact_teacher_to_student(teacher, student)
    teacher.eval()
    print(
        f"stage=conv2_only_prune_and_distill selected_teacher_conv2_channels={selected_conv2} "
        f"student_channels={(6, LeNet5Conv2Pruned.conv2_channels, 120, 84)}",
        flush=True,
    )
    student, student_history, student_validation = train_stage(
        "conv2_pruned_student", student, train_loader, val_loader, val_plate_samples,
        device, weights, args, args.student_output, teacher=teacher,
    )
    if student.features[0].weight.requires_grad or student.features[0].bias.requires_grad:
        raise AssertionError("Student Conv1 unexpectedly trainable.")
    teacher_ckpt = torch.load(args.teacher_output, map_location=device, weights_only=False)
    teacher_best = LeNet5(len(CLASS_NAMES)).to(device)
    teacher_best.load_state_dict(teacher_ckpt["model"])
    if not torch.equal(student.features[0].weight, teacher_best.features[0].weight):
        raise AssertionError("Final student Conv1 differs from its teacher.")
    if not torch.equal(student.features[0].bias, teacher_best.features[0].bias):
        raise AssertionError("Final student Conv1 bias differs from its teacher.")

    final_models = {
        "dense_previous": dense_previous,
        "structured50_previous": structured_previous,
        "cost_sensitive_teacher": teacher,
        "conv2_pruned_student": student,
    }
    validations = {}
    for name, model in final_models.items():
        model._report_name = name
        validations[name] = analyze_model(model, val_plate_samples)

    # Include the independently trained 50%-channel baseline in the loss/error table.
    model_specs = {
        "dense_previous": {
            "channels": [6, 16, 120, 84],
            "parameters": count_parameters(dense_previous),
            "macs_per_character": model_macs((6, 16, 120, 84), len(CLASS_NAMES)),
            "validation": validations["dense_previous"],
            "checkpoint": args.baseline_dense.as_posix(),
        },
        "structured50_previous": {
            "channels": list(LeNet5Structured50.channels),
            "parameters": count_parameters(structured_previous),
            "macs_per_character": model_macs(LeNet5Structured50.channels, len(CLASS_NAMES)),
            "validation": validations["structured50_previous"],
            "checkpoint": args.baseline_structured.as_posix(),
        },
        "cost_sensitive_teacher": {
            "channels": [6, 16, 120, 84],
            "parameters": count_parameters(teacher),
            "macs_per_character": model_macs((6, 16, 120, 84), len(CLASS_NAMES)),
            "validation": validations["cost_sensitive_teacher"],
            "checkpoint": args.teacher_output.as_posix(),
        },
        "conv2_pruned_student": {
            "channels": [6, LeNet5Conv2Pruned.conv2_channels, 120, 84],
            "parameters": count_parameters(student),
            "macs_per_character": model_macs((6, LeNet5Conv2Pruned.conv2_channels, 120, 84), len(CLASS_NAMES)),
            "validation": validations["conv2_pruned_student"],
            "checkpoint": args.student_output.as_posix(),
            "conv1_frozen_identical_to_teacher": True,
            "conv2_selected_teacher_filter_indices": selected_conv2,
        },
    }
    print("stage=cpu_latency_benchmark", flush=True)
    speed_summaries, speed_raw = benchmark_models(final_models, args, device)
    speed_metrics = {
        "method": "PyTorch CPU forward-only with random 32x32 grayscale character tensors",
        "cpu": platform.processor(),
        "torch_version": torch.__version__,
        "threads": 1,
        "warmup_forwards_per_model_and_batch": args.warmup,
        "measured_forwards_per_model_and_batch": args.repeats,
        "batch_sizes_characters": [1, 7, 8, 9, 10],
        "models": speed_summaries,
    }
    args.speed_json.parent.mkdir(parents=True, exist_ok=True)
    args.speed_json.write_text(json.dumps(speed_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.speed_raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "batch_characters", "iteration", "latency_ms"])
        writer.writeheader()
        writer.writerows(speed_raw)

    with args.history_csv.open("w", newline="", encoding="utf-8") as handle:
        history = teacher_history + student_history
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with args.student_confusion.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for char, row in zip(CLASS_NAMES, validations["conv2_pruned_student"]["confusion_matrix"]):
            writer.writerow([char, *row])

    metrics = {
        "experiment": "train(1) 30-class cost-sensitive/OHEM teacher fine-tuning, Conv1-frozen Conv2-only structured pruning, teacher distillation",
        "data": {
            "source": "data/OCR/OCR/images/train(1)/detection",
            "prepared_data": args.data.as_posix(),
            "train_character_crops": len(train_set),
            "validation_character_crops": len(val_set),
            "validation_plates": len(val_plate_samples),
            "split_method": "same frame-level split as prior 30-class channel experiment",
            "class_order": CLASS_NAMES,
        },
        "training": {
            "initialized_teacher_from": args.baseline_dense.as_posix(),
            "teacher_finetune_epochs": args.epochs,
            "student_distill_epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "cost_sensitive_formula": "class weight proportional to inverse square-root of train frequency; normalized then clipped [0.5, 3.0]",
            "hard_fraction": args.hard_fraction,
            "hard_weight": args.hard_weight,
            "hard_mining_formula": "(1-hard_weight)*mean(weighted_CE_all) + hard_weight*mean(top hard_fraction weighted_CE)",
            "kd_alpha": args.kd_alpha,
            "temperature": args.temperature,
            "kd_formula": "(1-alpha)*cost_sensitive_OHEM + alpha*T^2*KL(student/T || teacher/T)",
            "student_conv1_frozen": True,
            "student_conv1_identical_to_teacher": True,
            "student_pruned_layer": "Conv2 output channels only: 16 -> 8; FC1 input columns compacted to match",
            "selected_teacher_conv2_channels": selected_conv2,
            "class_cost_weights": weight_report,
            "seed": args.seed,
        },
        "baseline_error_breakdown": baseline_errors,
        "models": model_specs,
        "training_history": teacher_history + student_history,
        "speed_benchmark": speed_metrics,
        "private_error_cases_local_file_not_for_git": args.error_private_csv.as_posix(),
        "evaluation_caveats": [
            "Validation split is frame-level, not identity grouped; near-duplicate plate frames can cross splits.",
            "OCR is evaluated on accepted character crops with filename-derived target length; no free-running detection or segmentation errors are measured.",
            "Blur/angle associations describe validation subsets and do not establish causal error sources.",
            "CPU model-forward timing is not DE10-Lite/FPGA timing or resource measurement.",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, metrics)
    print(
        json.dumps({
            name: {
                "character_accuracy": model_specs[name]["validation"]["character_accuracy"],
                "exact_plate_accuracy": model_specs[name]["validation"]["exact_plate_accuracy"],
                "macs_per_character": model_specs[name]["macs_per_character"],
                "parameters": model_specs[name]["parameters"],
            }
            for name in model_specs
        }, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"report={args.report} metrics={args.metrics} private_errors={args.error_private_csv}", flush=True)


if __name__ == "__main__":
    main()
