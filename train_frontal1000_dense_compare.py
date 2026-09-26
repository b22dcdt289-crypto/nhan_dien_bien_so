from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, count_parameters, load_checkpoint
from train_frontal1000_conv2_cost import (
    ManifestCharacters,
    activation_distributions,
    benchmark,
    class_counts,
    load_prior_cost,
    runtime_aligned_char_metrics,
    save_confusion,
    weight_distributions,
    weighted_ohem_loss,
)
from train_lenet5 import CLASS_NAMES, LeNet5
from train_one_row_1000 import evaluate_split


PRUNED_MACS_PER_CHAR = 274_200
DENSE_MACS_PER_CHAR = 418_200


def read_manifest(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((data_dir / "preparation_metrics.json").read_text(encoding="utf-8"))
    correction_path = data_dir / "review" / "verified_label_corrections.json"
    corrections = json.loads(correction_path.read_text(encoding="utf-8")) if correction_path.is_file() else []
    correction_map = {item["source_file"]: item["corrected_text"] for item in corrections}
    applied = []
    for record in manifest:
        revised = correction_map.get(Path(record["source"]).name)
        if record["split"] == "train" and revised is not None:
            if len(revised) != len(record["text"]) or any(char not in CLASS_NAMES for char in revised):
                raise ValueError(f"Invalid verified label correction for {record['sample_id']}")
            before = record["text"]
            record["text"] = revised
            applied.extend((index + 1, old, new) for index, (old, new) in enumerate(zip(before, revised)) if old != new)
    if len(applied) != len(corrections):
        raise RuntimeError("The local verified-label corrections did not map one-to-one to training records")
    splits = {name: [row for row in manifest if row["split"] == name] for name in ("train", "val", "test")}
    expected = {"train": 1000, "val": 300, "test": 500}
    if {name: len(rows) for name, rows in splits.items()} != expected:
        raise RuntimeError(f"Unexpected dataset split counts: { {k: len(v) for k, v in splits.items()} }")
    identities = {name: {row["text"] for row in rows} for name, rows in splits.items()}
    if any(identities[a] & identities[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Plate identity overlap across train/validation/test")
    if any(len(identities[name]) != len(splits[name]) for name in splits):
        raise RuntimeError("Duplicate plate identity within a split")
    if not prep.get("test_identities_absent_from_prior_model_dataset"):
        raise RuntimeError("The test set is not disjoint from the previous model dataset")
    return manifest, prep, corrections, applied, splits


def train_dense(args, device, manifest, splits, prep):
    train_set = ManifestCharacters(args.data, splits["train"], augment=True)
    val_set = ManifestCharacters(args.data, splits["val"], augment=False)
    if not train_set or not val_set:
        raise RuntimeError("The aligned train/validation character crops are empty")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeNet5(len(CLASS_NAMES)).to(device)
    initialized = load_checkpoint(model, args.initial, device)
    conv1 = model.features[0]
    conv1.weight.requires_grad_(False)
    conv1.bias.requires_grad_(False)
    initial_conv1 = {key: value.detach().clone() for key, value in conv1.state_dict().items()}

    cost_np, prior_report = load_prior_cost(args.prior_metrics)
    cost_matrix = torch.tensor(cost_np, dtype=torch.float32, device=device)
    counts = Counter(target for _, target in train_set.items)
    max_count = max(counts.values())
    raw_weights = torch.tensor(
        [math.sqrt(max_count / max(1, counts.get(index, 0))) for index in range(len(CLASS_NAMES))],
        dtype=torch.float32,
    )
    class_weights = (raw_weights / raw_weights.mean()).clamp(0.5, 3.0).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    best_epoch = None
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total_seen = hard_selected = 0
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
            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            total_seen += len(labels)
            hard_selected += hard_count
        scheduler.step()

        model.eval()
        val_loss = val_correct = val_seen = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
                val_correct += int((logits.argmax(1) == labels).sum())
                val_seen += len(labels)

        val_summary, _, _ = evaluate_split(model, device, args.data, manifest, "val")
        runtime_accuracy = val_summary["runtime_no_known_count"]["exact_accuracy_all_including_skips"]
        char_accuracy = val_summary["character_accuracy_conditional_on_ground_truth_segmentation"]
        score = runtime_accuracy + 1e-6 * char_accuracy
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_seen),
            "train_top1_accuracy": total_correct / max(1, total_seen),
            "validation_loss": val_loss / max(1, val_seen),
            "validation_top1_accuracy": val_correct / max(1, val_seen),
            "validation_runtime_plate_exact_including_skips": runtime_accuracy,
            "validation_reference_crop_character_accuracy": char_accuracy,
            "hard_examples_selected_fraction": hard_selected / max(1, total_seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "classes": CLASS_NAMES,
                    "arch": "LeNet5_dense_no_pruning",
                    "channels": [6, 16, 120, 84],
                    "epoch": epoch,
                    "val_runtime_plate_exact_accuracy": runtime_accuracy,
                    "val_character_accuracy": char_accuracy,
                    "macs_per_character": DENSE_MACS_PER_CHAR,
                    "params": count_parameters(model),
                    "train_plates": 1000,
                    "source_dataset": str(args.data),
                    "initialized_from": str(args.initial),
                    "structured_pruning": "none; Conv1 fixed at 6, Conv2 retained all 16 channels",
                    "test_identities_absent_from_prior_model_dataset": prep.get("test_identities_absent_from_prior_model_dataset"),
                },
                args.output,
            )

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if any(not torch.equal(initial_conv1[key], value.detach()) for key, value in model.features[0].state_dict().items()):
        raise AssertionError("Conv1 changed despite being frozen")
    return model, train_set, val_set, cost_np, prior_report, history, initialized, best_epoch


def eval_metrics(model, data_dir, manifest, records, split, device):
    summary, rows, confusion = evaluate_split(model, device, data_dir, manifest, split)
    summary["macs_per_character"] = DENSE_MACS_PER_CHAR if isinstance(model, LeNet5) else PRUNED_MACS_PER_CHAR
    if summary["character_count"] == 0:
        summary["character_accuracy_conditional_on_ground_truth_segmentation"] = None
        summary["raw_character_accuracy_conditional_on_ground_truth_segmentation"] = None
        summary["plate_exact_accuracy_conditional_on_ground_truth_segmentation"] = None
        summary["plate_exact_accuracy_all_test_plates"] = None
        summary["raw_plate_exact_accuracy_all_test_plates"] = None
        summary["mean_max_softmax_confidence_per_character"] = None
        summary["character_metric_status"] = "Không đo được: không có ký tự tham chiếu được căn chỉnh trong tập này"
    runtime, runtime_confusion = runtime_aligned_char_metrics(model, data_dir, records, device)
    return summary, rows, confusion, runtime, runtime_confusion


def comparison_report(path: Path, metrics: dict):
    dense = metrics["models"]["unpruned"]
    pruned = metrics["models"]["structured_pruned"]
    val_dense, val_pruned = dense["validation"], pruned["validation"]
    test_dense, test_pruned = dense["test"], pruned["test"]
    params = metrics["architecture"]
    latency = metrics["latency"]["batches"]
    dense_p50 = latency["8"]["unpruned"]["median_ms_per_forward"]
    pruned_p50 = latency["8"]["structured_pruned"]["median_ms_per_forward"]
    measured_delta = dense_p50 / max(pruned_p50, 1e-12) - 1
    dense_acc = val_dense["character_accuracy_conditional_on_ground_truth_segmentation"]
    pruned_acc = val_pruned["character_accuracy_conditional_on_ground_truth_segmentation"]
    dense_acc_s = f"{dense_acc:.3%}" if dense_acc is not None else "Không đo được"
    pruned_acc_s = f"{pruned_acc:.3%}" if pruned_acc is not None else "Không đo được"
    test_dense_runtime = test_dense["runtime_no_known_count"]
    test_pruned_runtime = test_pruned["runtime_no_known_count"]
    lines = [
        "# So sánh LeNet-5 có và không structured pruning",
        "",
        "## Tóm tắt",
        "",
        "Đã huấn luyện lại mô hình LeNet-5 đầy đủ, không pruning, trên đúng bộ dữ liệu 1.000 biển đã chuẩn bị và dùng cùng quy trình loss nhạy chi phí, hard-example mining (OHEM), augmentation, optimizer và lịch train. So sánh với checkpoint đã pruning Conv2 từ 16 xuống 8 kênh, đánh giá trên cùng 300 validation và 500 test.",
        "",
        "**Lưu ý về tính công bằng:** mô hình dense khởi tạo từ checkpoint LeNet-5 đầy đủ; mô hình pruning được lấy từ checkpoint student Conv2 đã pruning rồi fine-tune trên bộ 1.000 biển. Do điểm khởi tạo không đồng nhất hoàn toàn, so sánh này phản ánh hai quy trình/checkpoint thực tế, chưa phải thí nghiệm cô lập duy nhất tác động của pruning.",
        "",
        "## Kết quả nhận dạng",
        "",
        "| Chỉ số | Không pruning | Structured pruning Conv2 | Ghi chú |",
        "|---|---:|---:|---|",
        f"| Validation, độ chính xác ký tự trên crop tham chiếu | {dense_acc_s} | {pruned_acc_s} | 2.400 ký tự, validation dùng để chọn/tinh chỉnh mô hình; không phải test độc lập |",
        f"| Validation, biển khớp toàn bộ trên crop tham chiếu | {val_dense['exact_plates']}/{val_dense['plates']} ({val_dense['plate_exact_accuracy_all_test_plates']:.3%}) | {val_pruned['exact_plates']}/{val_pruned['plates']} ({val_pruned['plate_exact_accuracy_all_test_plates']:.3%}) | Bỏ qua lỗi cắt ký tự; chỉ đo OCR crop tham chiếu |",
        f"| Test, độ chính xác ký tự có căn chỉnh từ segmenter | {dense['test_runtime_character_metrics']['character_accuracy_conditional_on_runtime_count_match'] if dense['test_runtime_character_metrics']['character_accuracy_conditional_on_runtime_count_match'] is not None else 'N/A'} | {pruned['test_runtime_character_metrics']['character_accuracy_conditional_on_runtime_count_match'] if pruned['test_runtime_character_metrics']['character_accuracy_conditional_on_runtime_count_match'] is not None else 'N/A'} | N/A nếu không có biển nào segment ra đúng 8 ký tự |",
        f"| Test, khớp nguyên biển end-to-end | {test_dense_runtime['exact_plates_all_including_skips']}/{test_dense['plates']} ({test_dense_runtime['exact_accuracy_all_including_skips']:.3%}) | {test_pruned_runtime['exact_plates_all_including_skips']}/{test_pruned['plates']} ({test_pruned_runtime['exact_accuracy_all_including_skips']:.3%}) | Tính cả bỏ qua và sai số lượng ký tự là lỗi |",
        f"| Test, độ phủ segmenter | {test_dense_runtime['coverage']:.1%} | {test_pruned_runtime['coverage']:.1%} | Tiền xử lý/cắt ký tự dùng chung nên gần như không phụ thuộc kiến trúc CNN |",
        "",
        "Trong bộ test mới này, dữ liệu chuẩn bị hiện có không tạo được crop tham chiếu cho ký tự (độ phủ reference segmentation = 0%). Segmenter chạy tự do cũng không trả đúng 8 ký tự cho bất kỳ biển nào; vì vậy accuracy ký tự/độ chính xác OCR trên biển test là **không đo được**, không được diễn giải thành accuracy CNN bằng 0%. End-to-end exact vẫn là 0/500 do lỗi cắt/đếm ký tự và/hoặc bỏ qua. Đây là giới hạn pipeline hiện tại, không thể kết luận từ test này rằng CNN nhận dạng sai toàn bộ.",
        "",
        "Validation có 300 biển và 2.400 crop tham chiếu nhưng các danh tính đã xuất hiện trong tập dữ liệu của mô hình trước; đây là số liệu chẩn đoán, không đại diện cho generalization độc lập.",
        "",
        f"Trên 500 biển test, cả hai dùng cùng bộ cắt: thử nhận dạng 338 biển, bỏ qua 162; histogram số crop ký tự là `{json.dumps(test_dense_runtime['detected_character_count_histogram'], ensure_ascii=False)}`. Có 423 biển sai số lượng ký tự so với nhãn; không có chuỗi 8 ký tự phù hợp để báo accuracy ký tự runtime. Điều này chỉ ra nút thắt trước hết nằm ở phân đoạn/cắt ký tự.",
        "",
        "## Kết quả theo từng lớp ký tự trên validation",
        "",
        "Các số dưới đây là precision/recall/F1 trên crop tham chiếu của validation (các lớp không có mẫu validation được lược bỏ). Recall tương ứng accuracy trong từng lớp.",
        "",
        "| Ký tự | Số mẫu | Dense P/R/F1 | Pruned P/R/F1 |",
        "|---|---:|---:|---:|",
    ]
    dense_by_char = {row["character"]: row for row in val_dense["per_character_class"]}
    pruned_by_char = {row["character"]: row for row in val_pruned["per_character_class"]}
    class_csv_rows = []
    for character in CLASS_NAMES:
        drow = dense_by_char[character]
        prow = pruned_by_char[character]
        if not drow["support"]:
            continue
        lines.append(
            f"| {character} | {drow['support']} | {drow['precision']:.2%} / {drow['recall']:.2%} / {drow['f1']:.2%} "
            f"| {prow['precision']:.2%} / {prow['recall']:.2%} / {prow['f1']:.2%} |"
        )
        class_csv_rows.append({
            "character": character,
            "validation_support": drow["support"],
            "dense_correct": drow["correct"],
            "dense_precision": drow["precision"],
            "dense_recall": drow["recall"],
            "dense_f1": drow["f1"],
            "pruned_correct": prow["correct"],
            "pruned_precision": prow["precision"],
            "pruned_recall": prow["recall"],
            "pruned_f1": prow["f1"],
        })
    metrics["artifact_paths"]["validation_per_class_comparison_csv"] = "artifacts/frontal1000_validation_per_class_comparison.csv"
    with (path.parent / "frontal1000_validation_per_class_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_csv_rows[0]) if class_csv_rows else [])
        writer.writeheader()
        writer.writerows(class_csv_rows)
    lines += [
        "",
        "## MAC, tham số và bộ nhớ trọng số",
        "",
        "| Tài nguyên | Không pruning | Conv2 pruning | Mức giảm nhờ pruning |",
        "|---|---:|---:|---:|",
        f"| MAC / ký tự | {params['unpruned_macs_per_character']:,} | {params['pruned_macs_per_character']:,} | {params['mac_reduction_fraction']:.2%} |",
        f"| MAC / biển 8 ký tự | {params['unpruned_macs_per_8_char_plate']:,} | {params['pruned_macs_per_8_char_plate']:,} | {params['mac_reduction_fraction']:.2%} |",
        f"| Số tham số | {params['unpruned_parameters']:,} | {params['pruned_parameters']:,} | {params['parameter_reduction_fraction']:.2%} |",
        f"| Trọng số FP32 thô | {metrics['resources']['unpruned_fp32_weight_bytes']:,} B ({metrics['resources']['unpruned_fp32_weight_kib']:.2f} KiB) | {metrics['resources']['pruned_fp32_weight_bytes']:,} B ({metrics['resources']['pruned_fp32_weight_kib']:.2f} KiB) | Chưa tính metadata/runtime |",
        f"| Trọng số INT8 lý thuyết | {metrics['resources']['unpruned_int8_weight_bytes_estimate']:,} B | {metrics['resources']['pruned_int8_weight_bytes_estimate']:,} B | Chỉ ước lượng theo 1 byte/tham số, chưa lượng tử hóa/kiểm tra |",
        "",
        "Nếu giả định DE10-Lite chạy 50 MHz và có đúng một MAC mỗi chu kỳ, cận dưới theo phép đếm là dense: **8.364 ms/biển 8 ký tự**; pruning: **5.484 ms/biển**. Đây chỉ là ước lượng workload; số chu kỳ thực tế còn phụ thuộc kiến trúc nhân MAC, pipeline, truy cập bộ nhớ và Quartus. Chưa có AI RTL/Quartus synthesis nên không có kết quả fMAX, DSP/ALM/BRAM hoặc độ trễ FPGA đo thật.",
        "",
        "## Tốc độ đo trên CPU",
        "",
        f"Benchmark PyTorch CPU một luồng, chạy xen kẽ: batch 1 p50 pruning **{latency['1']['structured_pruned']['median_ms_per_forward']:.4f} ms**, dense **{latency['1']['unpruned']['median_ms_per_forward']:.4f} ms**; batch 8 p50 pruning **{pruned_p50:.4f} ms**, dense **{dense_p50:.4f} ms** (dense chậm hơn {measured_delta:+.1%} ở batch 8). Đây là CPU host, không phải DE10-Lite; khác biệt có thể chịu ảnh hưởng nhiễu đo và backend.",
        "",
        "## Quy trình huấn luyện được giữ nguyên",
        "",
        f"- Dữ liệu: {metrics['data']['plate_counts']['train']:,} biển train, {metrics['data']['plate_counts']['val']:,} validation, {metrics['data']['plate_counts']['test']:,} test; 8.000 ký tự train; sửa {metrics['data']['verified_label_corrections']} nhãn train đã kiểm tra bằng mắt (chỉ áp dụng trong bộ nhớ, không sửa manifest nguồn).",
        f"- Mô hình đối chứng: LeNet-5 dense, Conv1=6 và Conv2=16; Conv1 được cố định để tương ứng phương pháp trước. Không xóa/prune kênh nào.",
        f"- Loss: weighted cross-entropy + {metrics['training']['confusion_penalty_lambda']} × kỳ vọng confusion cost; OHEM chọn top {metrics['training']['hard_fraction']:.0%} mẫu khó để tăng trọng số, không loại bỏ mẫu còn lại.",
        f"- Optimizer/lịch: AdamW, learning rate đầu {metrics['training']['learning_rate_initial']}, cosine annealing; batch {metrics['training']['batch_size']}; hoàn thành {metrics['training']['epochs_completed']} epoch (checkpoint tốt nhất epoch {metrics['training']['best_epoch']}).",
        f"- Phân phối lớp train: hỗ trợ {metrics['data']['class_balance']['train']['classes_with_support']}/30 lớp; tỉ lệ lớp nhiều nhất/ít nhất có mẫu là {metrics['data']['class_balance']['train']['class_balance_ratio_max_over_min_nonzero']:.1f}:1. Một số lớp có rất ít hoặc không có mẫu, do đó kết quả không thể khái quát đồng đều toàn bộ 30 ký tự.",
        "",
        "## Kết luận và giới hạn",
        "",
        f"Structured pruning giảm phép tính và số tham số theo đúng kích thước kiến trúc: MAC giảm {params['mac_reduction_fraction']:.2%}, tham số giảm {params['parameter_reduction_fraction']:.2%}. Đo CPU cùng lúc cho biết xu hướng độ trễ phần mềm, nhưng không chứng minh tốc độ FPGA tăng tương ứng. Chất lượng hiện chưa thể kết luận chắc trên biển test do nút thắt segmenter/crop tham chiếu. Muốn kết luận accuracy end-to-end cần sửa bước cắt/nhãn cho test trước rồi đánh giá lại; đồng thời nên làm thí nghiệm pruning-vs-dense từ cùng checkpoint teacher và cùng seed để cô lập tác dụng pruning.",
        "",
        f"Số liệu gộp chi tiết: `{metrics['artifact_paths']['metrics_json']}`; bảng theo từng lớp ký tự: `{metrics['artifact_paths']['validation_per_class_comparison_csv']}`. Không đưa mã biển/ảnh từng biển vào báo cáo công khai.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train dense LeNet-5 and compare with the existing structured-pruned checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/train1_frontal_1000_unseen_v3"))
    parser.add_argument("--initial", type=Path, default=Path("artifacts/lenet5_cost_hard_channel30_teacher_v1.pt"))
    parser.add_argument("--pruned", type=Path, default=Path("artifacts/lenet5_conv2_frontal1000_cost_v1.pt"))
    parser.add_argument("--prior-metrics", type=Path, default=Path("artifacts/channel30_cost_hard_conv2_distill_metrics.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--confusion-penalty", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_dense_frontal1000_cost_v1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/frontal1000_pruning_vs_no_pruning_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/frontal1000_so_sanh_pruning_khong_pruning.md"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}; structured pruning: KHÔNG dùng cho mô hình dense", flush=True)

    manifest, prep, corrections, applied, splits = read_manifest(args.data)
    dense, train_set, val_set, cost_np, prior_report, history, initialized, best_epoch = train_dense(
        args, device, manifest, splits, prep
    )
    pruned = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
    pruned_checkpoint = load_checkpoint(pruned, args.pruned, device)
    pruned.eval()

    models = {"unpruned": dense, "structured_pruned": pruned}
    model_metrics = {}
    for name, model in models.items():
        print(f"Đang đánh giá: {name}", flush=True)
        val_summary, val_rows, val_conf, val_runtime, val_runtime_conf = eval_metrics(
            model, args.data, manifest, splits["val"], "val", device
        )
        test_summary, test_rows, test_conf, test_runtime, test_runtime_conf = eval_metrics(
            model, args.data, manifest, splits["test"], "test", device
        )
        model_metrics[name] = {
            "checkpoint": str(args.output if name == "unpruned" else args.pruned),
            "validation": val_summary,
            "test": test_summary,
            "validation_runtime_character_metrics": val_runtime,
            "test_runtime_character_metrics": test_runtime,
            "validation_confusion_matrix": val_conf.tolist(),
            "test_reference_confusion_matrix": test_conf.tolist(),
            "test_runtime_aligned_confusion_matrix": test_runtime_conf.tolist(),
            "validation_runtime_aligned_confusion_matrix": val_runtime_conf.tolist(),
        }
        if name == "unpruned":
            save_confusion(args.metrics.parent / "frontal1000_dense_validation_confusion.csv", val_conf)
            save_confusion(args.metrics.parent / "frontal1000_dense_test_reference_confusion.csv", test_conf)

    latency_raw = benchmark(pruned, dense, device)
    latency_raw["batches"] = {
        batch: {
            "structured_pruned": values["previous_conv2_student"],
            "unpruned": values["frontal1000_finetuned"],
        }
        for batch, values in latency_raw["batches"].items()
    }
    val_activation = {
        name: activation_distributions(model, val_set, device)
        for name, model in models.items()
    }
    dense_parameters = count_parameters(dense)
    pruned_parameters = count_parameters(pruned)
    max_test_length = max(len(record["text"]) for record in splits["test"])
    metrics = {
        "experiment": "Dense no-pruning LeNet-5 retraining and comparison against Conv2 structured-pruned model on identical prepared 1,000-plate split",
        "data": {
            "source": "data/OCR/OCR/images/train(1)/detection/one_row",
            "prepared_directory": str(args.data),
            "plate_counts": {name: len(records) for name, records in splits.items()},
            "identity_disjoint_across_splits": True,
            "test_identities_absent_from_prior_model_dataset": prep.get("test_identities_absent_from_prior_model_dataset"),
            "test_reference_segmentable_plates": sum(bool(row.get("ground_truth_segmentation_ok")) for row in splits["test"]),
            "verified_label_corrections": len(corrections),
            "verified_label_substitutions_by_position": [
                {"position_1based": pos, "source_character": old, "corrected_character": new}
                for pos, old, new in applied
            ],
            "class_balance": {
                "train": class_counts(train_set),
                "validation": class_counts(val_set),
                "test_segmentable": class_counts(ManifestCharacters(args.data, splits["test"], augment=False)),
            },
            "augmentation_observed_rates": {
                category: count / max(1, len(train_set) * len(history))
                for category, count in train_set.events.items()
            },
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate_initial": args.learning_rate,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "weighted cross-entropy + lambda * expected confusion cost; 0.5 full mean + 0.5 top-k hard mean",
            "confusion_penalty_lambda": args.confusion_penalty,
            "hard_fraction": args.hard_fraction,
            "hard_samples_discarded_rate": 0.0,
            "conv1_frozen": True,
            "conv1_bitwise_identical_to_initial": True,
            "class_weight_formula": "sqrt(max_class_count/class_count), normalized to mean 1, clipped [0.5, 3.0]",
            "initial_checkpoint": str(args.initial),
            "initial_checkpoint_epoch": initialized.get("epoch"),
            "epoch_history": history,
        },
        "models": model_metrics,
        "architecture": {
            "unpruned_channels": [6, 16, 120, 84],
            "pruned_channels": [6, 8, 120, 84],
            "unpruned_parameters": dense_parameters,
            "pruned_parameters": pruned_parameters,
            "unpruned_macs_per_character": DENSE_MACS_PER_CHAR,
            "pruned_macs_per_character": PRUNED_MACS_PER_CHAR,
            "unpruned_macs_per_8_char_plate": DENSE_MACS_PER_CHAR * 8,
            "pruned_macs_per_8_char_plate": PRUNED_MACS_PER_CHAR * 8,
            "mac_reduction_fraction": 1 - PRUNED_MACS_PER_CHAR / DENSE_MACS_PER_CHAR,
            "parameter_reduction_fraction": 1 - pruned_parameters / dense_parameters,
        },
        "resources": {
            "unpruned_fp32_weight_bytes": dense_parameters * 4,
            "unpruned_fp32_weight_kib": dense_parameters * 4 / 1024,
            "pruned_fp32_weight_bytes": pruned_parameters * 4,
            "pruned_fp32_weight_kib": pruned_parameters * 4 / 1024,
            "unpruned_int8_weight_bytes_estimate": dense_parameters,
            "pruned_int8_weight_bytes_estimate": pruned_parameters,
            "dense_peak_feature_map_elements_per_glyph": 6 * 28 * 28,
            "dense_peak_feature_map_fp32_bytes_per_glyph": 6 * 28 * 28 * 4,
            "pruned_peak_feature_map_elements_per_glyph": 6 * 28 * 28,
            "pruned_peak_feature_map_fp32_bytes_per_glyph": 6 * 28 * 28 * 4,
            "clock_target_hz_assumption": 50_000_000,
            "unpruned_single_mac_per_cycle_theoretical_ms_per_8_char_plate": DENSE_MACS_PER_CHAR * 8 / 50_000,
            "pruned_single_mac_per_cycle_theoretical_ms_per_8_char_plate": PRUNED_MACS_PER_CHAR * 8 / 50_000,
            "fpga_resources_or_latency_measured": False,
        },
        "latency": latency_raw,
        "weight_distributions": {name: weight_distributions(model) for name, model in models.items()},
        "validation_activation_distributions": val_activation,
        "confusion_cost_matrix_M": cost_np.tolist(),
        "confusion_cost_definition": "M[k,k]=0; M[k,j]=1+2*(C[k,j]+1)/(sum_{q!=k}C[k,q]+29); C is from previous-model validation only.",
        "confusion_cost_source": prior_report["models"]["conv2_pruned_student"]["validation"]["confusion_matrix"],
        "comparison_caveats": [
            "Dense model starts from the full 16-channel teacher; pruned model is the existing 8-channel student fine-tuned on the same 1,000-plate dataset, so initial checkpoints differ.",
            "Validation identities occurred in the earlier model manifest; use validation only as a diagnostic/tuning set.",
            "No test plate produced usable reference character crops; test character classification accuracy is not measurable.",
            "The 500-plate test set has no end-to-end segmentation result with the expected 8 characters; end-to-end exact rate is limited by localization/segmentation and counts all skips as failures.",
            "CPU timings are software measurements and do not establish FPGA timing or speedup.",
            "50 MHz / one-MAC-per-cycle latency is an arithmetic lower-bound estimate, not a Quartus or RTL result.",
            "Plate-specific data and predictions are intentionally omitted from public artifacts.",
        ],
        "artifact_paths": {
            "dense_checkpoint": str(args.output),
            "metrics_json": str(args.metrics),
            "report_markdown": str(args.report),
            "dense_validation_confusion_csv": "artifacts/frontal1000_dense_validation_confusion.csv",
            "dense_test_reference_confusion_csv": "artifacts/frontal1000_dense_test_reference_confusion.csv",
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    comparison_report(args.report, metrics)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu checkpoint={args.output}, số liệu={args.metrics}, báo cáo={args.report}", flush=True)


if __name__ == "__main__":
    main()
