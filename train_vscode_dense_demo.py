"""Run the reproducible dense LeNet-5 one-row → two-row experiment in VS Code.

The Run and Debug configuration uses VS Code's internal debug console so the
epoch log remains visible without requiring a terminal command in the UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_frontal1000_dense_compare import eval_metrics, read_manifest, train_dense
from train_two_row_dense import main as train_two_row_main


DATA_ONE = Path("data/train1_frontal_1000_unseen_v3")
DATA_TWO = Path("data/two_row_frontal_dense_v1")
INITIAL = Path("artifacts/lenet5_cost_hard_channel30_teacher_v1.pt")
PRIOR_COST = Path("artifacts/channel30_cost_hard_conv2_distill_metrics.json")
OUTPUT = Path("artifacts/vscode_demo")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str):
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_history(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def plot_history(one_history: list[dict], two_history: list[dict], output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot([r["epoch"] for r in one_history], [100 * r["validation_reference_crop_character_accuracy"] for r in one_history], label="Ký tự trên crop")
    axes[0].plot([r["epoch"] for r in one_history], [100 * r["validation_runtime_plate_exact_including_skips"] for r in one_history], label="Đúng toàn biển, validation")
    axes[0].set_title("Pha 1: ô tô một hàng")
    axes[1].plot([r["epoch"] for r in two_history], [100 * r["validation_character_accuracy_two_row"] for r in two_history], label="Ký tự trên crop")
    axes[1].plot([r["epoch"] for r in two_history], [100 * r["validation_plate_exact_two_row"] for r in two_history], label="Đúng toàn biển trên crop")
    axes[1].set_title("Pha 2: ô tô + xe máy hai hàng")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 101)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def top_confusions(path: Path, limit: int = 8):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    headers = table[0][1:]
    results = []
    for row in table[1:]:
        truth = row[0]
        for guess, value in zip(headers, row[1:]):
            count = int(value)
            if count and truth != guess:
                results.append((count, truth, guess))
    return sorted(results, reverse=True)[:limit]


def write_summary(output: Path, one: dict, two: dict, started: str, finished: str):
    one_result = one["models"]["unpruned"]
    two_model = two["models"]["dense_after_two_row_finetune"]
    before_model = two["models"]["phase1_dense_before_two_row"]
    reference = two_model["two_row_test_reference"]
    runtime = two_model["two_row_test_runtime"]
    previous_reference = before_model["two_row_test_reference"]
    previous_runtime = before_model["two_row_test_runtime"]
    one_val = one_result["validation"]
    one_test = one_result["test"]
    one_test_char = one_test["character_accuracy_conditional_on_ground_truth_segmentation"]
    one_test_char_label = f"{one_test_char:.2%}" if one_test_char is not None else "Không đo được (thiếu crop tham chiếu)"
    one_test_plate = one_test["plate_exact_accuracy_all_test_plates"]
    one_test_plate_label = f"{one_test_plate:.2%}" if one_test_plate is not None else "Không đo được (thiếu crop tham chiếu)"
    class_balance = two["data"]["class_balance"]
    confusions = top_confusions(output / "two_row_test_reference_after_confusion.csv")
    lines = [
        "# Trình bày tiến độ: LeNet-5 biển một hàng và hai hàng",
        "",
        f"Chạy trực tiếp bằng cấu hình Run and Debug của VS Code. Bắt đầu: **{started}**; kết thúc: **{finished}**. Mô hình có 30 lớp ký tự, Conv1=6 kênh, Conv2=16 kênh; không pruning.",
        "",
        "## Quy trình đã chạy",
        "",
        "1. Lấy 1.000 biển ô tô một hàng từ bộ đã lọc, crop ROI và ký tự 32×32.",
        "   Tập một hàng train/validation/test của lần chạy này đều là biển 8 ký tự; không suy rộng số này cho biển ô tô 9 hoặc 10 ký tự.",
        "2. Train LeNet-5 dense bằng weighted cross-entropy, confusion cost, hard mining 25%, augmentation; chọn checkpoint theo validation.",
        "3. Khi validation ký tự và toàn biển trên crop đều ≥95%, fine-tune trên ô tô và xe máy hai hàng; giữ Conv1 cố định, Conv2 vẫn đủ 16 kênh.",
        "4. Đánh giá riêng OCR trên crop tham chiếu và pipeline tự phân đoạn; bỏ qua được tính là sai trong exact toàn biển.",
        "",
        "![Biểu đồ accuracy theo epoch](training_accuracy.png)",
        "",
        "## Kết quả đo được",
        "",
        "| Tập / điều kiện | Accuracy ký tự | Exact toàn biển | Mẫu biển |",
        "|---|---:|---:|---:|",
        f"| Một hàng, validation crop tham chiếu | {one_val['character_accuracy_conditional_on_ground_truth_segmentation']:.2%} | {one_val['plate_exact_accuracy_all_test_plates']:.2%} | {one_val['plates']} |",
        f"| Một hàng, validation tự phân đoạn ROI | N/A | {one_val['runtime_no_known_count']['exact_accuracy_all_including_skips']:.2%} | {one_val['plates']} |",
        f"| Một hàng, test crop tham chiếu | {one_test_char_label} | {one_test_plate_label} | {one_test['plates']} |",
        f"| Một hàng, test tự phân đoạn ROI | N/A | {one_test['runtime_no_known_count']['exact_accuracy_all_including_skips']:.2%} ({one_test['runtime_no_known_count']['exact_plates_all_including_skips']}/{one_test['plates']}) | {one_test['plates']} |",
        f"| Hai hàng, test crop tham chiếu | {reference['character_accuracy']:.2%} ({reference['correct_characters']}/{reference['characters']}) | {reference['plate_exact_accuracy']:.2%} ({reference['exact_plates']}/{reference['plates']}) | {reference['plates']} |",
        f"| Hai hàng, test tự phân đoạn từ ROI | {runtime['character_accuracy_conditional_on_detected_count_match']:.2%} khi đếm đúng | {runtime['full_plate_exact_accuracy_all_frames_skips_fail']:.2%} ({runtime['full_plate_exact_all_frames_skips_fail']}/{runtime['plate_frames']}) | {runtime['plate_frames']} |",
        "",
        f"Fine-tune hai hàng tăng accuracy ký tự trên crop test từ {previous_reference['character_accuracy']:.2%} lên {reference['character_accuracy']:.2%}; exact crop từ {previous_reference['plate_exact_accuracy']:.2%} lên {reference['plate_exact_accuracy']:.2%}. Exact tự phân đoạn từ ROI tăng từ {previous_runtime['full_plate_exact_accuracy_all_frames_skips_fail']:.2%} lên {runtime['full_plate_exact_accuracy_all_frames_skips_fail']:.2%}.",
        "",
        "| Hai hàng theo xe | Accuracy ký tự crop | Exact crop | Exact tự phân đoạn từ ROI |",
        "|---|---:|---:|---:|",
    ]
    for kind, title in (("two_row_car", "Ô tô"), ("two_row_motorcycle", "Xe máy")):
        ref = reference["by_plate_type"][kind]
        raw = runtime["by_plate_type"][kind]
        lines.append(f"| {title} | {ref['character_accuracy']:.2%} | {ref['plate_exact_accuracy']:.2%} | {raw['full_plate_exact_accuracy_all_frames']:.2%} ({raw['exact']}/{raw['plates']}) |")
    lines += [
        "",
        "## Phân tích lỗi và giới hạn",
        "",
        f"Tập một hàng test có crop tham chiếu hợp lệ cho {one_test['ground_truth_segmentable_plates']}/{one_test['plates']} biển; pipeline tự phân đoạn có output ở {one_test['runtime_no_known_count']['attempted']}/{one_test['plates']} biển, nhưng chỉ đúng trọn {one_test['runtime_no_known_count']['exact_plates_all_including_skips']}. Do đó không được diễn giải exact test bằng 0 thành accuracy của riêng CNN bằng 0.",
        "Trong evaluator một hàng hiện tại, `plate_crop` đã chuẩn bị lại được đưa qua `perspective_correct` thêm một lần trước khi phân đoạn. Đây là lỗi quy trình có khả năng làm giảm mạnh kết quả test; phải sửa và đo lại trên cùng split trước khi dùng số 0/500 để đánh giá phiên bản triển khai thực tế.",
        "",
        f"Trong {runtime['plate_frames']} ảnh test hai hàng, chỉ {runtime['plates_with_valid_two_row_output']} ảnh có output hợp lệ; {runtime['skipped']} ảnh không có output. Ngoài ra {runtime['skip_reasons'].get('detected_count_mismatch', 0)} output có số ký tự không khớp nhãn. Nút thắt hiện tại chủ yếu nằm ở tìm/tách ký tự và phân loại hàng.",
        f"Số crop huấn luyện hai hàng: {class_balance['characters']:,} từ {class_balance['plates_with_reference_crops']:,} biển; đủ {class_balance['classes_with_support']}/30 lớp nhưng tỉ số lớp nhiều/ít nhất là {class_balance['class_balance_ratio_max_over_min_nonzero']:.1f}:1. Cần thêm mẫu lớp hiếm trước khi kết luận mọi ký tự đều chính xác như nhau.",
        "",
        "Các cặp nhầm lẫn nhiều nhất trên test crop tham chiếu:",
        "",
        "| Ký tự thật | Dự đoán | Lần |",
        "|---|---|---:|",
    ]
    for count, truth, guess in confusions:
        lines.append(f"| {truth} | {guess} | {count} |")
    lines += [
        "",
        "Validation một hàng dùng crop tham chiếu và đã vượt ngưỡng, nhưng test một hàng chưa đạt điều kiện tự phân đoạn. Chỉ số hai hàng trên crop tham chiếu cũng dùng độ dài nhãn khi chọn candidate phân đoạn, khác chế độ runtime. Tập hai hàng được chia theo text biển; checkpoint khởi tạo có thể đã thấy ảnh từ nguồn `train(1)` trong giai đoạn trước, nên số test này chưa phải đánh giá trên nguồn độc lập hoàn toàn. Nhãn gốc chưa được kiểm tra thủ công toàn bộ. Số runtime bắt đầu từ ROI biển theo bbox có sẵn; chưa gồm phát hiện biển từ ảnh toàn cảnh hay chạy trên DE10-Lite.",
        "",
        "## MAC, bộ nhớ, tốc độ",
        "",
        f"{two['architecture']['parameters']:,} tham số, {two['architecture']['macs_per_character']:,} MAC/ký tự. Với 8 ký tự: {two['architecture']['macs_per_plate_by_length']['8']:,} MAC; 9 ký tự: {two['architecture']['macs_per_plate_by_length']['9']:,} MAC; 10 ký tự: {two['architecture']['macs_per_plate_by_length']['10']:,} MAC. Nếu số ký tự bằng nhau, một hàng và hai hàng có cùng MAC của riêng LeNet-5; thuật toán phân loại/sắp xếp hàng là tiền xử lý bổ sung, chưa tính vào con số này. Trọng số FP32 thô {two['resources']['fp32_weight_kib']:.2f} KiB. Độ trễ CPU batch 1 p50 {two['latency_cpu']['batches']['1']['dense_after_two_row_finetune']['median_ms_per_forward']:.4f} ms/ký tự. Chưa có số timing và tài nguyên FPGA đo thật.",
        "",
        "## Mở trong VS Code để trình bày",
        "",
        "- `train_vscode_dense_demo.py`: quy trình hai pha; Run and Debug → **Train LeNet-5 1 hang + 2 hang (dense)** để chạy lại.",
        "- `live_train.log`: log từng epoch và thông báo đánh giá.",
        "- `training_accuracy.png`: biểu đồ validation theo epoch.",
        "- `one_row_training_history.csv`, `two_row_dense_training_history.csv`: loss, accuracy, learning rate, thời gian epoch.",
        "- `one_row_metrics.json`, `two_row_dense_metrics.json`: đầy đủ mẫu số, tỷ lệ, confusion matrix, augmentation, confidence, MAC và benchmark.",
        "- `two_row_test_per_class_comparison.csv`, `two_row_test_reference_after_confusion.csv`: chi tiết từng ký tự và lỗi nhầm lẫn.",
        "",
        "Ảnh crop minh họa lưu tại `data/train1_frontal_1000_unseen_v3/review/frontal_train_examples.jpg` và `data/two_row_frontal_dense_v1/review/two_row_frontal_examples.jpg`.",
    ]
    (output / "BANG_KET_QUA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"=== TRAIN TRÊN VS CODE | {started} | {device} | NO PRUNING ===", flush=True)
    print("Pha 1/2: 1.000 biển ô tô một hàng", flush=True)

    manifest, prep, corrections, applied, splits = read_manifest(DATA_ONE)
    stage_one_args = SimpleNamespace(
        data=DATA_ONE,
        initial=INITIAL,
        prior_metrics=PRIOR_COST,
        epochs=args.one_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hard_fraction=0.25,
        confusion_penalty=0.25,
        output=args.output / "lenet5_dense_one_row.pt",
    )
    model, train_set, _val_set, _cost, _prior, one_history, initialized, best_epoch = train_dense(
        stage_one_args, device, manifest, splits, prep
    )
    val_summary, _val_rows, _val_conf, val_runtime, _val_runtime_conf = eval_metrics(
        model, DATA_ONE, manifest, splits["val"], "val", device
    )
    test_summary, _test_rows, _test_conf, test_runtime, _test_runtime_conf = eval_metrics(
        model, DATA_ONE, manifest, splits["test"], "test", device
    )
    one_metrics = {
        "experiment": "VS Code dense LeNet-5 one-row stage; no pruning",
        "data": {"source": str(DATA_ONE), "train_plates": len(splits["train"]), "validation_plates": len(splits["val"]), "test_plates": len(splits["test"]), "verified_training_label_corrections": len(corrections)},
        "training": {"epochs_completed": len(one_history), "best_epoch": best_epoch, "history": one_history, "initialized_from": str(INITIAL), "pruning": "none", "conv1_frozen": True},
        "models": {"unpruned": {"validation": val_summary, "test": test_summary, "validation_runtime_character_metrics": val_runtime, "test_runtime_character_metrics": test_runtime}},
    }
    one_metrics_path = args.output / "one_row_metrics.json"
    one_metrics_path.write_text(json.dumps(one_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_history(args.output / "one_row_training_history.csv", one_history)
    gate_character = val_summary["character_accuracy_conditional_on_ground_truth_segmentation"]
    gate_plate = val_summary["plate_exact_accuracy_all_test_plates"]
    print(f"Pha 1 xong: val ký tự={gate_character:.3%}; val toàn biển trên crop={gate_plate:.3%}; test tự phân đoạn={test_summary['runtime_no_known_count']['exact_accuracy_all_including_skips']:.3%}", flush=True)
    if gate_character < 0.95 or gate_plate < 0.95:
        (args.output / "BANG_KET_QUA.md").write_text(
            f"# Kết quả pha một hàng\n\nValidation ký tự {gate_character:.3%}; toàn biển trên crop {gate_plate:.3%}. Chưa đạt ngưỡng tối thiểu 95%, nên chưa chuyển sang train hai hàng.\n\nXem `one_row_metrics.json` và `one_row_training_history.csv`.\n",
            encoding="utf-8",
        )
        print("Dừng theo cổng validation 95%; chưa train pha hai hàng.", flush=True)
        return

    print("Pha 2/2: ô tô + xe máy hai hàng; Conv2 giữ đủ 16 kênh", flush=True)
    train_two_row_main([
        "--one-row-data", str(DATA_ONE),
        "--two-row-data", str(DATA_TWO),
        "--phase1-checkpoint", str(stage_one_args.output),
        "--phase1-metrics", str(one_metrics_path),
        "--prior-cost-metrics", str(PRIOR_COST),
        "--epochs", str(args.two_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--output", str(args.output / "lenet5_dense_two_row.pt"),
        "--metrics", str(args.output / "two_row_dense_metrics.json"),
        "--report", str(args.output / "BAO_CAO_HAI_HANG.md"),
    ])
    two_metrics = json.loads((args.output / "two_row_dense_metrics.json").read_text(encoding="utf-8"))
    plot_history(one_history, two_metrics["training"]["epoch_history"], args.output / "training_accuracy.png")
    finished = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    write_summary(args.output, one_metrics, two_metrics, started, finished)
    print(f"=== HOÀN TẤT {finished} ===", flush=True)
    print(f"Mở báo cáo: {args.output / 'BANG_KET_QUA.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and display dense LeNet-5 on one-row then two-row plate data in VS Code.")
    parser.add_argument("--one-epochs", type=int, default=20)
    parser.add_argument("--two-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report-only", action="store_true", help="rebuild the chart and report from completed metric files without retraining")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        one = json.loads((args.output / "one_row_metrics.json").read_text(encoding="utf-8"))
        two = json.loads((args.output / "two_row_dense_metrics.json").read_text(encoding="utf-8"))
        plot_history(one["training"]["history"], two["training"]["epoch_history"], args.output / "training_accuracy.png")
        log_lines = (args.output / "live_train.log").read_text(encoding="utf-8").splitlines()
        started = log_lines[0].split(" | ")[1] if log_lines else "xem live_train.log"
        finished_lines = [line.removeprefix("=== HOÀN TẤT ").removesuffix(" ===") for line in log_lines if line.startswith("=== HOÀN TẤT ")]
        finished = finished_lines[-1] if finished_lines else "xem live_train.log"
        write_summary(args.output, one, two, started, finished)
        print(f"Đã tạo lại báo cáo: {args.output / 'BANG_KET_QUA.md'}")
        return
    with (args.output / "live_train.log").open("w", encoding="utf-8", buffering=1) as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            run(args)
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()
