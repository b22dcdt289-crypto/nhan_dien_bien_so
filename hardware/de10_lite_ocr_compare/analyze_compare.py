"""Cross-check all four JTAG runs and Quartus reports against matched training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ("dense_rom", "dense_stream", "pruned_rom", "pruned_stream")
CLOCK_HZ = 50_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource(text: str, label: str) -> tuple[int, int]:
    found = re.search(r"^" + re.escape(label) + r"\s*:\s*([\d,]+)\s*/\s*([\d,]+)", text, re.M)
    if not found:
        raise ValueError(f"Missing Quartus fit field: {label}")
    return tuple(int(value.replace(",", "")) for value in found.groups())


def percentile(values: list[float], percentile_fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(percentile_fraction * (len(ordered) - 1))]


def read_variant(name: str) -> dict:
    project = HERE / name
    generated = project / "generated"
    output = project / "output_files"
    manifest_file = generated / "manifest.json"
    board_file = generated / "board_results.csv"
    fit_file = output / "OcrBench.fit.summary"
    sta_file = output / "OcrBench.sta.rpt"
    bitstream_file = output / "OcrBench.sof"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with board_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest["character_count"]:
        raise ValueError(f"{name}: incomplete JTAG run {len(rows)} of {manifest['character_count']}")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{name}: sample sequence not 0..N-1")
    seq = [int(row["result_seq"]) for row in rows]
    if any((right - left) % 256 != 1 for left, right in zip(seq, seq[1:])):
        raise ValueError(f"{name}: JTAG result sequence skipped or duplicated")
    predictions = [manifest["class_order"][int(row["predicted_class"])] for row in rows]
    prediction_digest = hashlib.sha256("".join(predictions).encode("ascii")).hexdigest()
    cycles = [int(row["cycles"]) for row in rows]
    if len(set(cycles)) != 1:
        raise ValueError(f"{name}: unexpected varying hardware cycles {sorted(set(cycles))}")
    fixed_agreement = sum(prediction == item["fixed"] for prediction, item in zip(predictions, manifest["samples"]))
    char_correct = sum(prediction == item["label"] for prediction, item in zip(predictions, manifest["samples"]))
    type_counts = defaultdict(lambda: {"plates": 0, "exact": 0, "chars": 0, "char_correct": 0})
    plate_correct = 0
    for plate in manifest["plates"]:
        predicted_text = "".join(predictions[index] for index in plate["sample_indices"])
        exact = int(predicted_text == plate["truth"])
        plate_correct += exact
        stats = type_counts[plate["type"]]
        stats["plates"] += 1
        stats["exact"] += exact
        stats["chars"] += len(plate["truth"])
        stats["char_correct"] += sum(a == b for a, b in zip(predicted_text, plate["truth"]))
    fit = fit_file.read_text(encoding="utf-8", errors="replace")
    sta = sta_file.read_text(encoding="utf-8", errors="replace")
    fmax_match = re.search(r";\s*([\d.]+) MHz\s*;\s*[\d.]+ MHz\s*;\s*MAX10_CLK1_50\s*;", sta)
    slack_match = re.search(r"Worst-case setup slack is\s*([\d.-]+)", sta)
    if not fmax_match or not slack_match:
        raise ValueError(f"{name}: missing timing values")
    load = [int(row["jtag_load_ms"]) for row in rows]
    total = [int(row["jtag_total_ms"]) for row in rows]
    return {
        "variant": name, "architecture": manifest["architecture"],
        "streaming_input": manifest["streaming_input"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "manifest_sha256": sha256(manifest_file), "board_csv_sha256": sha256(board_file),
        "bitstream_sha256": sha256(bitstream_file), "bitstream_bytes": bitstream_file.stat().st_size,
        "jtag_csv_last_modified_local": datetime.fromtimestamp(board_file.stat().st_mtime).isoformat(timespec="seconds"),
        "plate_count": manifest["plate_count"], "character_count": manifest["character_count"],
        "fp32_character_correct": manifest["fp32_character_correct"],
        "fixed_character_correct": manifest["fixed_character_correct"],
        "fp32_plate_correct": manifest["fp32_plate_correct"],
        "fixed_plate_correct": manifest["fixed_plate_correct"],
        "fpga_character_correct": char_correct, "fpga_plate_correct": plate_correct,
        "fpga_fixed_agreement": fixed_agreement,
        "prediction_sequence_sha256": prediction_digest,
        "by_plate_type": dict(type_counts),
        "cycles_per_character": cycles[0],
        "compute_ms_per_character_at_50mhz": cycles[0] * 1000 / CLOCK_HZ,
        "jtag_load_ms_median": statistics.median(load),
        "jtag_load_ms_p95": percentile(load, 0.95),
        "jtag_total_ms_median": statistics.median(total),
        "jtag_total_ms_p95": percentile(total, 0.95),
        "logic_elements": dict(zip(("used", "available"), resource(fit, "Total logic elements"))),
        "memory_bits": dict(zip(("used", "available"), resource(fit, "Total memory bits"))),
        "embedded_multiplier_9bit": dict(zip(("used", "available"), resource(fit, "Embedded Multiplier 9-bit elements"))),
        "fmax_mhz_slow_85c": float(fmax_match.group(1)),
        "setup_slack_ns_slow_85c": float(slack_match.group(1)),
        "timing_not_fully_constrained": "not fully constrained" in sta.lower(),
        "quant_shifts_by_layer": manifest["shifts"],
        "model_weight_bytes_int8": manifest["weight_bytes"],
        "image_bytes_embedded_for_test": 0 if manifest["streaming_input"] else manifest["character_count"] * 1024,
        "image_frame_buffer_bytes": 1024 if manifest["streaming_input"] else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=ROOT / "artifacts/compare_conv2_stream_20260929/training_metrics.json")
    args = parser.parse_args()
    training = json.loads(args.training.read_text(encoding="utf-8"))
    variants = {name: read_variant(name) for name in VARIANTS}
    for pair in (("dense_rom", "dense_stream"), ("pruned_rom", "pruned_stream")):
        if variants[pair[0]]["checkpoint_sha256"] != variants[pair[1]]["checkpoint_sha256"]:
            raise ValueError("ROM/streaming builds do not use same checkpoint")
        if variants[pair[0]]["prediction_sequence_sha256"] != variants[pair[1]]["prediction_sequence_sha256"]:
            raise ValueError("ROM/streaming builds do not return identical per-character predictions")
    reference = variants["dense_rom"]
    pruned = variants["pruned_rom"]
    result = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "clock_hz": CLOCK_HZ,
        "training": training,
        "variants": variants,
        "comparison": {
            "pruning_macs_reduction_percent": 100 * (1 - training["models"]["conv2_pruned"]["macs_per_character"] / training["models"]["dense"]["macs_per_character"]),
            "pruning_rom_compute_speedup_x": reference["cycles_per_character"] / pruned["cycles_per_character"],
            "pruning_rom_le_reduction_percent": 100 * (1 - pruned["logic_elements"]["used"] / reference["logic_elements"]["used"]),
            "pruning_rom_ram_reduction_percent": 100 * (1 - pruned["memory_bits"]["used"] / reference["memory_bits"]["used"]),
            "dense_stream_ram_reduction_percent": 100 * (1 - variants["dense_stream"]["memory_bits"]["used"] / reference["memory_bits"]["used"]),
            "pruned_stream_ram_reduction_percent": 100 * (1 - variants["pruned_stream"]["memory_bits"]["used"] / pruned["memory_bits"]["used"]),
            "pruning_stream_compute_speedup_x": variants["dense_stream"]["cycles_per_character"] / variants["pruned_stream"]["cycles_per_character"],
            "pruning_stream_ram_reduction_percent": 100 * (1 - variants["pruned_stream"]["memory_bits"]["used"] / variants["dense_stream"]["memory_bits"]["used"]),
            "pruned_stream_vs_dense_rom_compute_speedup_x": reference["cycles_per_character"] / variants["pruned_stream"]["cycles_per_character"],
            "pruned_stream_vs_dense_rom_ram_reduction_percent": 100 * (1 - variants["pruned_stream"]["memory_bits"]["used"] / reference["memory_bits"]["used"]),
            "pruned_stream_vs_dense_rom_le_change_percent": 100 * (variants["pruned_stream"]["logic_elements"]["used"] / reference["logic_elements"]["used"] - 1),
            "pruned_stream_vs_dense_rom_observed_jtag_total_speedup_x": reference["jtag_total_ms_median"] / variants["pruned_stream"]["jtag_total_ms_median"],
        },
        "limitations": [
            "Streaming means JTAG writes one pre-cropped 32x32 character into FPGA RAM; it is not direct camera pixel processing or a fully pipelined convolution stream.",
            "Quartus ROM builds embed all 50 test images in the bitstream, whereas streamed builds keep only a 1 KiB frame buffer; memory differences include this benchmark setup.",
            "USB-Blaster script wall time includes JTAG and intentional handshake delays; it is not camera throughput or pure FPGA compute latency.",
            "Validation crop segmentation used source labels; frame-level train/val split may leak plate identities. The 6-plate FPGA subset is too small to establish field accuracy.",
            "Fmax/slack are post-fit estimates with incomplete timing constraints and are not a physical oscilloscope frequency measurement.",
        ],
    }
    (HERE / "comparison_full.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(args.training.parent / "history.csv", HERE / "train_history.csv")
    tm = training["models"]
    fmt = lambda value: f"{value:.2f}"
    lines = [
        "# Đối chứng LeNet-5 dense, structured Conv2 pruning và JTAG frame streaming",
        "",
        f"Lần tổng hợp báo cáo: {result['generated_at_local']}. Cùng tập `train(1)` đã chuẩn bị: "
        f"{training['split']['train_character_crops']:,} crop train, "
        f"{training['split']['validation_character_crops']:,} crop validation từ "
        f"{training['split']['validation_plates']:,} biển. Dùng 30 lớp ký tự.",
        "",
        "## Train (CPU laptop, không phải tài nguyên FPGA)",
        "",
        "Cả hai nhánh xem dữ liệu 10 epoch: 5 epoch dense chung; sau đó dense tiếp tục 5 epoch, "
        "còn bản tỉa Conv2 từ 16 xuống 8 kênh và train tiếp 5 epoch. Không KD; cùng loss, "
        "optimizer, batch, seed và augmentation. Streaming chỉ là cách nạp ảnh khi suy luận, không phải kỹ thuật train.",
        "",
        "| Mục | Dense | Tỉa kênh Conv2 |",
        "|---|---:|---:|",
        f"| Tham số | {tm['dense']['parameters']:,} | {tm['conv2_pruned']['parameters']:,} |",
        f"| MAC lý thuyết/ký tự | {tm['dense']['macs_per_character']:,} | {tm['conv2_pruned']['macs_per_character']:,} |",
        f"| Best epoch | {tm['dense']['best_epoch']} | {tm['conv2_pruned']['best_epoch']} |",
        f"| Validation crop ký tự (best) | {100*tm['dense']['best_val_crop_character_accuracy']:.2f}% | {100*tm['conv2_pruned']['best_val_crop_character_accuracy']:.2f}% |",
        f"| Validation toàn biển trên crop | {100*tm['dense']['validation_plate_metrics']['exact_plate_accuracy']:.2f}% ({tm['dense']['validation_plate_metrics']['exact_plates']}/{tm['dense']['validation_plate_metrics']['plates']}) | {100*tm['conv2_pruned']['validation_plate_metrics']['exact_plate_accuracy']:.2f}% ({tm['conv2_pruned']['validation_plate_metrics']['exact_plates']}/{tm['conv2_pruned']['validation_plate_metrics']['plates']}) |",
        f"| Train + validation (gồm 5 epoch chung) | {fmt(training['training']['dense_branch_train_plus_val_seconds'])} s | {fmt(training['training']['pruned_branch_train_plus_val_seconds'])} s |",
        f"| CPU forward p50 batch 1 | {tm['dense']['cpu_forward']['1']['median_ms_per_batch']:.4f} ms | {tm['conv2_pruned']['cpu_forward']['1']['median_ms_per_batch']:.4f} ms |",
        f"| CPU forward p50 batch 8 | {tm['dense']['cpu_forward']['8']['median_ms_per_batch']:.4f} ms | {tm['conv2_pruned']['cpu_forward']['8']['median_ms_per_batch']:.4f} ms |",
        "",
        f"Thời gian toàn thí nghiệm (nạp data + train hai nhánh + đánh giá): {fmt(training['training']['experiment_wall_seconds'])} s; "
        f"đỉnh RSS của **tiến trình chứa cả hai model và cache ảnh**: {training['training']['process_peak_rss_bytes']/1024**2:.1f} MiB. "
        "Không gán con số RAM này cho riêng từng model. Chi tiết từng epoch trong `artifacts/compare_conv2_stream_20260929/history.csv`.",
        "",
        "MAC/ký tự = Conv1 `6×25×28×28` + Conv2 `6×C2×25×10×10` + "
        "FC1 `C2×25×120` + FC2 `120×84` + FC3 `84×30`. "
        "Với `C2=16`: 117.600 + 240.000 + 48.000 + 10.080 + 2.520 = 418.200 MAC; "
        "với `C2=8`: 117.600 + 120.000 + 24.000 + 10.080 + 2.520 = 274.200 MAC. "
        "Không tính phép cộng bias, tanh và pooling vào MAC; đó là phép tính khác.",
        "",
        "| Nhóm biển validation | Dense: ký tự đúng | Dense: biển đúng | Pruned: ký tự đúng | Pruned: biển đúng |",
        "|---|---:|---:|---:|---:|",
    ]
    for plate_type, label in (("one_row", "Ô tô một hàng"),
                              ("two_row_car", "Ô tô hai hàng"),
                              ("two_row_motorcycle", "Xe máy hai hàng")):
        d = tm["dense"]["validation_plate_metrics"]["by_plate_type"][plate_type]
        p = tm["conv2_pruned"]["validation_plate_metrics"]["by_plate_type"][plate_type]
        lines.append(f"| {label} | {d['correct']}/{d['characters']} ({100*d['character_accuracy']:.2f}%) | "
                     f"{d['exact']}/{d['plates']} ({100*d['exact_plate_accuracy']:.2f}%) | "
                     f"{p['correct']}/{p['characters']} ({100*p['character_accuracy']:.2f}%) | "
                     f"{p['exact']}/{p['plates']} ({100*p['exact_plate_accuracy']:.2f}%) |")
    lines += [
        "",
        "Tổng validation và từng nhóm này là kết quả **crop ký tự với phân đoạn dựa vào nhãn**, "
        "không phải OCR biển nguyên ảnh/camera. Mục tiêu 95,22% biển đúng "
        "trên validation toàn bộ chưa đạt, nhất là nhóm xe máy hai hàng.",
        "",
        "## Kết quả sau fit và chạy trên DE10-Lite (50 crop từ 6 biển)",
        "",
        "| Chỉ số | Dense ROM | Dense stream | Pruned ROM | Pruned stream |",
        "|---|---:|---:|---:|---:|",
    ]
    def row(label: str, fn) -> None:
        lines.append("| " + label + " | " + " | ".join(str(fn(variants[name])) for name in VARIANTS) + " |")
    row("Ký tự đúng FPGA", lambda v: f"{v['fpga_character_correct']}/{v['character_count']}")
    row("Biển đúng toàn bộ", lambda v: f"{v['fpga_plate_correct']}/{v['plate_count']}")
    row("Khớp mô phỏng INT8", lambda v: f"{v['fpga_fixed_agreement']}/{v['character_count']}")
    row("LE Quartus", lambda v: f"{v['logic_elements']['used']:,}/{v['logic_elements']['available']:,}")
    row("RAM bit Quartus", lambda v: f"{v['memory_bits']['used']:,}/{v['memory_bits']['available']:,}")
    row("Phần tử nhân 9-bit", lambda v: f"{v['embedded_multiplier_9bit']['used']}/{v['embedded_multiplier_9bit']['available']}")
    row("Chu kỳ/ký tự", lambda v: f"{v['cycles_per_character']:,}")
    row("Tính toán/ký tự ở 50 MHz", lambda v: f"{v['compute_ms_per_character_at_50mhz']:.3f} ms")
    row("JTAG nạp ảnh p50", lambda v: f"{v['jtag_load_ms_median']:.1f} ms")
    row("JTAG toàn lượt p50", lambda v: f"{v['jtag_total_ms_median']:.1f} ms")
    row("Fmax post-fit slow 85°C", lambda v: f"{v['fmax_mhz_slow_85c']:.2f} MHz")
    row("Setup slack @50 MHz", lambda v: f"{v['setup_slack_ns_slow_85c']:.3f} ns")
    row("Trọng số INT8", lambda v: f"{v['model_weight_bytes_int8']:,} byte")
    row("Ảnh test nhúng ROM", lambda v: f"{v['image_bytes_embedded_for_test']:,} byte")
    row("Bộ đệm ảnh stream", lambda v: f"{v['image_frame_buffer_bytes']:,} byte")
    lines += [
        "",
        f"Giảm MAC do tỉa kênh: {result['comparison']['pruning_macs_reduction_percent']:.2f}%. "
        f"Tăng tốc **lõi tính toán FPGA** pruned ROM so với dense ROM: "
        f"{result['comparison']['pruning_rom_compute_speedup_x']:.3f}×. "
        f"Streaming chỉ thay đường nạp ảnh; không được gọi là tăng tốc train.",
        "",
        f"Trong cặp **cùng truyền ảnh**, pruning giảm RAM nội bộ "
        f"{result['comparison']['pruning_stream_ram_reduction_percent']:.2f}% "
        f"({variants['dense_stream']['memory_bits']['used']:,} → {variants['pruned_stream']['memory_bits']['used']:,} bit), "
        f"nhưng thời gian JTAG toàn lượt p50 chỉ đổi từ "
        f"{variants['dense_stream']['jtag_total_ms_median']:.1f} thành "
        f"{variants['pruned_stream']['jtag_total_ms_median']:.1f} ms/ký tự: "
        "đường truyền chậm lấn át lợi ích lõi. So với Dense ROM, bản Pruned stream "
        f"giảm {result['comparison']['pruned_stream_vs_dense_rom_ram_reduction_percent']:.2f}% RAM "
        f"nhưng tăng {result['comparison']['pruned_stream_vs_dense_rom_le_change_percent']:.2f}% LE "
        "vì có logic nhận JTAG và bộ đệm. Hai phép so này có phạm vi khác nhau.",
        "",
        "Thời gian lõi cho biển N ký tự, nếu xử lý nối tiếp, xấp xỉ `N × chu kỳ/ký tự ÷ 50 MHz`; "
        "chưa tính cắt ảnh, giao tiếp, điều khiển. Với 8 ký tự: "
        f"dense {8*reference['compute_ms_per_character_at_50mhz']:.3f} ms, "
        f"pruned {8*pruned['compute_ms_per_character_at_50mhz']:.3f} ms. "
        "Với giao tiếp JTAG thử nghiệm, thời gian thực tế lớn hơn đáng kể; "
        "không dùng phép tính lõi này để tuyên bố tốc độ hệ thống camera.",
        "",
        "## Phạm vi và cách kiểm chứng",
        "",
        "- Mở `comparison_full.json` để xem số đầy đủ, checksum, breakdown; "
        "`train_history.csv` để xem từng epoch.",
        "- Mỗi cấu hình có `generated/board_results.csv` (raw JTAG), `generated/manifest.json` "
        "(nhãn và dự đoán tham chiếu, **chỉ lưu trên laptop**), `output_files/OcrBench.fit.summary` "
        "và `output_files/OcrBench.sta.rpt`.",
        "- Streaming ở đây là nạp **một ảnh ký tự đã cắt** qua JTAG vào RAM 1 KiB; "
        "không phải camera pixel stream, không tăng tốc bộ MAC. Thời gian JTAG phụ thuộc "
        "USB-Blaster và thời gian chờ handshake trong Tcl, không đại diện cho giao tiếp triển khai cuối.",
        "- ROM gắn sẵn 50 ảnh test nên RAM của nó lớn hơn; chênh lệch tài nguyên không phải "
        "chỉ do pruning. Tập FPGA 6 biển quá nhỏ để công bố accuracy thực tế.",
        "- Validation dùng crop tham chiếu với số ký tự lấy từ nhãn nguồn, có nguy cơ trùng biển "
        "giữa train/validation. Fmax/slack chưa bảo đảm timing được ràng buộc đầy đủ.",
    ]
    (HERE / "BAO_CAO_SO_SANH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(HERE / "BAO_CAO_SO_SANH.md"),
                      "comparison": result["comparison"],
                      "fpga_correct": {name: [variants[name]["fpga_character_correct"], variants[name]["fpga_plate_correct"]]
                                       for name in VARIANTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
