"""Summarize actual FPGA JTAG results without publishing individual plate IDs."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
FIT = HERE / "output_files/OcrDemo.fit.summary"
STA = HERE / "output_files/OcrDemo.sta.rpt"
CLOCK_HZ = 50_000_000


def resource(pattern: str, text: str) -> tuple[int, int]:
    found = re.search(pattern, text, re.MULTILINE)
    if not found:
        raise ValueError(f"Missing Quartus resource line: {pattern}")
    return tuple(int(value.replace(",", "")) for value in found.groups())


def wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - margin, center + margin


def main() -> None:
    manifest = json.loads((GENERATED / "demo_manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((GENERATED / "board_results.csv").open(encoding="utf-8", newline="")))
    if len(rows) != manifest["character_count"]:
        raise ValueError(f"Board returned {len(rows)} of {manifest['character_count']} characters")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Missing or out-of-order sample indices")
    classes = manifest["class_order"]
    predictions = [classes[int(row["predicted_class"])] for row in rows]
    cycles = [int(row["cycles"]) for row in rows]
    if len(set(cycles)) != 1:
        raise ValueError(f"Cycle count varies unexpectedly: {sorted(set(cycles))}")
    if "result_seq" in rows[0]:
        seq = [int(row["result_seq"]) for row in rows]
        if any((b - a) % 256 != 1 for a, b in zip(seq, seq[1:])):
            raise ValueError("JTAG completion sequence skipped or duplicated")
    repeat_path = GENERATED / "board_results_stable_run1.csv"
    repeat_agreement = None
    if repeat_path.is_file():
        first_run = list(csv.DictReader(repeat_path.open(encoding="utf-8", newline="")))
        if len(first_run) != len(rows):
            raise ValueError("Repeat run has a different sample count")
        repeat_agreement = sum(
            a["sample_index"] == b["sample_index"]
            and a["predicted_class"] == b["predicted_class"]
            and a["cycles"] == b["cycles"]
            for a, b in zip(first_run, rows)
        )
    char_correct = sum(p == item["label"] for p, item in zip(predictions, manifest["samples"]))
    fixed_agree = sum(p == item["fixed"] for p, item in zip(predictions, manifest["samples"]))
    plate_correct = 0
    type_stats = defaultdict(lambda: {"plates": 0, "plate_correct": 0, "chars": 0, "char_correct": 0})
    private_plate_rows = []
    private_character_rows = []
    for plate_index, plate in enumerate(manifest["plates"]):
        predicted = "".join(predictions[index] for index in plate["sample_indices"])
        correct = predicted == plate["truth"]
        plate_correct += correct
        private_plate_rows.append({
            "plate_index": plate_index,
            "type": plate["type"],
            "source": plate["source"],
            "truth": plate["truth"],
            "fp32": plate["fp32"],
            "fixed": plate["fixed"],
            "fpga": predicted,
            "exact": int(correct),
        })
        for position, sample_index in enumerate(plate["sample_indices"]):
            private_character_rows.append({
                "plate_index": plate_index,
                "position": position,
                "sample_index": sample_index,
                "truth": plate["truth"][position],
                "fixed": manifest["samples"][sample_index]["fixed"],
                "fpga": predictions[sample_index],
                "correct": int(predictions[sample_index] == plate["truth"][position]),
                "cycles": cycles[sample_index],
            })
        stats = type_stats[plate["type"]]
        stats["plates"] += 1
        stats["plate_correct"] += correct
        stats["chars"] += len(plate["truth"])
        stats["char_correct"] += sum(a == b for a, b in zip(predicted, plate["truth"]))
    for path, detail_rows in (
        (GENERATED / "plate_results_private.csv", private_plate_rows),
        (GENERATED / "character_results_private.csv", private_character_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
            writer.writeheader()
            writer.writerows(detail_rows)

    fit = FIT.read_text(encoding="utf-8", errors="replace")
    sta = STA.read_text(encoding="utf-8", errors="replace")
    le_used, le_total = resource(r"^Total logic elements\s*:\s*([\d,]+)\s*/\s*([\d,]+)", fit)
    mem_used, mem_total = resource(r"^Total memory bits\s*:\s*([\d,]+)\s*/\s*([\d,]+)", fit)
    mult_used, mult_total = resource(r"^Embedded Multiplier 9-bit elements\s*:\s*([\d,]+)\s*/\s*([\d,]+)", fit)
    fmax_match = re.search(r";\s*([\d.]+) MHz\s*;\s*[\d.]+ MHz\s*;\s*MAX10_CLK1_50\s*;", sta)
    slack_match = re.search(r"Worst-case setup slack is\s*([\d.-]+)", sta)
    if not fmax_match or not slack_match:
        raise ValueError("Missing Fmax or setup slack")
    result = {
        "scope": "INT8 fixed-point LeNet-5 Conv2-pruned on pre-cropped 32x32 characters embedded in bitstream",
        "source_split": "train(1) validation, frame-level split, not independent plate identities",
        "seed": manifest["seed"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "plate_count": len(manifest["plates"]),
        "character_count": len(rows),
        "character_correct": char_correct,
        "character_accuracy": char_correct / len(rows),
        "character_wilson95": wilson(char_correct, len(rows)),
        "plate_correct": plate_correct,
        "plate_accuracy": plate_correct / len(manifest["plates"]),
        "plate_wilson95": wilson(plate_correct, len(manifest["plates"])),
        "fpga_vs_fixed_agreement": fixed_agree,
        "repeat_run_agreement": repeat_agreement,
        "type_breakdown": dict(type_stats),
        "macs_per_character": 274200,
        "cycles_per_character": cycles[0],
        "clock_hz": CLOCK_HZ,
        "character_compute_ms": cycles[0] * 1000 / CLOCK_HZ,
        "eight_character_compute_ms": 8 * cycles[0] * 1000 / CLOCK_HZ,
        "logic_elements": {"used": le_used, "available": le_total},
        "memory_bits": {"used": mem_used, "available": mem_total},
        "embedded_multiplier_9bit": {"used": mult_used, "available": mult_total},
        "fmax_mhz_slow_85c": float(fmax_match.group(1)),
        "setup_slack_ns_slow_85c": float(slack_match.group(1)),
        "model_weight_bytes_int8": manifest["weight_bytes"],
        "included_test_image_bytes": len(rows) * 1024,
    }
    (GENERATED / "board_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    percent = lambda value: f"{100 * value:.2f}%"
    lines = [
        "# Kết quả trực tiếp trên DE10-Lite: OCR ký tự (demo tham chiếu)",
        "",
        f"- Tập mẫu: {result['plate_count']} biển, {len(rows)} ký tự; chọn ngẫu nhiên cố định seed {manifest['seed']} từ validation `train(1)`.",
        f"- SHA-256 checkpoint: `{manifest['checkpoint_sha256']}`.",
        f"- FPGA đúng ký tự: **{char_correct}/{len(rows)} = {percent(result['character_accuracy'])}**; khoảng Wilson 95% {percent(result['character_wilson95'][0])}–{percent(result['character_wilson95'][1])}.",
        f"- FPGA đúng nguyên biển trên crop tham chiếu: **{plate_correct}/{len(manifest['plates'])} = {percent(result['plate_accuracy'])}**; khoảng Wilson 95% {percent(result['plate_wilson95'][0])}–{percent(result['plate_wilson95'][1])}.",
        f"- FPGA trùng mô phỏng số nguyên: {fixed_agree}/{len(rows)} ký tự.",
        f"- Hai lượt chạy liên tiếp trùng dự đoán và chu kỳ: {repeat_agreement}/{len(rows)} mẫu." if repeat_agreement is not None else "",
        "",
        "| Loại biển | Ký tự đúng | Biển đúng hoàn toàn |",
        "|---|---:|---:|",
    ]
    names = {"one_row": "Một hàng ô tô", "two_row_car": "Hai hàng ô tô", "two_row_motorcycle": "Hai hàng xe máy"}
    for kind, stats in type_stats.items():
        lines.append(f"| {names.get(kind, kind)} | {stats['char_correct']}/{stats['chars']} | {stats['plate_correct']}/{stats['plates']} |")
    lines += [
        "",
        "| Thông số trực tiếp | Kết quả |",
        "|---|---:|",
        f"| LE Quartus sau fit | {le_used:,}/{le_total:,} ({percent(le_used/le_total)}) |",
        f"| RAM nội bộ sau fit | {mem_used:,}/{mem_total:,} bit ({percent(mem_used/mem_total)}) |",
        f"| Phần tử nhân 9-bit sau fit | {mult_used}/{mult_total} |",
        f"| Fmax ước lượng sau fit (slow 85°C) | {result['fmax_mhz_slow_85c']:.2f} MHz |",
        f"| Setup slack tại xung 50 MHz (slow 85°C) | {result['setup_slack_ns_slow_85c']:.3f} ns |",
        f"| Chu kỳ đọc từ bộ đếm FPGA | {cycles[0]:,}/ký tự |",
        "| MAC lý thuyết của LeNet-5 đã tỉa Conv2 | 274.200/ký tự |",
        f"| Thời gian tính ở 50 MHz | {result['character_compute_ms']:.3f} ms/ký tự |",
        f"| Ước tính 8 ký tự tuần tự, chỉ lõi OCR | {result['eight_character_compute_ms']:.3f} ms/biển |",
        "",
        "RAM Quartus ở đây **bao gồm cả ảnh mẫu nhúng trong ROM**; không phải chỉ riêng trọng số mạng. Trọng số INT8 có "
        f"{manifest['weight_bytes']:,} byte; {len(rows)*1024:,} byte ảnh thử được đóng trong bitstream.",
        "",
        "Giới hạn: đây là lõi phân loại crop ký tự 32×32. Ảnh đã được cắt theo nhãn trên PC và nhúng sẵn vào bitstream; "
        "FPGA chưa nhận ảnh camera hoặc tự phát hiện/cắt/sắp xếp biển. Tập chỉ 6 biển, chia validation ở mức frame và có thể "
        "trùng danh tính với train; các tỷ lệ trên không chứng minh độ chính xác 95% của hệ thống thực địa. "
        "Thời gian trên không tính tiền xử lý, truyền ảnh, JTAG hay camera. Quartus cảnh báo thiết kế chưa ràng buộc đầy đủ "
        "tất cả đường timing; các số Fmax/slack chỉ áp dụng cho đường xung đã phân tích.",
    ]
    (HERE / "BANG_KET_QUA_TRUC_TIEP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
