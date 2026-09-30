"""Create a public, Vietnamese audit of the six-plate ablation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = ("dense", "conv2_pruned")
GROUPS = (
    ("one_row", "Ô tô một hàng"),
    ("two_row_car", "Ô tô hai hàng"),
    ("two_row_motorcycle", "Xe máy hai hàng"),
)
CLASSES = list("0123456789ABCDEFGHKLMNPSTUVXYZ")


def percent(correct: int, total: int) -> str:
    return f"{100 * correct / total:.2f}% ({correct:,}/{total:,})"


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def top_confusions(matrix: list[list[int]], n: int = 10) -> list[tuple[int, str, str]]:
    errors = [(int(value), CLASSES[i], CLASSES[j])
              for i, row in enumerate(matrix) for j, value in enumerate(row)
              if i != j and value]
    return sorted(errors, key=lambda item: (-item[0], item[1], item[2]))[:n]


def simulation(model_name: str, out: Path) -> dict:
    variant = "dense_rom" if model_name == "dense" else "pruned_rom"
    path = out / "quantized_demo" / variant / "generated" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "characters": manifest["character_count"], "plates": manifest["plate_count"],
        "fp32_correct": manifest["fp32_character_correct"],
        "fixed_correct": manifest["fixed_character_correct"],
        "fp32_plates": manifest["fp32_plate_correct"],
        "fixed_plates": manifest["fixed_plate_correct"],
        "per_character_agreement": sum(item["fp32"] == item["fixed"] for item in manifest["samples"]),
        "weight_bytes_int8": manifest["weight_bytes"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "artifacts/compare_micro6_verified_20260930")
    args = parser.parse_args()
    data = json.loads((args.experiment / "metrics.json").read_text(encoding="utf-8"))
    with (args.experiment / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    def epoch_seconds(phase: str, branch: str | None = None) -> float:
        return sum(float(row["train_seconds"]) + float(row["dev_seconds"])
                   for row in history if row["phase"] == phase
                   and (branch is None or row["branch"] == branch))
    shared_seconds = epoch_seconds("shared_pretraining")
    dense_extra_seconds = epoch_seconds("continuation", "dense")
    pruned_extra_seconds = epoch_seconds("continuation", "conv2_pruned")
    old = json.loads((HERE / "comparison_full.json").read_text(encoding="utf-8"))
    sim = {name: simulation(name, args.experiment) for name in NAMES}
    for name in NAMES:
        if sim[name]["checkpoint_sha256"] != data["models"][name]["checkpoint_sha256"]:
            raise ValueError(f"Quantization simulation uses wrong {name} checkpoint")
    public = {**data, "training_time_breakdown": {
                  "shared_train_plus_development_seconds": shared_seconds,
                  "dense_continuation_train_plus_development_seconds": dense_extra_seconds,
                  "pruned_continuation_train_plus_development_seconds": pruned_extra_seconds,
                  "dense_branch_including_shared_seconds": shared_seconds + dense_extra_seconds,
                  "pruned_branch_including_shared_seconds": shared_seconds + pruned_extra_seconds,
                  "combined_logged_train_plus_development_seconds": shared_seconds + dense_extra_seconds + pruned_extra_seconds,
              }, "quantization_reference_6_plates": sim,
              "prior_hardware_fit_only_not_six_plate_checkpoints": {
                  name: {key: old["variants"][name][key] for key in
                         ("logic_elements", "memory_bits", "embedded_multiplier_9bit",
                          "cycles_per_character", "compute_ms_per_character_at_50mhz",
                          "jtag_total_ms_median", "timing_not_fully_constrained")}
                  for name in ("dense_stream", "pruned_stream")}}
    (HERE / "micro6_public_metrics.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(args.experiment / "history.csv", HERE / "micro6_train_history.csv")
    d = data["models"]["dense"]
    p = data["models"]["conv2_pruned"]
    test_n = d["test"]["plates"]
    if p["test"]["plates"] != test_n:
        raise ValueError("Test populations differ")
    d_ci = wilson(d["test"]["exact_plates"], test_n)
    p_ci = wilson(p["test"]["exact_plates"], test_n)
    lines = [
        "# Thí nghiệm 6 biển: giải thích số liệu và đối chứng dense / pruning",
        "",
        "## Kết luận cần báo cáo trung thực",
        "",
        f"Đã train từ đầu trên **{data['train_plate_count']} biển** thuộc `train(1)` "
        "(2 ô tô một hàng, 2 ô tô hai hàng, 2 xe máy hai hàng), "
        f"tổng **{data['train_character_count']} crop ký tự**. Chỉ có "
        f"**{data['represented_class_count']}/30 lớp** xuất hiện; "
        f"{len(data['missing_classes'])} lớp không có ví dụ dương. "
        f"Kiểm tra cuối trên **{test_n:,} biển / {d['test']['characters']:,} ký tự** "
        "khác nội dung với tập train và tập chọn checkpoint.",
        "",
        "Nguồn ảnh gốc: `data/OCR/OCR/images/train(1)/detection/`; crop dùng để train/test: "
        "`data/independent_chars_train1_structured_channel30_v1/`. "
        "Mục tiêu trên 95,22% **đúng toàn biển** chưa đạt trong thí nghiệm này.",
        "",
        "| Chỉ số | Dense | Pruning cấu trúc 50% kênh Conv2 |",
        "|---|---:|---:|",
        f"| Số tham số | {d['parameters']:,} | {p['parameters']:,} |",
        f"| MAC lý thuyết/ký tự | {d['macs_per_character']:,} | {p['macs_per_character']:,} |",
        f"| Epoch tốt nhất trên development | {d['best_epoch']} | {p['best_epoch']} |",
        f"| Train đúng ký tự | {percent(d['train']['correct_characters'], d['train']['characters'])} | {percent(p['train']['correct_characters'], p['train']['characters'])} |",
        f"| Train đúng cả biển | {percent(d['train']['exact_plates'], d['train']['plates'])} | {percent(p['train']['exact_plates'], p['train']['plates'])} |",
        f"| Development đúng ký tự | {percent(d['development']['correct_characters'], d['development']['characters'])} | {percent(p['development']['correct_characters'], p['development']['characters'])} |",
        f"| Test đúng ký tự | {percent(d['test']['correct_characters'], d['test']['characters'])} | {percent(p['test']['correct_characters'], p['test']['characters'])} |",
        f"| Test đúng cả biển | {percent(d['test']['exact_plates'], d['test']['plates'])} | {percent(p['test']['exact_plates'], p['test']['plates'])} |",
        f"| Test ký tự thuộc lớp chưa từng train | {percent(d['seen_unseen_test']['unseen']['correct'], d['seen_unseen_test']['unseen']['total'])} | {percent(p['seen_unseen_test']['unseen']['correct'], p['seen_unseen_test']['unseen']['total'])} |",
        f"| CPU forward p50, batch 1 | {d['cpu_forward']['1']['median_ms_per_batch']:.4f} ms | {p['cpu_forward']['1']['median_ms_per_batch']:.4f} ms |",
        f"| CPU forward p50, batch 8 | {d['cpu_forward']['8']['median_ms_per_batch']:.4f} ms | {p['cpu_forward']['8']['median_ms_per_batch']:.4f} ms |",
        f"| Train + development, gồm 40 epoch chung | {shared_seconds+dense_extra_seconds:.3f} s | {shared_seconds+pruned_extra_seconds:.3f} s |",
        "",
        f"Thời gian toàn lượt từ nạp dữ liệu đến benchmark: {data['wall_seconds']:.2f} s; "
        f"tổng các thời gian train+development đã ghi chỉ "
        f"{shared_seconds+dense_extra_seconds+pruned_extra_seconds:.2f} s. "
        "Phần lớn thời gian còn lại là mở hàng chục nghìn crop để lập development/test "
        "và các bước đánh giá/benchmark; không được gọi toàn bộ wall-time này là thời gian train. "
        "Thời gian từng epoch rất nhỏ nên chênh lệch giữa hai nhánh dễ bị nhiễu hệ thống.",
        "",
        "Phân bố 49 nhãn train: " + ", ".join(f"`{character}`={count}"
                                             for character, count in data["train_class_counts"].items() if count)
        + ". Các lớp không xuất hiện: `" + "".join(data["missing_classes"]) + "`.",
        "",
        f"Chênh lệch test giữa hai bản chỉ {d['test']['correct_characters'] - p['test']['correct_characters']} "
        f"ký tự trên {d['test']['characters']:,}; pruned hơn dense đúng 1 biển trên {test_n:,}. "
        "Không thể diễn giải chênh lệch biển nhỏ này thành cải thiện độ chính xác do pruning. "
        f"Khoảng Wilson 95% tham khảo cho tỷ lệ biển đúng: dense "
        f"{100*d_ci[0]:.2f}–{100*d_ci[1]:.2f}%, pruned "
        f"{100*p_ci[0]:.2f}–{100*p_ci[1]:.2f}%; các frame lặp lại có thể làm khoảng này quá hẹp.",
        "",
        "Ở lượt train tập lớn trước đây có 180.768 crop train và 20.086 crop validation, "
        "nên không được đặt hai tỷ lệ accuracy cạnh nhau như một A/B test: "
        "lượt này chỉ có 49 crop train và dùng phân hoạch test riêng. "
        "Điều có thể kết luận chắc chắn trong lượt này là **hai kiến trúc được so "
        "trên cùng 6 ảnh train và cùng 1.279 biển test**.",
        "",
        "## Test phân theo loại biển (đúng ký tự; đúng toàn biển)",
        "",
        "| Nhóm | Dense ký tự | Dense biển | Pruned ký tự | Pruned biển |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in GROUPS:
        dg = d["test"]["by_plate_type"][key]
        pg = p["test"]["by_plate_type"][key]
        lines.append(f"| {label} | {percent(dg['correct'], dg['characters'])} | "
                     f"{percent(dg['exact'], dg['plates'])} | "
                     f"{percent(pg['correct'], pg['characters'])} | "
                     f"{percent(pg['exact'], pg['plates'])} |")
    lines += [
        "",
        "## Vì sao chọn các chỉ số này?",
        "",
        "- **Đúng toàn biển trên test độc lập** là chỉ số gần mục tiêu đồ án nhất: "
        "đầu ra phải trùng mọi ký tự. Đúng từng ký tự giúp chẩn đoán OCR nhưng "
        "không thay được đúng toàn biển.",
        "- **Từng nhóm biển** kiểm tra mô hình có làm tốt cả một/hai hàng và xe máy, "
        "tránh trung bình chung che nhóm yếu.",
        "- **Train–development–test** tách việc học thuộc, chọn checkpoint và đánh giá cuối. "
        "Khoảng cách train–test lớn là dấu hiệu tổng quát hóa kém.",
        "- **Ma trận nhầm lẫn và lớp chưa thấy** tìm lỗi dữ liệu/cắt ký tự; "
        "không có nhãn train cho một lớp thì không thể kỳ vọng học lớp đó đáng tin cậy.",
        "- **MAC/tham số** cho biết độ phức tạp lý thuyết, còn **LE, RAM bit, DSP, "
        "chu kỳ, timing** sau Quartus và thời gian truyền đo trên kit mới cho biết "
        "giới hạn phần cứng. MAC giảm không đồng nghĩa tổng thời gian giảm cùng tỷ lệ.",
        "",
        "## Quy trình train và lý do của từng bước",
        "",
        "1. Đọc `train(1)` và dùng crop 32×32 đã chuẩn bị, kiểm tra trực quan 6 ảnh gốc. "
        "Một file gắn nhãn ô tô hai hàng thực chất hiện biển một hàng, nên đã "
        "thay bằng ảnh hai hàng đúng trước train; tập gốc không bị sửa/xóa.",
        "2. Chọn đúng 2 biển/nhóm theo seed cố định, ghi riêng danh sách nguồn cục bộ. "
        "Chỉ dùng 6 biển này để cập nhật trọng số, không nạp checkpoint từng học tập lớn.",
        "3. Chia phần `val` theo **chuỗi ký tự biển** bằng hash cố định thành development "
        "và test; loại chuỗi trùng 6 biển train. Chọn checkpoint theo accuracy ký tự "
        "trên development; chỉ tính/chủ động xem kết quả test sau khi chọn checkpoint.",
        "4. Chuẩn hóa crop về [-1,1]; lúc train xoay ngẫu nhiên ±6° ở 55% lượt "
        "để giảm phụ thuộc góc rất nhỏ. Không xoay ảnh development/test.",
        f"5. Cùng LeNet-5 30 đầu ra, cross-entropy, AdamW, batch {data['batch_size']}, "
        f"learning rate {data['lr']}, weight decay {data['weight_decay']}. "
        f"Train dense chung {data['shared_epochs']} epoch, sau đó dense thêm "
        f"{data['continuation_epochs']} epoch; bản kia cắt nguyên 8/16 kênh "
        "Conv2 và các cột FC1 tương ứng rồi train tiếp cùng số epoch. Conv1 giữ "
        "6 kênh. Mỗi epoch chỉ có 4 mini-batch từ 49 crop, nên cần nhiều "
        "epoch hơn lượt tập lớn mới đủ bước cập nhật; số epoch khác cũng là lý "
        "do không so trực tiếp thời gian train hai thí nghiệm. Mỗi epoch của "
        "hai nhánh dùng cùng seed trộn/augment.",
        "6. Chấm train, development và test; đo CPU forward lặp 500 lần theo batch 1/8, "
        "tính MAC/tham số. Giả lập lượng tử INT8 trên cùng 6 biển tham chiếu để "
        "tách lỗi train khỏi lỗi lượng tử; không nhận giả lập là phép đo FPGA.",
        "",
        "## Vì sao số liệu chênh lệch?",
        "",
        f"- **Thiếu và lệch lớp:** {data['train_character_count']} nhãn train chỉ phủ "
        f"{data['represented_class_count']}/30 lớp; test có "
        f"{d['seen_unseen_test']['unseen']['total']} ký tự thuộc lớp chưa train, "
        "cả hai bản đều đúng 0. Các lớp đã thấy cũng chỉ đạt khoảng 59%, "
        "nên thiếu lớp chưa phải nguyên nhân duy nhất.",
        "- **Học thuộc 6 biển:** cả hai đúng 47/49 crop train nhưng chỉ khoảng "
        "57% crop test. Một số crop nguồn rất nhỏ/mờ: ký tự B trên biển ô tô hai "
        "hàng đã xem trực tiếp trông gần như 8. Nhãn từ tên file và phân nhóm "
        "thư mục không bảo đảm từng crop đúng; vì vậy cần duyệt thủ công nếu muốn "
        "đánh giá chất lượng thực địa.",
        "- **Đúng toàn biển thấp hơn đúng ký tự:** chỉ cần sai một ký tự là cả biển "
        "sai. Không nhân đơn giản accuracy ký tự lên lũy thừa vì lỗi giữa ký tự/biển "
        "có tương quan và chiều dài biển không cố định.",
        "- **Pruning đổi năng lực mô hình:** số tham số giảm "
        f"{100*(1-p['parameters']/d['parameters']):.2f}% và MAC giảm "
        f"{100*(1-p['macs_per_character']/d['macs_per_character']):.2f}%; "
        "accuracy có thể tăng/giảm vài mẫu do tối ưu và nhiễu chọn mẫu. "
        "Không được gọi 1 biển chênh lệch là pruning cải thiện accuracy.",
        "- **CPU, FPGA và truyền là ba thời gian khác nhau:** p50 batch 1 CPU "
        "gần như bằng nhau do overhead chi phối, dù MAC khác. Bộ RTL MAC tuần tự "
        "đã đo trên kit ở lần trước giảm từ 870.198 xuống 577.998 chu kỳ/ký tự "
        "(nhanh lõi 1,506×); qua USB-Blaster/JTAG vẫn khoảng 1 giây/ký tự. "
        "Đây là đo của cùng **kiến trúc** với checkpoint tập lớn, không phải phép "
        "đo lại trên kit cho trọng số train từ 6 biển.",
        "",
        "## Giả lập INT8 và giới hạn phần cứng",
        "",
        "| Kiểm tra trên cùng 6 biển tham chiếu (không nằm trong 6 biển train) | Dense | Pruned |",
        "|---|---:|---:|",
        f"| FP32 đúng ký tự | {sim['dense']['fp32_correct']}/50 | {sim['conv2_pruned']['fp32_correct']}/50 |",
        f"| INT8 đúng ký tự | {sim['dense']['fixed_correct']}/50 | {sim['conv2_pruned']['fixed_correct']}/50 |",
        f"| FP32–INT8 dự đoán giống nhau | {sim['dense']['per_character_agreement']}/50 | {sim['conv2_pruned']['per_character_agreement']}/50 |",
        f"| Đúng cả biển | {sim['dense']['fixed_plates']}/6 | {sim['conv2_pruned']['fixed_plates']}/6 |",
        f"| Trọng số INT8 | {sim['dense']['weight_bytes_int8']:,} byte | {sim['conv2_pruned']['weight_bytes_int8']:,} byte |",
        "",
        "Bản FP32 và giả lập INT8 ra đúng cùng từng ký tự trong 50 crop này, "
        "nên lượng tử **không giải thích** độ chính xác thấp của lượt thử này. "
        "Chưa nạp hai checkpoint 6-biển xuống kit; số tài nguyên/tốc độ Quartus "
        "bên dưới là **tham chiếu kiến trúc đã đo với checkpoint trước**, không "
        "được trình bày là phép đo của trọng số mới.",
        "",
        "| Tài nguyên/tốc độ Quartus trước đây, cùng RTL | Dense stream | Pruned stream |",
        "|---|---:|---:|",
    ]
    for label, key, subkey in (
        ("Logic elements", "logic_elements", "used"),
        ("RAM bit", "memory_bits", "used"),
        ("DSP 9-bit", "embedded_multiplier_9bit", "used"),
    ):
        left = old["variants"]["dense_stream"][key][subkey]
        right = old["variants"]["pruned_stream"][key][subkey]
        lines.append(f"| {label} | {left:,} | {right:,} |")
    for label, key, suffix in (
        ("Chu kỳ/ký tự", "cycles_per_character", ""),
        ("Lõi ms/ký tự @50MHz", "compute_ms_per_character_at_50mhz", " ms"),
        ("JTAG toàn lượt p50/ký tự", "jtag_total_ms_median", " ms"),
    ):
        left = old["variants"]["dense_stream"][key]
        right = old["variants"]["pruned_stream"][key]
        number_format = ",d" if key == "cycles_per_character" else ",.3f"
        lines.append(f"| {label} | {format(left, number_format)}{suffix} | "
                     f"{format(right, number_format)}{suffix} |")
    lines += [
        "",
        "Nạp lại checkpoint 6-biển **chưa cần** để kết luận về hiệu năng kiến trúc: "
        "số kênh, phép MAC và RTL không đổi, trong khi accuracy test quá thấp "
        "để dùng làm demo. Muốn xác nhận bit-exact của trọng số mới trên FPGA "
        "thì phải tổng hợp/nạp **bitstream riêng**, chạy lại JTAG và ghi số liệu "
        "mới; không được lấy số đo cũ gán cho checkpoint mới.",
        "",
        "## Nhầm lẫn lớn nhất trên test (nhãn thật → dự đoán)",
        "",
        "| Dense | Số lần | Pruned | Số lần |",
        "|---|---:|---|---:|",
    ]
    dense_errors = top_confusions(d["test"]["confusion_matrix"])
    pruned_errors = top_confusions(p["test"]["confusion_matrix"])
    for left, right in zip(dense_errors, pruned_errors):
        lines.append(f"| {left[1]}→{left[2]} | {left[0]:,} | {right[1]}→{right[2]} | {right[0]:,} |")
    lines += [
        "",
        "Chi tiết 30×30 nhầm lẫn và số đúng/tổng nằm trong `micro6_public_metrics.json`; "
        "loss, accuracy và thời gian từng epoch trong `micro6_train_history.csv`. "
        "Danh sách ảnh, crop, nhãn và checkpoint chỉ lưu cục bộ trong `artifacts/`, "
        "không đẩy lên GitHub.",
        "",
        "**Giới hạn quan trọng:** test vẫn là các crop ký tự với ranh giới/số ký tự "
        "biết từ nhãn, chưa phải ảnh camera → tìm biển → cắt → OCR trên FPGA. "
        "Một số frame/biển trùng danh tính trong nguồn; khoảng tin cậy chỉ tham khảo. "
        "Timing Quartus trước đây cũng chưa được ràng buộc đầy đủ. Chưa đo công suất "
        "điện hoặc tỷ lệ bỏ qua ảnh mờ/nghiêng trong thí nghiệm 6 biển.",
    ]
    (HERE / "BAO_CAO_6_BIEN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(HERE / "BAO_CAO_6_BIEN.md"),
                      "test_plates": test_n,
                      "dense_test": [d["test"]["correct_characters"], d["test"]["exact_plates"]],
                      "pruned_test": [p["test"]["correct_characters"], p["test"]["exact_plates"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
