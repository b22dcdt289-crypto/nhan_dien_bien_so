# Các checkpoint structured pruning

Các checkpoint dưới đây đã được lưu tại `artifacts/` ở thư mục gốc repository. Đây là các thử nghiệm LeNet dùng ảnh ký tự 32 × 32 và kernel 5 × 5, khác model K3 đầu vào 28 × 28 trong `best.pt`. Không nạp trực tiếp chúng vào bitstream FPGA hiện tại; việc triển khai cần chuyển đổi và kiểm chứng lại mô hình, trọng số và RTL.

| Checkpoint | Cấu trúc Conv1 / Conv2 / FC1 / FC2 | Kết quả |
|---|---|---|
| [Structured pruning mục tiêu 25%](../../artifacts/lenet5_ocr_structured_pruned25_final.pt) | 4 / 12 / 90 / 63 | Theo báo cáo `structured_pruning_metrics.json`: 95,22% ký tự validation, 233.338 MAC/ký tự; tỷ lệ giảm MAC 44,23%. |
| [Structured50 30 lớp](../../artifacts/lenet5_structured50_channel30_matched_v1.pt) | 4 / 10 / 100 / 70 | 91,61% ký tự và 81,16% toàn biển trên 2.505 biển validation đã chuẩn bị; 212.500 MAC/ký tự. |
| [Conv2 pruning + distillation 30 lớp](../../artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt) | 6 / 8 / 120 / 84 | 92,35% ký tự và 84,23% toàn biển trên validation đã chuẩn bị; 274.200 MAC/ký tự. |

Tên mục tiêu 25% không có nghĩa tổng MAC giảm 25%. Hai thí nghiệm 30 lớp có báo cáo và dữ liệu đối chiếu riêng: [structured50](../../artifacts/channel30_matched_structured_comparison.md) và [Conv2 + distillation](../../artifacts/channel30_cost_hard_conv2_distill_report.md). Kết quả validation trên crop đã chuẩn bị không phải độ chính xác toàn tuyến camera hoặc số đo FPGA.

Mã liên quan: [structured pruning](../../structured_prune_lenet5.py), [huấn luyện structured](../../train_independent_structured.py), [Conv2 distillation](../../train_cost_sensitive_conv2_distill.py), [LeNet gốc](../../train_lenet5.py). Dữ liệu huấn luyện đầy đủ không nằm trong repository.
