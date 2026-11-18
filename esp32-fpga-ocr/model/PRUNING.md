# Các checkpoint structured pruning

Các checkpoint dưới đây đã được lưu tại `artifacts/` ở thư mục gốc repository. Đây là các thử nghiệm LeNet dùng ảnh ký tự 32 × 32 và kernel 5 × 5, khác model K3 đầu vào 28 × 28 trong `best.pt`. Không nạp trực tiếp chúng vào bitstream FPGA hiện tại; việc triển khai cần chuyển đổi và kiểm chứng lại mô hình, trọng số và RTL.

| Checkpoint | Cấu trúc Conv1 / Conv2 / FC1 / FC2 | Kết quả |
|---|---|---|
| [Structured pruning mục tiêu 25%](../../artifacts/lenet5_ocr_structured_pruned25_final.pt) | 4 / 12 / 90 / 63 | Theo báo cáo `structured_pruning_metrics.json`: 95,22% ký tự validation, 233.338 MAC/ký tự; tỷ lệ giảm MAC 44,23%. |
| [Structured50 30 lớp](../../artifacts/lenet5_structured50_channel30_matched_v1.pt) | 4 / 10 / 100 / 70 | 91,61% ký tự và 81,16% toàn biển trên 2.505 biển validation đã chuẩn bị; 212.500 MAC/ký tự. |
| [Conv2 pruning + distillation 30 lớp](../../artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt) | 6 / 8 / 120 / 84 | 92,35% ký tự và 84,23% toàn biển trên validation đã chuẩn bị; 274.200 MAC/ký tự. |

Tên mục tiêu 25% không có nghĩa tổng MAC giảm 25%. Hai thí nghiệm 30 lớp có báo cáo và dữ liệu đối chiếu riêng: [structured50](../../artifacts/channel30_matched_structured_comparison.md) và [Conv2 + distillation](../../artifacts/channel30_cost_hard_conv2_distill_report.md). Kết quả validation trên crop đã chuẩn bị không phải độ chính xác toàn tuyến camera hoặc số đo FPGA.

Mã liên quan: [structured pruning](../../structured_prune_lenet5.py), [huấn luyện structured](../../train_independent_structured.py), [Conv2 distillation](../../train_cost_sensitive_conv2_distill.py), [LeNet gốc](../../train_lenet5.py). Dữ liệu huấn luyện đầy đủ không nằm trong repository.

## Chạy model structured50 trên máy tính

[`lenet5_structured50_30class.ts`](lenet5_structured50_30class.ts) là model TorchScript đã gộp kiến trúc với trọng số đã train, không cần tạo lại class model để chạy. Đầu vào tensor có kích thước `N × 1 × 32 × 32`, kiểu float, giá trị từ −1 đến 1; đầu ra gồm 30 logit theo thứ tự `0123456789ABCDEFGHKLMNPSTUVXYZ`. Ảnh đầu vào cần là từng ký tự đã tách và được chuẩn hóa tương tự dữ liệu train. Điểm softmax là điểm mô hình, không mặc định là tỷ lệ chính xác.

Với Python có `torch`, `numpy`, `Pillow`:

```bash
python esp32-fpga-ocr/model/infer_pruned_character.py path/to/glyph_32x32.png
```

Để thử xử lý một ảnh vùng biển đã cắt bằng pipeline máy tính hiện có tại gốc repository, cài thêm `opencv-python` rồi chạy từ thư mục gốc:

```bash
python recognize_independent.py path/to/plate.png --plate-crop --model artifacts/lenet5_structured50_channel30_matched_v1.pt
```

Lệnh vùng biển dùng checkpoint cùng kiến trúc và thực hiện tách ký tự trước suy luận. Tách ảnh có thể thất bại, nhất là biển hai hàng. Không có bước truyền đến FPGA trong hai lệnh này. Model TorchScript đã được so sánh 240 logit trên 8 tensor thử với model PyTorch tương ứng sau khi xuất và nạp lại; các logit khớp chính xác. Một crop ký tự validation kiểm tra chạy được nhưng dự đoán sai, nên một lần chạy không chứng minh độ chính xác.

## Thử thuật toán quyết định của model mới

```bash
python esp32-fpga-ocr/model/recognize_pruned_policy.py path/to/plate_roi.png
```

Chương trình tách ký tự, suy luận model structured50, báo đầu ra gốc, chuỗi ứng viên, top-3 và trạng thái nhận/từ chối; không nhận nhãn chuẩn khi suy luận. Quy tắc thử nghiệm hỗ trợ chuỗi 8 hoặc 9 ký tự theo phạm vi phép thử mới; chuỗi khác bị từ chối. Điều kiện cho phép thay đổi ký tự cần có đủ bằng chứng từ model, không ép theo định dạng.

Ngưỡng và temperature được chuyển từ model 31 lớp mới nhất, **chưa được hiệu chuẩn cho model pruning**. Cần dùng tập calibration tách biệt để hiệu chuẩn và tập test độc lập để đo lại; các điểm không phải xác suất đúng đã được đảm bảo. Model pruning 30 lớp không có lớp `O`, nên không thể áp dụng phép gộp điểm `0/O` hoặc khẳng định phân biệt được cặp này. Thử nghiệm chạy trên PC; chưa đưa thuật toán hay model xuống FPGA.
