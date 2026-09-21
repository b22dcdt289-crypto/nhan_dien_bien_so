# Báo cáo tài nguyên mô hình LeNet-5 Structured

Các số liệu dưới đây được đo trực tiếp từ checkpoint
`lenet5_ocr_structured_pruned25_final.pt`.

| Thành phần | Kích thước ước tính |
|---|---:|
| Số tham số | 36.443 |
| Trọng số INT8 | khoảng 0,035 MiB |
| MAC cho một ký tự | 233.338 |
| MAC cho biển một hàng (8 ký tự) | khoảng 1,87 triệu |
| MAC cho biển hai hàng (8 ký tự) | khoảng 1,87 triệu |
| Accuracy validation | 95,22% |

## Giả định

- Một biển một hàng được tính với 8 ký tự.
- Biển hai hàng vẫn được tính với 8 ký tự; hai hàng chỉ thay đổi bố cục và thứ tự đọc.
- Bảng chỉ tính bộ OCR LeNet-5, chưa cộng YOLOv5 detector.
- YOLOv5 detector chạy riêng có chi phí lớn hơn nhiều và cần báo cáo riêng nếu triển khai toàn pipeline.
- INT8 ở đây là kích thước trọng số lý thuyết; cần quantization/calibration khi triển khai FPGA.
