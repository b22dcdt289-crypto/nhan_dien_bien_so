# Báo cáo tài nguyên mô hình LeNet-5 Structured

Các số liệu dưới đây được đo trực tiếp từ checkpoint
`lenet5_ocr_structured_pruned25_final.pt`.

| Thành phần | Kích thước ước tính |
|---|---:|
| Số tham số | 36.443 |
| Trọng số INT8 | khoảng 0,035 MiB |
| MAC cho một ký tự | 233.338 |
| MAC cho biển một hàng (8 ký tự) | khoảng 1,87 triệu |
| MAC cho biển hai hàng (16 ký tự) | khoảng 3,73 triệu |
| Accuracy validation | 95,22% |

## Giả định

- Một biển một hàng được tính với 8 ký tự.
- Một biển hai hàng được tính với 16 ký tự, tức hai lần chi phí một hàng.
- Bảng chỉ tính bộ OCR LeNet-5, chưa cộng YOLOv5 detector.
- YOLOv5 detector chạy riêng có chi phí lớn hơn nhiều và cần báo cáo riêng nếu triển khai toàn pipeline.
- INT8 ở đây là kích thước trọng số lý thuyết; cần quantization/calibration khi triển khai FPGA.
