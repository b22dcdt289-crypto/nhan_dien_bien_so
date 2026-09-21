# Báo cáo LeNet-5 pruning 50%

Checkpoint: `lenet5_ocr_pruned50_final.pt`

| Thành phần | Kết quả |
|---|---:|
| Kiểu pruning | Global unstructured magnitude pruning 50% |
| Accuracy validation tốt nhất | **96,79%** |
| Số tham số toàn mô hình | 63.916 |
| Trọng số INT8 nếu giữ đủ tensor | khoảng 0,061 MiB |
| Trọng số khác 0 | 31.827 / 63.654 = 50% |
| MAC đầy đủ, một ký tự | 418.704 |
| MAC hiệu dụng bỏ qua weight = 0, một ký tự | khoảng 271.695 |
| MAC hiệu dụng, biển một hàng 8 ký tự | khoảng 2,17 triệu |
| MAC hiệu dụng, biển hai hàng 8 ký tự | khoảng 2,17 triệu |

Biển một hàng và biển hai hàng đều được tính là 8 ký tự; hai hàng chỉ thay
đổi bố cục và thứ tự sắp xếp ký tự.

## Lưu ý triển khai FPGA

Pruning 50% dạng unstructured đạt yêu cầu accuracy nhưng không làm nhỏ các
tensor Conv/Linear. Vì vậy MAC hình dạng đầy đủ vẫn là 418.704 và tài nguyên
DSP/logic chỉ giảm nếu bộ tăng tốc có thể bỏ qua weight bằng 0. Nếu không có
sparse engine, cần structured pruning để giảm tài nguyên vật lý; structured
25% trước đó có 233.338 MAC/ký tự nhưng accuracy thấp hơn trên dữ liệu
`train(1)` tự tách crop.

Accuracy 96,79% là accuracy từng ký tự trên tập OCR có bounding-box nhãn,
không phải accuracy đọc đúng toàn bộ chuỗi biển số end-to-end.
