# Cost-sensitive hard mining và Conv2-only pruning/distillation

## Tóm tắt

Thí nghiệm tiếp tục từ checkpoint dense 30 lớp đã train trên cùng train(1).
Teacher được fine-tune với class-cost inverse-sqrt và hard-example mining;
sau đó student giữ nguyên, đóng băng toàn bộ Conv1, chỉ giảm Conv2 từ 16 xuống 8 kênh
và học từ teacher bằng supervised loss kết hợp KL knowledge distillation.

| Mô hình | Accuracy ký tự | Khớp nguyên biển | MAC/ký tự | Tham số |
|---|---:|---:|---:|---:|
| dense_previous | 92.24% | 83.75% | 418,200 | 63,406 |
| structured50_previous | 91.61% | 81.16% | 212,500 | 35,414 |
| cost_sensitive_teacher | 92.56% | 84.87% | 418,200 | 63,406 |
| conv2_pruned_student | 92.35% | 84.23% | 274,200 | 38,198 |

### Exact-plate theo nhóm validation

| Mô hình | Một hàng | Ô tô hai hàng | Xe máy hai hàng |
|---|---:|---:|---:|
| dense_previous | 93.06% | 85.20% | 41.35% |
| structured50_previous | 91.74% | 81.33% | 36.07% |
| cost_sensitive_teacher | 93.48% | 87.00% | 43.99% |
| conv2_pruned_student | 93.20% | 86.03% | 42.52% |

## Thiết lập train

Nguồn train(1); train/validation có 180,768/20,086 crop ký tự và 2,505 biển validation.
Teacher fine-tune 8 epoch từ checkpoint dense cũ; student distill 8 epoch. Batch=256, AdamW lr=0.0002, weight_decay=1e-05, seed=20260927.
Thời gian epoch tích lũy: teacher 135.5s; student 130.0s (không gồm nạp cache/benchmark).

## Phân rã lỗi trên validation trước tối ưu

Mỗi sample là một biển; OCR chỉ phân loại các crop ký tự đã được tạo trước.
Vì vậy mismatch ký tự được đo độc lập với lỗi phát hiện/segmentation khi chạy tự do.
Bảng cá nhân theo từng biển nằm trong file local-only channel30_error_cases_local.csv;
file này có nội dung biển số và không được đẩy lên GitHub.

| Mô hình cũ | Biển sai | Exact accuracy | Histogram số ký tự sai trên mỗi biển |
|---|---:|---:|---|
| dense_previous | 407/2505 | 83.75% | {'0': 2098, '1': 147, '2': 39, '3+': 221} |
| structured50_previous | 472/2505 | 81.16% | {'0': 2033, '1': 203, '2': 43, '3+': 226} |

Chi tiết theo nhóm (số biển, số biển lỗi, error rate):

### dense_previous: lỗi theo nhóm

| Nhóm | Biển | Sai biển | Tỷ lệ sai |
|---|---:|---:|---:|
| Loại biển: one_row | 1441 | 100 | 6.94% |
| Loại biển: two_row_car | 723 | 107 | 14.80% |
| Loại biển: two_row_motorcycle | 341 | 200 | 58.65% |
| Độ dài nhãn: 10 | 2 | 2 | 100.00% |
| Độ dài nhãn: 7 | 216 | 39 | 18.06% |
| Độ dài nhãn: 8 | 2029 | 215 | 10.60% |
| Độ dài nhãn: 9 | 258 | 151 | 58.53% |

- Confidence trung bình ở ký tự dự đoán đúng: 0.9541.
- Confidence trung bình ở ký tự dự đoán sai: 0.4289.
- Cặp nhầm ký tự thường gặp nhất: [{'pair': '2→9', 'count': 65}, {'pair': '3→9', 'count': 49}, {'pair': '6→9', 'count': 48}, {'pair': '6→3', 'count': 40}, {'pair': '2→1', 'count': 40}, {'pair': '1→9', 'count': 39}, {'pair': '3→1', 'count': 37}, {'pair': '7→9', 'count': 33}, {'pair': '0→9', 'count': 32}, {'pair': '8→9', 'count': 28}].
- Tỷ lệ lỗi theo blur, góc nghiêng, perspective-correction và từng vị trí có trong JSON metrics.

### structured50_previous: lỗi theo nhóm

| Nhóm | Biển | Sai biển | Tỷ lệ sai |
|---|---:|---:|---:|
| Loại biển: one_row | 1441 | 119 | 8.26% |
| Loại biển: two_row_car | 723 | 135 | 18.67% |
| Loại biển: two_row_motorcycle | 341 | 218 | 63.93% |
| Độ dài nhãn: 10 | 2 | 2 | 100.00% |
| Độ dài nhãn: 7 | 216 | 57 | 26.39% |
| Độ dài nhãn: 8 | 2029 | 246 | 12.12% |
| Độ dài nhãn: 9 | 258 | 167 | 64.73% |

- Confidence trung bình ở ký tự dự đoán đúng: 0.9482.
- Confidence trung bình ở ký tự dự đoán sai: 0.4283.
- Cặp nhầm ký tự thường gặp nhất: [{'pair': '2→1', 'count': 68}, {'pair': '9→1', 'count': 65}, {'pair': '3→1', 'count': 60}, {'pair': '9→2', 'count': 50}, {'pair': '6→3', 'count': 46}, {'pair': '1→2', 'count': 41}, {'pair': '0→1', 'count': 37}, {'pair': '7→1', 'count': 35}, {'pair': '6→1', 'count': 35}, {'pair': '8→1', 'count': 31}].
- Tỷ lệ lỗi theo blur, góc nghiêng, perspective-correction và từng vị trí có trong JSON metrics.

### Error theo blur bucket

| Nhóm | Biển | Dense lỗi | Dense error rate | Structured lỗi | Structured error rate |
|---|---:|---:|---:|---:|---:|
| 200–499 | 248 | 25 | 10.08% | 32 | 12.90% |
| 80–199 | 56 | 12 | 21.43% | 13 | 23.21% |
| <80 | 5 | 1 | 20.00% | 1 | 20.00% |
| ≥500 | 2196 | 369 | 16.80% | 426 | 19.40% |

### Error theo góc ước lượng bucket

| Nhóm | Biển | Dense lỗi | Dense error rate | Structured lỗi | Structured error rate |
|---|---:|---:|---:|---:|---:|
| 0–2° | 2477 | 403 | 16.27% | 469 | 18.93% |
| >2–5° | 16 | 3 | 18.75% | 2 | 12.50% |
| >5–8° | 3 | 1 | 33.33% | 1 | 33.33% |
| >8° | 9 | 0 | 0.00% | 0 | 0.00% |

### Error theo perspective correction

| Nhóm | Biển | Dense lỗi | Dense error rate | Structured lỗi | Structured error rate |
|---|---:|---:|---:|---:|---:|
| False | 1046 | 206 | 19.69% | 242 | 23.14% |
| True | 1459 | 201 | 13.78% | 230 | 15.76% |


Các thống kê blur/góc là tương quan trong validation, không chứng minh nguyên nhân lỗi.
Tên/nhãn biển trong CSV cá nhân chỉ hỗ trợ người dùng kiểm tra lại ảnh và segmentation.

## Loss và distillation

- Cost-sensitive: mỗi lớp có trọng số tỷ lệ nghịch căn bậc hai tần suất train,
  chuẩn hóa về trung bình gần 1 và giới hạn [0,5; 3,0].
- Hard mining: trong mỗi batch lấy top 25% mẫu có weighted CE cao nhất;
  loss = 0.50×mean(all) + 0.50×mean(hardest).
- Student objective: 0.70×cost-sensitive OHEM +
  0.30×T² KL(student/T, teacher/T), T=2.0.
- Conv1 weights/bias được copy chính xác từ teacher và frozen; Conv2 output được rút 16→8;
  downstream FC1 input columns được rút tương ứng để tensor thực sự nhỏ lại.

## MAC và tài nguyên

| Mô hình | Kênh (Conv1, Conv2, FC1, FC2) | MAC/ký tự | Thay đổi MAC vs dense | Tham số | FP32 weights thô |
|---|---|---:|---:|---:|---:|
| dense_previous | (6, 16, 120, 84) | 418,200 | +0.00% | 63,406 | 247.68 KiB |
| structured50_previous | (4, 10, 100, 70) | 212,500 | -49.19% | 35,414 | 138.34 KiB |
| cost_sensitive_teacher | (6, 16, 120, 84) | 418,200 | +0.00% | 63,406 | 247.68 KiB |
| conv2_pruned_student | (6, 8, 120, 84) | 274,200 | -34.43% | 38,198 | 149.21 KiB |

### MAC cho toàn chuỗi sau segmentation

| Số ký tự | Dense cũ | Structured-50 cũ | Teacher | Conv2 student |
|---:|---:|---:|---:|---:|
| 7 | 2,927,400 | 1,487,500 | 2,927,400 | 1,919,400 |
| 8 | 3,345,600 | 1,700,000 | 3,345,600 | 2,193,600 |
| 9 | 3,763,800 | 1,912,500 | 3,763,800 | 2,467,800 |
| 10 | 4,182,000 | 2,125,000 | 4,182,000 | 2,742,000 |

Conv2-only student giảm MAC 34.43% so với dense cũ; do giữ nguyên Conv1 và FC hidden widths, nó có MAC cao hơn 29.04% so với structured-50 cũ.
Công thức student mỗi ký tự: Conv1 6×25×28×28 = 117.600; Conv2 8×6×25×10×10 = 120.000; FC1 8×25×120 = 24.000; FC2 120×84 = 10.080; FC3 84×30 = 2.520; tổng = 274.200 MAC.

## Latency CPU

| Batch ký tự | Dense cũ p50 ms | Structured-50 cũ p50 ms | Teacher p50 ms | Conv2 student p50 ms | Student speedup vs dense | Student speedup vs struct-50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.17740 | 0.16700 | 0.17710 | 0.17180 | 1.033× | 0.972× |
| 7 | 0.40410 | 0.33770 | 0.40570 | 0.37500 | 1.078× | 0.901× |
| 8 | 0.44860 | 0.37070 | 0.44815 | 0.41430 | 1.083× | 0.895× |
| 9 | 0.47910 | 0.39430 | 0.47910 | 0.44300 | 1.081× | 0.890× |
| 10 | 0.51390 | 0.41740 | 0.51350 | 0.47160 | 1.090× | 0.885× |

Batch 8: student latency giảm 7.65% so với dense cũ; p95 của dense/student là 0.80939/0.75062 ms.
MAC ratio dense/student = 1.525× nếu workload compute-bound; đây chỉ là tỷ lệ phép toán, không phải dự báo timing FPGA. Student có p50 latency cao hơn structured-50 cũ 11.76% ở batch 8.
Benchmark dùng 250 lượt warm-up và 2.500 lượt đo mỗi mô hình/batch; thứ tự mô hình được xáo trộn trong từng vòng đo.

Forward latency đo PyTorch CPU, 1 thread, 250 warm-up + 2.500 lượt đo, thứ tự model xáo trộn,
cùng tensor N×1×32×32; không gồm đọc ảnh,
perspective correction, phân hàng, segmentation hoặc giải mã. Speedup thực tế phụ thuộc runtime.

## Giới hạn

- Validation được chia theo frame chứ không group theo identity; frame gần trùng có thể xuất hiện ở hai split.
- Validation dùng crop và số crop theo nhãn tên file; không phải full-frame end-to-end.
- Cost-sensitive OHEM làm thay đổi objective và có thể cải thiện lớp hiếm nhưng làm giảm accuracy tổng thể;
  cần đọc cả exact-plate và per-class metrics trước khi chọn model.
- CPU benchmark không dự báo trực tiếp DE10-Lite. Cần tổng hợp hai phiên bản trên FPGA
  rồi so ALM/DSP/RAM/fMAX, cycles và timing slack.

Metrics đầy đủ, class weights, histories, per-character confusion và số liệu benchmark nằm cạnh báo cáo.
Bảng per-plate có tên/nhãn được giữ local-only; aggregate error breakdown không chứa định danh biển.

## Reproduce

Lệnh dưới đây cần data đã chuẩn bị và dùng tên output mới vì trainer không ghi đè checkpoint:

```powershell
.venv\Scripts\python.exe train_cost_sensitive_conv2_distill.py --epochs 8 --teacher-output artifacts/repro_cost_teacher.pt --student-output artifacts/repro_conv2_student.pt --metrics artifacts/repro_metrics.json --history-csv artifacts/repro_history.csv --report artifacts/repro_report.md --error-private-csv artifacts/repro_local_error_plates.csv --error-aggregate artifacts/repro_error_aggregate.json --speed-json artifacts/repro_speed.json --speed-raw artifacts/repro_speed_raw.csv --student-confusion artifacts/repro_confusion.csv
```
