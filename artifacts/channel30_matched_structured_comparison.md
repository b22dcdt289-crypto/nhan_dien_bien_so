# So sánh LeNet-5 dense và structured channel pruning trên train(1)

## Kết luận ngắn

Trong phép thử ghép cặp này, structured channel pruning làm số MAC giảm từ
418.200 xuống 212.500 MAC/ký tự (giảm 49,19%) và số tham số giảm 44,15%. Trên
CPU máy phát triển, median latency của batch 8 ký tự giảm từ 0,43905 ms xuống
0,35775 ms (nhanh hơn 1,227×, giảm latency 18,52%). Đổi lại, accuracy ký tự
giảm 0,63 điểm phần trăm và exact-plate accuracy giảm 2,59 điểm phần trăm.

Đây là kết quả đo CPU của forward pass LeNet, không phải tốc độ toàn pipeline
và không phải kết quả trên FPGA DE10-Lite.

## Thiết lập thí nghiệm

- Nguồn: data/OCR/OCR/images/train(1)/detection.
- Dùng bảng chữ 30 lớp: chữ số 0–9 và các chữ cái A, B, C, D, E, F, G, H,
  K, L, M, N, P, S, T, U, V, X, Y, Z.
- Cùng ảnh/crop đã chuẩn bị, cùng train/validation split, cùng seed 20260926,
  batch size 256, AdamW (learning rate 5e-4, weight decay 1e-5), cùng
  tiền xử lý clahe_sharp và augmentation xoay crop ngẫu nhiên trong khoảng
  ±6°.
- Cả hai nhánh có 10 lượt đi qua tập train. Dense train từ đầu đủ 10 epoch.
  Nhánh structured khởi đầu bằng dense trong 5 epoch; sau đó xếp hạng kênh
  bằng tổng trị tuyệt đối trọng số (L1), chuyển trọng số của các kênh được
  chọn sang mạng nhỏ, rồi fine-tune 5 epoch còn lại. Vì vậy số lượt xem dữ
  liệu được giữ ngang nhau, nhưng đây không phải hai lần khởi tạo độc lập từ
  đầu.
- Mạng compact có kích thước kênh (4, 10, 100, 70), so với dense
  (6, 16, 120, 84). Việc cắt là structured: tensor của các lớp thật sự
  nhỏ đi; không chỉ đặt mask lên những trọng số 0 trong tensor giữ nguyên kích
  thước. Tên “50%” chỉ mục tiêu gần một nửa MAC tổng, không có nghĩa từng lớp
  đều bị cắt đúng 50% kênh.
- Huấn luyện chạy trên CPU, mất khoảng 280,5 giây trong lần chạy đã ghi nhận.

## Dữ liệu và lọc

| Hạng mục | Số lượng |
|---|---:|
| Ảnh nguồn được quét | 37.297 |
| Ảnh được chấp nhận | 25.048 (67,16%) |
| Ảnh bị bỏ qua | 12.249 (32,84%) |
| Crop ký tự tạo ra | 200.854 |
| Crop train | 180.768 |
| Crop validation | 20.086 |
| Biển validation | 2.505 |

Trong các ảnh bị bỏ qua, 11.802 trường hợp bị lỗi segmentation/không khớp số
ký tự và 447 trường hợp bị parser đánh dấu bad_filename. Không có bước kiểm
tra thủ công để kết luận cả 447 trường hợp đều là ký tự ngoài bộ 30 lớp; nhóm
này có thể bao gồm tên file, tọa độ hoặc định dạng nhãn không hợp lệ. Ảnh gốc
không bị sửa hay xóa.

| Nhóm biển được giữ | Số biển |
|---|---:|
| Một hàng | 14.494 |
| Ô tô hai hàng | 7.105 |
| Xe máy hai hàng | 3.449 |

Pipeline tiền xử lý đã thử perspective correction trên 25.048 crop và áp dụng
được cho 14.157 crop; bước phân loại hàng ghi nhận 16.214 crop một hàng và
8.834 crop hai hàng. Các ngưỡng bỏ qua gồm Laplacian blur variance < 80 và
góc ước lượng > 8°. Trong số mẫu được bộ thống kê gắn nhãn mờ, 36/87 bị loại;
trong nhóm nghiêng, 38/94 bị loại. Đây là thống kê của bộ lọc chuẩn bị dữ liệu,
không phải tỷ lệ bỏ qua của ứng dụng thời gian thực sau khi triển khai.

## Độ chính xác validation

| Chỉ số | Dense | Structured | Chênh lệch structured − dense |
|---|---:|---:|---:|
| Ký tự đúng | 18.527 / 20.086 = **92,24%** | 18.400 / 20.086 = **91,61%** | −0,63 điểm % |
| Biển khớp hoàn toàn | 2.098 / 2.505 = **83,75%** | 2.033 / 2.505 = **81,16%** | −2,59 điểm % |

Ngay sau khi chuyển sang các kênh đã chọn, trước fine-tuning, validation
character accuracy của nhánh compact chỉ là 11,07%; 5 epoch fine-tune đã khôi
phục lên 91,61%. Vì vậy không nên đánh giá structured pruning chỉ ở thời điểm
cắt kênh mà bỏ qua bước fine-tuning.

### Theo loại biển

| Loại | Số biển | Dense ký tự | Structured ký tự | Dense exact biển | Structured exact biển |
|---|---:|---:|---:|---:|---:|
| Một hàng | 1.441 | 98,24% | 98,01% | 93,06% | 91,74% |
| Ô tô hai hàng | 723 | 94,36% | 93,96% | 85,20% | 81,33% |
| Xe máy hai hàng | 341 | 65,22% | 62,64% | 41,35% | 36,07% |

### Theo độ dài chuỗi nhãn

| Số ký tự | Số biển | Dense exact biển | Structured exact biển |
|---:|---:|---:|---:|
| 7 | 216 | 81,94% | 73,61% |
| 8 | 2.029 | 89,40% | 87,88% |
| 9 | 258 | 41,47% | 35,27% |
| 10 | 2 | 0% | 0% |

Nhóm xe máy, chuỗi 9 ký tự và các chữ cái hiếm yếu rõ rệt; nhóm 10 ký tự chỉ
có 2 mẫu nên không đủ để kết luận. Toàn bộ thống kê theo vị trí ký tự, từng
lớp ký tự và confusion matrix nằm trong
[channel30_matched_structured_metrics.json](channel30_matched_structured_metrics.json)
và [channel30_matched_structured_confusion.csv](channel30_matched_structured_confusion.csv).

## Tính MAC và tài nguyên mô hình

Đầu vào mỗi ký tự là 1×32×32. LeNet dùng hai convolution kernel 5×5, mỗi
lớp theo sau bởi pooling 2×2; kích thước feature map trước pooling là
28×28 rồi 10×10. Với 30 lớp đầu ra:

| Lớp | Công thức MAC dense | MAC dense | Công thức MAC structured | MAC structured |
|---|---:|---:|---:|---:|
| Conv1 | 6×25×28×28 | 117.600 | 4×25×28×28 | 78.400 |
| Conv2 | 16×6×25×10×10 | 240.000 | 10×4×25×10×10 | 100.000 |
| FC1 | 400×120 | 48.000 | 250×100 | 25.000 |
| FC2 | 120×84 | 10.080 | 100×70 | 7.000 |
| FC3 | 84×30 | 2.520 | 70×30 | 2.100 |
| **Tổng/ký tự** |  | **418.200** |  | **212.500** |

Tỷ lệ giảm MAC được tính:

    (418.200 − 212.500) / 418.200 × 100 = 49,19%

| Chuỗi biển | Dense MAC | Structured MAC |
|---|---:|---:|
| 7 ký tự | 2.927.400 | 1.487.500 |
| 8 ký tự | 3.345.600 | 1.700.000 |
| 9 ký tự | 3.763.800 | 1.912.500 |
| 10 ký tự | 4.182.000 | 2.125.000 |

Sau khi đã tách ký tự, số hàng không tự làm thay đổi MAC của LeNet; số ký tự
được phân loại mới quyết định tổng MAC. Phép tính trên **không** bao gồm tìm
biển trong ảnh toàn cảnh, perspective correction, row classification,
segmentation, làm nét hay truyền dữ liệu.

| Tài nguyên | Dense | Structured | Mức giảm |
|---|---:|---:|---:|
| Tham số | 63.406 | 35.414 | 44,15% |
| Bộ nhớ trọng số thô FP32 | 247,68 KiB | 138,34 KiB | 44,15% |

Đây là kích thước trọng số thô, không phải kích thước file checkpoint. Lần thử
này chưa lượng tử hóa INT8 và chưa tổng hợp Verilog/FPGA.

## Đo latency CPU

Đo forward pass PyTorch CPU với tensor N×1×32×32, 1 thread, 250 lượt warm-up
và 2.500 lượt đo cho mỗi kiến trúc/kích thước batch. Máy đo: Intel Core
i5-12500H, PyTorch 2.14.0+cpu. Số liệu không tính I/O ảnh, tiền xử lý, giải
mã hay pipeline camera.

| Batch ký tự | Dense p50 (ms) | Structured p50 (ms) | Speedup p50 | Giảm latency p50 | Dense p95 (ms) | Structured p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,17715 | 0,16460 | 1,076× | 7,08% | 0,25372 | 0,23834 |
| 7 | 0,41705 | 0,34440 | 1,211× | 17,42% | 0,71984 | 0,61234 |
| 8 | 0,43905 | 0,35775 | 1,227× | 18,52% | 0,75640 | 0,63331 |
| 9 | 0,48135 | 0,39215 | 1,227× | 18,53% | 0,80295 | 0,65573 |
| 10 | 0,51290 | 0,41610 | 1,233× | 18,87% | 0,82186 | 0,70080 |

Cách tính với batch 8:

    Speedup = 0,43905 / 0,35775 = 1,227×
    Giảm latency = (0,43905 − 0,35775) / 0,43905 × 100 = 18,52%

MAC lý thuyết giảm 49,19% nhưng latency CPU không giảm cùng tỷ lệ: thời gian
còn phụ thuộc kernel, băng thông bộ nhớ, overhead framework và kích thước
batch. Kết quả batch 1 cũng ít cải thiện hơn batch lớn. Vì vậy cần xem latency
đo thực tế song song với MAC, không thể suy tốc độ phần cứng trực tiếp từ MAC.
Summary và 25.000 mẫu timing thô được lưu ở
[channel30_matched_speed_benchmark.json](channel30_matched_speed_benchmark.json)
và [channel30_matched_speed_raw.csv](channel30_matched_speed_raw.csv).

## Giới hạn và cách diễn giải

1. Validation được chia ngẫu nhiên ở mức frame, không gom nhóm identity biển
   trước khi chia. Các frame gần trùng/cùng identity có thể xuất hiện ở cả
   train và validation; vì thế điểm số này phù hợp để so sánh hai kiến trúc
   trên cùng split, chưa phải ước lượng tổng quát hóa độc lập đáng tin cậy.
2. OCR được đánh giá trên crop ký tự đã qua segmentation. Số lượng/vị trí ký
   tự validation được xây từ nhãn tên file (ground-truth length); metric không
   tính lỗi phát hiện biển, phân loại một/hai hàng hoặc lỗi segmentation chạy
   tự do. Không nên diễn giải đây là độ chính xác end-to-end của camera.
3. Chưa có đo tài nguyên/clock trên DE10-Lite. Muốn chứng minh tăng tốc FPGA
   cần triển khai cả hai kiến trúc cùng pipeline/cùng clock rồi so ALM, DSP,
   RAM, fMAX/timing slack và latency/cycles. Công thức quy đổi là
   latency = cycles / clock_frequency; speedup là
   latency_dense / latency_structured.
4. Structured đạt tiết kiệm tính toán và latency trong benchmark CPU này,
   nhưng accuracy giảm; đặc biệt cần bổ sung dữ liệu/đánh giá độc lập cho xe
   máy, biển nghiêng/mờ và các ký tự hiếm trước khi chọn model cuối.

## Tệp kết quả và tái lập

- Trainer: train_matched_channel_pruning.py.
- Chuẩn bị dữ liệu và pipeline crop: train_independent_structured.py.
- Benchmark CPU: benchmark_structured_speed.py.
- Checkpoint: lenet5_dense_channel30_matched_v1.pt và
  lenet5_structured50_channel30_matched_v1.pt.
- Metrics, history, confusion matrix và raw benchmark nằm cạnh báo cáo trong
  artifacts/. Ảnh nguồn/crop và dự đoán theo từng biển không được commit.

Ví dụ tạo lại data ở một thư mục output mới, chưa tồn tại:

    .venv/Scripts/python.exe train_independent_structured.py --mode prepare --source "data/OCR/OCR/images/train(1)/detection" --data data/independent_chars_train1_structured_channel30_v1 --metrics artifacts/channel30_matched_structured_prepare.json --enhancement clahe_sharp

Sau đó chạy trainer trên manifest đã chuẩn bị. Vì trainer không ghi đè
checkpoint, hãy dùng các tên output mới nếu muốn chạy lại trên máy đã có các
checkpoint của thí nghiệm này:

    .venv/Scripts/python.exe train_matched_channel_pruning.py --data data/independent_chars_train1_structured_channel30_v1 --preparation-metrics artifacts/channel30_matched_structured_prepare.json --dense-output artifacts/repro_dense.pt --structured-output artifacts/repro_structured.pt --metrics artifacts/repro_metrics.json --history-csv artifacts/repro_history.csv --confusion-csv artifacts/repro_confusion.csv

Để chạy lại timing với tên file khác:

    .venv/Scripts/python.exe benchmark_structured_speed.py --dense artifacts/lenet5_dense_channel30_matched_v1.pt --structured artifacts/lenet5_structured50_channel30_matched_v1.pt --output-json artifacts/repro_speed.json --output-csv artifacts/repro_speed_raw.csv --repeats 2500 --warmup 250 --threads 1
