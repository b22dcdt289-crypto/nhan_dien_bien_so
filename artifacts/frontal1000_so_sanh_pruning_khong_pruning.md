# So sánh LeNet-5 có và không structured pruning

## Tóm tắt

Đã huấn luyện lại mô hình LeNet-5 đầy đủ, không pruning, trên đúng bộ dữ liệu 1.000 biển đã chuẩn bị và dùng cùng quy trình loss nhạy chi phí, hard-example mining (OHEM), augmentation, optimizer và lịch train. So sánh với checkpoint đã pruning Conv2 từ 16 xuống 8 kênh, đánh giá trên cùng 300 validation và 500 test.

**Lưu ý về tính công bằng:** mô hình dense khởi tạo từ checkpoint LeNet-5 đầy đủ; mô hình pruning được lấy từ checkpoint student Conv2 đã pruning rồi fine-tune trên bộ 1.000 biển. Do điểm khởi tạo không đồng nhất hoàn toàn, so sánh này phản ánh hai quy trình/checkpoint thực tế, chưa phải thí nghiệm cô lập duy nhất tác động của pruning.

## Kết quả nhận dạng

| Chỉ số | Không pruning | Structured pruning Conv2 | Ghi chú |
|---|---:|---:|---|
| Validation, độ chính xác ký tự trên crop tham chiếu | 99.917% | 99.875% | 2.400 ký tự, validation dùng để chọn/tinh chỉnh mô hình; không phải test độc lập |
| Validation, biển khớp toàn bộ trên crop tham chiếu | 298/300 (99.333%) | 297/300 (99.000%) | Bỏ qua lỗi cắt ký tự; chỉ đo OCR crop tham chiếu |
| Test, độ chính xác ký tự có căn chỉnh từ segmenter | N/A | N/A | N/A nếu không có biển nào segment ra đúng 8 ký tự |
| Test, khớp nguyên biển end-to-end | 0/500 (0.000%) | 0/500 (0.000%) | Tính cả bỏ qua và sai số lượng ký tự là lỗi |
| Test, độ phủ segmenter | 67.6% | 67.6% | Tiền xử lý/cắt ký tự dùng chung nên gần như không phụ thuộc kiến trúc CNN |

Trong bộ test mới này, dữ liệu chuẩn bị hiện có không tạo được crop tham chiếu cho ký tự (độ phủ reference segmentation = 0%). Segmenter chạy tự do cũng không trả đúng 8 ký tự cho bất kỳ biển nào; vì vậy accuracy ký tự/độ chính xác OCR trên biển test là **không đo được**, không được diễn giải thành accuracy CNN bằng 0%. End-to-end exact vẫn là 0/500 do lỗi cắt/đếm ký tự và/hoặc bỏ qua. Đây là giới hạn pipeline hiện tại, không thể kết luận từ test này rằng CNN nhận dạng sai toàn bộ.

Validation có 300 biển và 2.400 crop tham chiếu nhưng các danh tính đã xuất hiện trong tập dữ liệu của mô hình trước; đây là số liệu chẩn đoán, không đại diện cho generalization độc lập.

Trên 500 biển test, cả hai dùng cùng bộ cắt: thử nhận dạng 338 biển, bỏ qua 162; histogram số crop ký tự là `{"7": 262, "6": 39, "9": 66, "5": 15, "1": 12, "10": 10, "2": 7, "3": 5, "11": 2, "4": 4, "12": 1}`. Có 423 biển sai số lượng ký tự so với nhãn; không có chuỗi 8 ký tự phù hợp để báo accuracy ký tự runtime. Điều này chỉ ra nút thắt trước hết nằm ở phân đoạn/cắt ký tự.

## Kết quả theo từng lớp ký tự trên validation

Các số dưới đây là precision/recall/F1 trên crop tham chiếu của validation (các lớp không có mẫu validation được lược bỏ). Recall tương ứng accuracy trong từng lớp.

| Ký tự | Số mẫu | Dense P/R/F1 | Pruned P/R/F1 |
|---|---:|---:|---:|
| 0 | 229 | 99.57% / 100.00% / 99.78% | 99.57% / 100.00% / 99.78% |
| 1 | 199 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| 2 | 208 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| 3 | 419 | 100.00% / 100.00% / 100.00% | 100.00% / 99.76% / 99.88% |
| 4 | 170 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| 5 | 123 | 100.00% / 99.19% / 99.59% | 100.00% / 100.00% / 100.00% |
| 6 | 383 | 99.74% / 99.74% / 99.74% | 100.00% / 99.74% / 99.87% |
| 7 | 121 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| 8 | 128 | 100.00% / 100.00% / 100.00% | 99.22% / 100.00% / 99.61% |
| 9 | 120 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| A | 150 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| B | 23 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| C | 119 | 100.00% / 100.00% / 100.00% | 100.00% / 99.16% / 99.58% |
| D | 5 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| E | 2 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |
| G | 1 | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% |

## MAC, tham số và bộ nhớ trọng số

| Tài nguyên | Không pruning | Conv2 pruning | Mức giảm nhờ pruning |
|---|---:|---:|---:|
| MAC / ký tự | 418,200 | 274,200 | 34.43% |
| MAC / biển 8 ký tự | 3,345,600 | 2,193,600 | 34.43% |
| Số tham số | 63,406 | 38,198 | 39.76% |
| Trọng số FP32 thô | 253,624 B (247.68 KiB) | 152,792 B (149.21 KiB) | Chưa tính metadata/runtime |
| Trọng số INT8 lý thuyết | 63,406 B | 38,198 B | Chỉ ước lượng theo 1 byte/tham số, chưa lượng tử hóa/kiểm tra |

Nếu giả định DE10-Lite chạy 50 MHz và có đúng một MAC mỗi chu kỳ, cận dưới theo phép đếm là dense: **8.364 ms/biển 8 ký tự**; pruning: **5.484 ms/biển**. Đây chỉ là ước lượng workload; số chu kỳ thực tế còn phụ thuộc kiến trúc nhân MAC, pipeline, truy cập bộ nhớ và Quartus. Chưa có AI RTL/Quartus synthesis nên không có kết quả fMAX, DSP/ALM/BRAM hoặc độ trễ FPGA đo thật.

## Tốc độ đo trên CPU

Benchmark PyTorch CPU một luồng, chạy xen kẽ: batch 1 p50 pruning **0.1753 ms**, dense **0.1802 ms**; batch 8 p50 pruning **0.4118 ms**, dense **0.4491 ms** (dense chậm hơn +9.1% ở batch 8). Đây là CPU host, không phải DE10-Lite; khác biệt có thể chịu ảnh hưởng nhiễu đo và backend.

## Quy trình huấn luyện được giữ nguyên

- Dữ liệu: 1,000 biển train, 300 validation, 500 test; 8.000 ký tự train; sửa 7 nhãn train đã kiểm tra bằng mắt (chỉ áp dụng trong bộ nhớ, không sửa manifest nguồn).
- Mô hình đối chứng: LeNet-5 dense, Conv1=6 và Conv2=16; Conv1 được cố định để tương ứng phương pháp trước. Không xóa/prune kênh nào.
- Loss: weighted cross-entropy + 0.25 × kỳ vọng confusion cost; OHEM chọn top 25% mẫu khó để tăng trọng số, không loại bỏ mẫu còn lại.
- Optimizer/lịch: AdamW, learning rate đầu 0.0001, cosine annealing; batch 256; hoàn thành 20 epoch (checkpoint tốt nhất epoch 5).
- Phân phối lớp train: hỗ trợ 19/30 lớp; tỉ lệ lớp nhiều nhất/ít nhất có mẫu là 669.0:1. Một số lớp có rất ít hoặc không có mẫu, do đó kết quả không thể khái quát đồng đều toàn bộ 30 ký tự.

## Kết luận và giới hạn

Structured pruning giảm phép tính và số tham số theo đúng kích thước kiến trúc: MAC giảm 34.43%, tham số giảm 39.76%. Đo CPU cùng lúc cho biết xu hướng độ trễ phần mềm, nhưng không chứng minh tốc độ FPGA tăng tương ứng. Chất lượng hiện chưa thể kết luận chắc trên biển test do nút thắt segmenter/crop tham chiếu. Muốn kết luận accuracy end-to-end cần sửa bước cắt/nhãn cho test trước rồi đánh giá lại; đồng thời nên làm thí nghiệm pruning-vs-dense từ cùng checkpoint teacher và cùng seed để cô lập tác dụng pruning.

Số liệu gộp chi tiết: `artifacts\frontal1000_pruning_vs_no_pruning_metrics.json`; bảng theo từng lớp ký tự: `artifacts/frontal1000_validation_per_class_comparison.csv`. Không đưa mã biển/ảnh từng biển vào báo cáo công khai.
