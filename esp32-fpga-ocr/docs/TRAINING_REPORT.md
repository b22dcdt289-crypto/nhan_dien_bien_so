# Báo cáo huấn luyện nhận dạng biển một hàng và hai hàng

Ngày 17/11/2026. Phạm vi thử: ROI đã cắt chứa biển trực diện trên máy tính; biển ô tô một hàng, ô tô hai hàng và xe máy hai hàng. Chỉ ảnh tổng hợp được dùng để train; ảnh camera không được dùng để train.

## 1. Dữ liệu và phương pháp

- Mạng LeNet K3 giữ hai lớp Conv 6/16 kênh, FC 120/84, ảnh xám 28 × 28; mở rộng từ 30 lên 31 lớp bằng cách thêm chữ O cho bài thử OCR. Lớp O là mẫu gây nhiễu tổng hợp; không mặc nhiên coi mọi biển chứa O là định dạng được cấp.
- Bộ mới có 2.700 biển train, 460 validation, 380 calibration và 800 test. Biển test: 250 ô tô một hàng, 250 ô tô hai hàng, 300 xe máy hai hàng; xe máy có 150 mẫu seri chữ–chữ và 150 mẫu chữ–số. Chia theo chuỗi biển trước khi dựng, không trùng danh tính với các phân vùng hoặc bộ cũ đã biết.
- Test có 402 biển sử dụng phông giữ riêng không có trong các tập train đã kiểm tra, 145 biển chứa O. Ảnh dùng lại một số mẫu nền tổng hợp; đây không phải ảnh biển vật lý, và phân bố mẫu O được tăng chủ ý để thử nhầm lẫn.
- Bổ sung kiểu chữ nét vuông, số 0 có nét chéo, làm mờ nhẹ và mất một phần nét. Những tập test đã xem trong các lượt trước chỉ dùng phát triển; test cuối là chuỗi biển và phông được tách riêng trước khi chọn mô hình.
- Tách ký tự theo ảnh, không dùng nhãn hoặc bounding box chuẩn làm đầu vào. Bản thử bỏ phần viền khi cần, phân nhóm hàng, sắp từ trái sang phải rồi chuẩn hóa 28 × 28. Quyết định có xét vị trí trong phạm vi định dạng thử; ở vị trí thứ tư của biển 9 ký tự, cho phép cả chữ và số. Nhãn chuẩn chỉ dùng sau suy luận để chấm.
- Với cặp 0/O, chỉ gộp bằng chứng nếu cả hai có điểm đáng kể và vị trí có một loại phù hợp; nếu điểm yếu hoặc xung đột thì từ chối. Điểm model sau hiệu chuẩn không bảo đảm là xác suất đúng ngoài tập hiệu chuẩn.

## 2. Kết quả INT8 trên 800 biển tổng hợp test

| Nhóm | Số biển | Tách được | Chuỗi ứng viên đúng | Đúng và được nhận / toàn bộ | Nhận sai |
|---|---:|---:|---:|---:|---:|
| Ô tô một hàng | 250 | 249 | 247/250 (98.80%) | 232/250 (92.80%) | 1 |
| Ô tô hai hàng | 250 | 248 | 248/250 (99.20%) | 229/250 (91.60%) | 0 |
| Xe máy hai hàng | 300 | 299 | 295/300 (98.33%) | 279/300 (93.00%) | 1 |
| Xe máy seri chữ–chữ | 150 | 149 | 148/150 (98.67%) | 137/150 (91.33%) | 1 |
| Xe máy seri chữ–số | 150 | 150 | 147/150 (98.00%) | 142/150 (94.67%) | 0 |
| Phông đã có trong train | 398 | 396 | 393/398 (98.74%) | 383/398 (96.23%) | 1 |
| Phông giữ riêng | 402 | 400 | 397/402 (98.76%) | 357/402 (88.81%) | 1 |
| Tổng | 800 | 796 | 790/800 (98.75%) | 740/800 (92.50%) | 2 |

Toàn bộ: nhận 742/800 biển, trong đó đúng 740, sai 2; độ chính xác **trong các biển đã nhận** là 99.73%. Tỷ lệ **đúng và được nhận trên toàn bộ** là 92.50%; trường hợp từ chối/tách lỗi đều không được tính đúng.
Tỷ lệ ký tự sau ràng buộc định dạng, tính không điểm cho toàn bộ ký tự của biển tách lỗi hoặc sai số lượng: 6653/6700 = 99.30%. Đây là cách tính bảo thủ ở mức hệ thống, không phải accuracy chỉ trên glyph đã tách thành công.

Nhóm phông giữ riêng đạt 88.81% đúng và được nhận trên toàn bộ, thấp hơn 90%; cần cải thiện độ bao phủ và kiểm tra bằng camera độc lập. Chỉ tiêu 90% được đối chiếu với **đúng và được nhận trên toàn bộ**, không dùng độ chính xác trong nhóm đã nhận để che số biển bị từ chối.

## 3. Các cặp dễ nhầm

Bảng dưới là dự đoán thô INT8 trước ràng buộc định dạng; chỉ tính các biển mà bộ tách trả số lượng ký tự tương ứng. Trước/sau là hai checkpoint đánh giá cùng ảnh, cùng bộ tách. Đây không phải số đo nhầm lẫn trên biển thật.

| Ký tự thật → dự đoán | Số ký tự thật | Trước huấn luyện đa dạng | Sau huấn luyện | Tỉ lệ nhầm sau |
|---|---:|---:|---:|---:|
| A → 0 | 40 | 0 | 0 | 0.00% |
| 0 → A | 491 | 0 | 0 | 0.00% |
| A → O | 40 | 0 | 0 | 0.00% |
| O → A | 145 | 0 | 0 | 0.00% |
| 0 → O | 491 | 76 | 80 | 16.29% |
| O → 0 | 145 | 35 | 27 | 18.62% |
| 0 → 8 | 491 | 0 | 0 | 0.00% |
| 8 → 0 | 558 | 0 | 0 | 0.00% |
| 6 → 8 | 585 | 0 | 0 | 0.00% |
| 8 → 6 | 558 | 9 | 0 | 0.00% |
| 8 → 9 | 558 | 0 | 0 | 0.00% |
| 9 → 8 | 597 | 0 | 0 | 0.00% |
| 5 → S | 593 | 2 | 0 | 0.00% |
| S → 5 | 42 | 0 | 0 | 0.00% |
| 7 → Z | 592 | 1 | 0 | 0.00% |
| Z → 7 | 38 | 0 | 0 | 0.00% |

Nhầm lẫn 0/O ở đầu ra thô còn đáng kể; quy tắc vị trí và cơ chế từ chối giúp xử lý một phần trong các định dạng đã hỗ trợ, nhưng không chứng minh model đã phân biệt chắc chắn hai ký tự khi hình ảnh tương tự.

Ma trận đầy đủ và bảng cặp trọng điểm được lưu trong `test_int8_confusion.csv` và `confusion_focus.csv`.

## 4. Kiểm chứng, giới hạn và triển khai

- Đã so sánh 868 logit INT8 trên 28 mẫu giữa hai tham chiếu phần mềm độc lập: 0 sai khác. Chưa mô phỏng RTL hoặc kiểm tra FPGA cho model 31 lớp.
- Validation chọn epoch 12; macro đúng toàn biển thô của ảnh bình thường là 90.15%, của bản tăng cường 0 có nét chéo là 92.56%. Validation dùng để chọn model, không thay thế test.
- FPGA/UART và firmware ESP32 đang chạy dùng 30 lớp, bộ tách 8 ký tự một hàng và mặt nạ định dạng cũ. Các tệp INT8 31 lớp mới không được nạp trực tiếp vào bitstream cũ; biển hai hàng và 9 ký tự chưa chạy trên kit.
- Cần cập nhật bộ tách ESP32, giao thức/đếm 8–9 ký tự và RTL FC cuối, ROM, calibration, top-3, mask/accept cho 31 lớp; sau đó kiểm tra lại Python–RTL–FPGA và tốc độ. Chưa có số đo 100 fps cho phiên bản model mới.
- Cần đánh giá mù bằng nhiều biển và phiên chụp camera độc lập. Với nét đã mất hoàn toàn, hình ảnh có thể không đủ để phân biệt một cách chắc chắn; hệ thống phải cho phép từ chối thay vì tự khẳng định.

## 5. Nguồn và tệp

- Tham khảo hình dạng, tỷ lệ biển ngắn ô tô và mô tô: [QCVN 08:2024/BCA của Bộ Công an](https://bocongan.gov.vn/KND/vb/vbqp/Lists/GioiThieuVanBan/Attachments/1592/TT81.2024.TTBCA-QCVN%2008.2024.pdf). Ảnh tổng hợp không phải bản sao quy chuẩn.
- Quy định về nhóm chữ seri đối với biển nền trắng trong nước: [Bộ Công an](https://bocongan.gov.vn/bai-viet/nhan-dien-mau-sac-seri-ky-hieu-bien-so-xe-cua-co-quan-to-chuc-ca-nhan-tu-01012025-d1-t1617). Nhóm O trong bộ dữ liệu chỉ dùng thử khả năng OCR.
- `best.pt`: checkpoint FP32; `generated31/`: ROM INT8 để phát triển tiếp, chưa tương thích bitstream đang chạy.
- `test_int8_metrics.json`, `test_int8_predictions.json`, `test_int8_confusion.csv`: dữ liệu kết quả; `int8_crosscheck.json`: kiểm tra hai tham chiếu phần mềm.
- `../dataset_v4_audit.json`: kiểm tra chồng lặp chuỗi và phông; `../segment_enhanced.py`, `../evaluate_diverse.py`: tách ảnh và suy luận trên máy tính.
