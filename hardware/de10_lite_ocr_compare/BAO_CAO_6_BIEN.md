# Thí nghiệm 6 biển: giải thích số liệu và đối chứng dense / pruning

## Kết luận cần báo cáo trung thực

Đã train từ đầu trên **6 biển** thuộc `train(1)` (2 ô tô một hàng, 2 ô tô hai hàng, 2 xe máy hai hàng), tổng **49 crop ký tự**. Chỉ có **14/30 lớp** xuất hiện; 16 lớp không có ví dụ dương. Kiểm tra cuối trên **1,279 biển / 10,255 ký tự** khác nội dung với tập train và tập chọn checkpoint.

Nguồn ảnh gốc: `data/OCR/OCR/images/train(1)/detection/`; crop dùng để train/test: `data/independent_chars_train1_structured_channel30_v1/`. Mục tiêu trên 95,22% **đúng toàn biển** chưa đạt trong thí nghiệm này.

| Chỉ số | Dense | Pruning cấu trúc 50% kênh Conv2 |
|---|---:|---:|
| Số tham số | 63,406 | 38,198 |
| MAC lý thuyết/ký tự | 418,200 | 274,200 |
| Epoch tốt nhất trên development | 80 | 80 |
| Train đúng ký tự | 95.92% (47/49) | 95.92% (47/49) |
| Train đúng cả biển | 66.67% (4/6) | 66.67% (4/6) |
| Development đúng ký tự | 57.00% (5,604/9,831) | 56.54% (5,558/9,831) |
| Test đúng ký tự | 57.74% (5,921/10,255) | 57.38% (5,884/10,255) |
| Test đúng cả biển | 2.03% (26/1,279) | 2.11% (27/1,279) |
| Test ký tự thuộc lớp chưa từng train | 0.00% (0/280) | 0.00% (0/280) |
| CPU forward p50, batch 1 | 0.2125 ms | 0.2113 ms |
| CPU forward p50, batch 8 | 0.3921 ms | 0.3701 ms |
| Train + development, gồm 40 epoch chung | 4.467 s | 4.335 s |

Thời gian toàn lượt từ nạp dữ liệu đến benchmark: 227.00 s; tổng các thời gian train+development đã ghi chỉ 6.49 s. Phần lớn thời gian còn lại là mở hàng chục nghìn crop để lập development/test và các bước đánh giá/benchmark; không được gọi toàn bộ wall-time này là thời gian train. Thời gian từng epoch rất nhỏ nên chênh lệch giữa hai nhánh dễ bị nhiễu hệ thống.

Phân bố 49 nhãn train: `0`=6, `1`=6, `2`=3, `3`=7, `4`=2, `5`=4, `6`=6, `7`=1, `8`=3, `9`=5, `A`=2, `B`=2, `C`=1, `K`=1. Các lớp không xuất hiện: `DEFGHLMNPSTUVXYZ`.

Chênh lệch test giữa hai bản chỉ 37 ký tự trên 10,255; pruned hơn dense đúng 1 biển trên 1,279. Không thể diễn giải chênh lệch biển nhỏ này thành cải thiện độ chính xác do pruning. Khoảng Wilson 95% tham khảo cho tỷ lệ biển đúng: dense 1.39–2.96%, pruned 1.45–3.05%; các frame lặp lại có thể làm khoảng này quá hẹp.

Ở lượt train tập lớn trước đây có 180.768 crop train và 20.086 crop validation, nên không được đặt hai tỷ lệ accuracy cạnh nhau như một A/B test: lượt này chỉ có 49 crop train và dùng phân hoạch test riêng. Điều có thể kết luận chắc chắn trong lượt này là **hai kiến trúc được so trên cùng 6 ảnh train và cùng 1.279 biển test**.

## Test phân theo loại biển (đúng ký tự; đúng toàn biển)

| Nhóm | Dense ký tự | Dense biển | Pruned ký tự | Pruned biển |
|---|---:|---:|---:|---:|
| Ô tô một hàng | 61.23% (3,582/5,850) | 2.30% (17/740) | 61.74% (3,612/5,850) | 2.70% (20/740) |
| Ô tô hai hàng | 60.59% (1,742/2,875) | 2.47% (9/364) | 58.78% (1,690/2,875) | 1.92% (7/364) |
| Xe máy hai hàng | 39.02% (597/1,530) | 0.00% (0/175) | 38.04% (582/1,530) | 0.00% (0/175) |

## Vì sao chọn các chỉ số này?

- **Đúng toàn biển trên test độc lập** là chỉ số gần mục tiêu đồ án nhất: đầu ra phải trùng mọi ký tự. Đúng từng ký tự giúp chẩn đoán OCR nhưng không thay được đúng toàn biển.
- **Từng nhóm biển** kiểm tra mô hình có làm tốt cả một/hai hàng và xe máy, tránh trung bình chung che nhóm yếu.
- **Train–development–test** tách việc học thuộc, chọn checkpoint và đánh giá cuối. Khoảng cách train–test lớn là dấu hiệu tổng quát hóa kém.
- **Ma trận nhầm lẫn và lớp chưa thấy** tìm lỗi dữ liệu/cắt ký tự; không có nhãn train cho một lớp thì không thể kỳ vọng học lớp đó đáng tin cậy.
- **MAC/tham số** cho biết độ phức tạp lý thuyết, còn **LE, RAM bit, DSP, chu kỳ, timing** sau Quartus và thời gian truyền đo trên kit mới cho biết giới hạn phần cứng. MAC giảm không đồng nghĩa tổng thời gian giảm cùng tỷ lệ.

## Quy trình train và lý do của từng bước

1. Đọc `train(1)` và dùng crop 32×32 đã chuẩn bị, kiểm tra trực quan 6 ảnh gốc. Một file gắn nhãn ô tô hai hàng thực chất hiện biển một hàng, nên đã thay bằng ảnh hai hàng đúng trước train; tập gốc không bị sửa/xóa.
2. Chọn đúng 2 biển/nhóm theo seed cố định, ghi riêng danh sách nguồn cục bộ. Chỉ dùng 6 biển này để cập nhật trọng số, không nạp checkpoint từng học tập lớn.
3. Chia phần `val` theo **chuỗi ký tự biển** bằng hash cố định thành development và test; loại chuỗi trùng 6 biển train. Chọn checkpoint theo accuracy ký tự trên development; chỉ tính/chủ động xem kết quả test sau khi chọn checkpoint.
4. Chuẩn hóa crop về [-1,1]; lúc train xoay ngẫu nhiên ±6° ở 55% lượt để giảm phụ thuộc góc rất nhỏ. Không xoay ảnh development/test.
5. Cùng LeNet-5 30 đầu ra, cross-entropy, AdamW, batch 16, learning rate 0.0005, weight decay 1e-05. Train dense chung 40 epoch, sau đó dense thêm 40 epoch; bản kia cắt nguyên 8/16 kênh Conv2 và các cột FC1 tương ứng rồi train tiếp cùng số epoch. Conv1 giữ 6 kênh. Mỗi epoch chỉ có 4 mini-batch từ 49 crop, nên cần nhiều epoch hơn lượt tập lớn mới đủ bước cập nhật; số epoch khác cũng là lý do không so trực tiếp thời gian train hai thí nghiệm. Mỗi epoch của hai nhánh dùng cùng seed trộn/augment.
6. Chấm train, development và test; đo CPU forward lặp 500 lần theo batch 1/8, tính MAC/tham số. Giả lập lượng tử INT8 trên cùng 6 biển tham chiếu để tách lỗi train khỏi lỗi lượng tử; không nhận giả lập là phép đo FPGA.

## Vì sao số liệu chênh lệch?

- **Thiếu và lệch lớp:** 49 nhãn train chỉ phủ 14/30 lớp; test có 280 ký tự thuộc lớp chưa train, cả hai bản đều đúng 0. Các lớp đã thấy cũng chỉ đạt khoảng 59%, nên thiếu lớp chưa phải nguyên nhân duy nhất.
- **Học thuộc 6 biển:** cả hai đúng 47/49 crop train nhưng chỉ khoảng 57% crop test. Một số crop nguồn rất nhỏ/mờ: ký tự B trên biển ô tô hai hàng đã xem trực tiếp trông gần như 8. Nhãn từ tên file và phân nhóm thư mục không bảo đảm từng crop đúng; vì vậy cần duyệt thủ công nếu muốn đánh giá chất lượng thực địa.
- **Đúng toàn biển thấp hơn đúng ký tự:** chỉ cần sai một ký tự là cả biển sai. Không nhân đơn giản accuracy ký tự lên lũy thừa vì lỗi giữa ký tự/biển có tương quan và chiều dài biển không cố định.
- **Pruning đổi năng lực mô hình:** số tham số giảm 39.76% và MAC giảm 34.43%; accuracy có thể tăng/giảm vài mẫu do tối ưu và nhiễu chọn mẫu. Không được gọi 1 biển chênh lệch là pruning cải thiện accuracy.
- **CPU, FPGA và truyền là ba thời gian khác nhau:** p50 batch 1 CPU gần như bằng nhau do overhead chi phối, dù MAC khác. Bộ RTL MAC tuần tự đã đo trên kit ở lần trước giảm từ 870.198 xuống 577.998 chu kỳ/ký tự (nhanh lõi 1,506×); qua USB-Blaster/JTAG vẫn khoảng 1 giây/ký tự. Đây là đo của cùng **kiến trúc** với checkpoint tập lớn, không phải phép đo lại trên kit cho trọng số train từ 6 biển.

## Giả lập INT8 và giới hạn phần cứng

| Kiểm tra trên cùng 6 biển tham chiếu (không nằm trong 6 biển train) | Dense | Pruned |
|---|---:|---:|
| FP32 đúng ký tự | 25/50 | 25/50 |
| INT8 đúng ký tự | 25/50 | 25/50 |
| FP32–INT8 dự đoán giống nhau | 50/50 | 50/50 |
| Đúng cả biển | 0/6 | 0/6 |
| Trọng số INT8 | 63,150 byte | 37,950 byte |

Bản FP32 và giả lập INT8 ra đúng cùng từng ký tự trong 50 crop này, nên lượng tử **không giải thích** độ chính xác thấp của lượt thử này. Chưa nạp hai checkpoint 6-biển xuống kit; số tài nguyên/tốc độ Quartus bên dưới là **tham chiếu kiến trúc đã đo với checkpoint trước**, không được trình bày là phép đo của trọng số mới.

| Tài nguyên/tốc độ Quartus trước đây, cùng RTL | Dense stream | Pruned stream |
|---|---:|---:|
| Logic elements | 2,356 | 2,324 |
| RAM bit | 560,432 | 358,832 |
| DSP 9-bit | 1 | 1 |
| Chu kỳ/ký tự | 870,198 | 577,998 |
| Lõi ms/ký tự @50MHz | 17.404 ms | 11.560 ms |
| JTAG toàn lượt p50/ký tự | 1,074.000 ms | 1,059.000 ms |

Nạp lại checkpoint 6-biển **chưa cần** để kết luận về hiệu năng kiến trúc: số kênh, phép MAC và RTL không đổi, trong khi accuracy test quá thấp để dùng làm demo. Muốn xác nhận bit-exact của trọng số mới trên FPGA thì phải tổng hợp/nạp **bitstream riêng**, chạy lại JTAG và ghi số liệu mới; không được lấy số đo cũ gán cho checkpoint mới.

## Nhầm lẫn lớn nhất trên test (nhãn thật → dự đoán)

| Dense | Số lần | Pruned | Số lần |
|---|---:|---|---:|
| 7→2 | 337 | 7→2 | 342 |
| 4→6 | 252 | 4→6 | 275 |
| 8→9 | 244 | 8→9 | 231 |
| 6→5 | 202 | 6→5 | 187 |
| 2→3 | 164 | 8→6 | 165 |
| 0→3 | 155 | 1→3 | 154 |
| 1→3 | 155 | 0→3 | 145 |
| 8→6 | 141 | C→0 | 133 |
| C→0 | 139 | 2→0 | 126 |
| 2→0 | 114 | 1→2 | 115 |

Chi tiết 30×30 nhầm lẫn và số đúng/tổng nằm trong `micro6_public_metrics.json`; loss, accuracy và thời gian từng epoch trong `micro6_train_history.csv`. Danh sách ảnh, crop, nhãn và checkpoint chỉ lưu cục bộ trong `artifacts/`, không đẩy lên GitHub.

**Giới hạn quan trọng:** test vẫn là các crop ký tự với ranh giới/số ký tự biết từ nhãn, chưa phải ảnh camera → tìm biển → cắt → OCR trên FPGA. Một số frame/biển trùng danh tính trong nguồn; khoảng tin cậy chỉ tham khảo. Timing Quartus trước đây cũng chưa được ràng buộc đầy đủ. Chưa đo công suất điện hoặc tỷ lệ bỏ qua ảnh mờ/nghiêng trong thí nghiệm 6 biển.
