# Báo cáo đối chứng LeNet-5: ba loại biển, tỉa kênh Conv2 và đo DE10-Lite

Ngày thí nghiệm: 09/10/2026. Chip đích: MAX 10 `10M50DAF484C7G` trên DE10-Lite, xung ngoài 50 MHz. Mọi số dưới đây lấy từ log train, tệp checkpoint, báo cáo Quartus post-fit hoặc lượt JTAG **của lần chạy này**, không ghép với kết quả cũ.

## 1. Phạm vi và dữ liệu

- 1.000 ảnh dựng biển ô tô một hàng và 1.000 ảnh dựng biển ô tô hai hàng: `data/car_frontal_synthetic_2x1000_v1/`.
- 1.000 ảnh dựng biển xe máy hai hàng 9 ký tự: `data/motorcycle_frontal_synthetic_1000_v2/`. Bộ sinh lấy **chuỗi nhãn** từ tên ảnh nguồn thuộc nhánh train; không sao chép pixel ảnh thật. Có 9 ứng viên tách ký tự thất bại được loại và thay bằng ứng viên khác. Nhãn nguồn chưa được kiểm tra thủ công toàn bộ.
- Dữ liệu dựng được tách 800/100/100 biển mỗi nhóm theo định danh biển: 2.400 train, 300 validation, 300 test. Các phép phân đoạn ảnh dựng phải tự tìm đủ 8/8/9 ký tự; nếu sai thì bộ nạp dữ liệu dừng, không dùng số ký tự thật để chữa ảnh test.
- Để kiểm tra khả năng chuyển sang miền ảnh thật, lấy thêm 600 biển/nhóm (1.800 biển) từ tập crop ký tự nguồn cho giai đoạn fine-tune, 50 biển/nhóm cho validation và 100 biển/nhóm (300 biển) cho test. Chia theo hàm băm của **nội dung biển**, không theo thư mục ảnh. Giao giữa định danh train–validation, train–test, validation–test đều bằng **0**.
- Bộ test thật có 2.500 ký tự: ô tô một hàng 800, ô tô hai hàng 800, xe máy hai hàng 900. Bộ này **không** gồm biển xe máy một hàng, biển 10 ký tự, biển bị che khuất nặng hay bộ camera thời gian thực độc lập.

## 2. Quy trình train công bằng

1. Cùng một seed `20261009`, cùng ảnh dựng train và cùng augmentation ±4°/độ sáng/nhiễu cho hai nhánh. LeNet-5 có Conv1 `1→6`, Conv2 dense `6→16`, sau đó FC `400→120→84→30`. Đầu ra 30 lớp là 10 chữ số và 20 chữ cái được định nghĩa trong mã.
2. Huấn luyện dense 4 epoch ảnh dựng; sao chép trọng số sang student và **tỉa cấu trúc 50% filter/kênh đầu ra Conv2 (`16→8`)** theo độ lớn trọng số. FC đầu tiên được thu gọn từ 400 xuống 200 đầu vào; Conv1 giữ nguyên cấu trúc. Sau đó cả dense và pruned huấn luyện tiếp trên cùng batch ảnh dựng đến epoch 8.
3. Đánh giá hai nhánh sau giai đoạn chỉ dùng ảnh dựng. Tiếp tục fine-tune **cả hai** thêm tối đa 10 epoch trên cùng tập 2.400 biển dựng train + 1.800 biển thật train. Chọn checkpoint bằng accuracy ký tự rồi accuracy nguyên biển trên 150 biển thật validation; test thật chỉ mở sau chọn checkpoint. Đây không phải phép đo “chỉ train bằng ảnh dựng”; hai giai đoạn được báo riêng.
4. Đo ký tự và biển trên tập crop đã phân đoạn. Đo thêm tuyến **ảnh thật → hộp biển từ nhãn nguồn → nắn phối cảnh → tách dòng/ký tự → LeNet-5 OCR** trên cùng 300 định danh test. Hộp biển là chú thích có sẵn; chưa có bước phát hiện biển từ toàn khung hình.

Checkpoint được chọn: dense ở epoch fine-tune 9, pruned ở epoch fine-tune 10. Hai nhánh có cùng dữ liệu/augmentation/seed, nhưng số epoch tiền huấn luyện dense và pruned sau thời điểm tỉa không bằng nhau: student chỉ có 4 epoch sau khi tỉa, còn dense có 8 epoch ảnh dựng. Đây là giới hạn đối chứng cần nêu khi diễn giải chênh lệch nhỏ.

## 3. Độ chính xác FP32 trên tập chưa thấy

| Giai đoạn và tập test | Dense ký tự | Pruned ký tự | Dense nguyên biển | Pruned nguyên biển |
|---|---:|---:|---:|---:|
| Chỉ học ảnh dựng → test ảnh dựng (300 biển, 2.500 ký tự) | 2.499/2.500 = 99,96% | 2.499/2.500 = 99,96% | 299/300 = 99,67% | 299/300 = 99,67% |
| Chỉ học ảnh dựng → test ảnh thật đã crop (300 biển) | 696/2.500 = 27,84% | 712/2.500 = 28,48% | 2/300 = 0,67% | 2/300 = 0,67% |
| Sau fine-tune hỗn hợp → test ảnh thật đã crop (300 biển) | 2.401/2.500 = 96,04% | 2.408/2.500 = 96,32% | 236/300 = 78,67% | 239/300 = 79,67% |
| Sau fine-tune → ảnh thật từ **hộp biển đã chú thích** (300 biển) | 2.214/2.296 ký tự ở biển đủ độ dài = 96,43% | 2.221/2.296 = 96,73% | 224/300 = 74,67% | 227/300 = 75,67% |

Ở hàng cuối, accuracy ký tự chỉ tính trên các biển được tách ra **đúng độ dài chuỗi**; không được dùng làm accuracy ký tự đầu-cuối trên mọi 2.500 ký tự. Chỉ số đầu-cuối nghiêm ngặt hơn là lỗi chỉnh sửa chuỗi: dense `127/2.500 = 5,08%`, pruned `122/2.500 = 4,88%`. Phân đoạn/tách dòng thành công ở 297/300 biển cho cả hai; còn lỗi sai số lượng ký tự hoặc sai phân đoạn khiến accuracy nguyên biển giảm thêm. Các số này vẫn dùng hộp biển được cung cấp, không phải camera toàn khung.

### Từng loại biển thật (100 biển/loại)

| Loại | Dense ký tự trên crop | Pruned ký tự trên crop | Dense nguyên biển trên crop | Pruned nguyên biển trên crop | Dense/Pruned từ hộp biển |
|---|---:|---:|---:|---:|---:|
| Ô tô một hàng, 8 ký tự | 786/800 = 98,25% | 786/800 = 98,25% | 91/100 | 90/100 | 91/100 / 90/100 |
| Ô tô hai hàng, 8 ký tự | 768/800 = 96,00% | 773/800 = 96,63% | 79/100 | 84/100 | 77/100 / 82/100 |
| Xe máy hai hàng, 9 ký tự | 847/900 = 94,11% | 849/900 = 94,33% | 66/100 | 65/100 | 56/100 / 55/100 |

Chênh lệch 3 biển trên 300 giữa hai mô hình không đủ để khẳng định pruning **cải thiện độ chính xác** một cách có ý nghĩa thống kê. Hai nhóm hai hàng, nhất là xe máy 9 ký tự, còn yếu; sai một ký tự đã làm sai nguyên biển. Nhầm lẫn nhiều nhất trên crop thật là `D→0` (dense 9 lần, pruned 7 lần), `6→3` (5 lần mỗi nhánh), `Z→2` (3 lần mỗi nhánh).

## 4. Trọng số, MAC, lượng tử hóa và tài nguyên post-fit

| Chỉ số | Dense | Tỉa kênh Conv2 50% | Diễn giải |
|---|---:|---:|---|
| Conv2 output channels | 16 | 8 | Tỉa filter **có cấu trúc**, không chỉ đặt trọng số bằng 0 |
| Trọng số / bias | 63.150 / 256 | 37.950 / 248 | Giảm 39,90% số trọng số |
| Trọng số INT8 + bias INT32 | 64.174 byte | 38.942 byte | Chưa gồm toàn bộ RAM kích hoạt và overhead bộ nhớ vật lý |
| MAC / ký tự | 418.200 | 274.200 | Giảm 34,43% |
| MAC / biển 8 ký tự | 3.345.600 | 2.193.600 | Chỉ nhân 8 lần chi phí OCR ký tự |
| MAC / biển 9 ký tự | 3.763.800 | 2.467.800 | Chỉ nhân 9 lần chi phí OCR ký tự |
| Logic elements (LE), Quartus | 2.318 / 49.760 | 2.334 / 49.760 | Pruned **tăng 16 LE**, không giảm |
| Combinational functions (thước đo gần LUT của MAX 10) | 1.770 / 49.760 | 1.786 / 49.760 | Không nên gọi LE và LUT là cùng một đại lượng |
| M9K block RAM vật lý | 75 / 182 | 51 / 182 | Giảm 24 M9K = 32,0% |
| Memory bits hữu dụng | 560.432 / 1.677.312 | 358.832 / 1.677.312 | Giảm 35,97% |
| Phần tử nhân 9-bit | 1 / 288 | 1 / 288 | Cùng lõi MAC tuần tự; tỉa không tăng PE song song |
| Fmax góc chậm 85°C | 66,53 MHz | 65,69 MHz | Cả hai > 50 MHz, pruned **thấp hơn** 0,84 MHz |
| Setup slack đường đã ràng buộc ở 50 MHz | +4,970 ns | +4,776 ns | Báo cáo còn ghi **not fully constrained** cho setup và hold |

Số LE, combinational functions, M9K, bit RAM, DSP và Fmax là báo cáo **sau Fitter/Timing Analyzer của Quartus** trên đúng bitstream thử nghiệm. Fmax > 50 MHz và slack dương trên đường có ràng buộc cho phép nạp và chạy phép thử, nhưng **chưa đủ kết luận timing sign-off** vì tồn tại đường chưa được ràng buộc. Cần hoàn thiện SDC và kiểm tra CDC/JTAG, rồi biên dịch/đo lại trước khi gọi là thiết kế hoàn chỉnh.

## 5. Số đo trực tiếp trên DE10-Lite qua USB-Blaster/JTAG

Đã nạp `.sof` của từng nhánh bằng Quartus Programmer vào cùng kit (JTAG ID `0x031050DD`), rồi truyền **cùng 100 ảnh ký tự 32×32** từ 12 biển test: 2 biển ảnh dựng + 2 biển crop thật cho mỗi loại. Đây là tập demo nhỏ, không đại diện cho toàn bộ 300 biển thật. FPGA đã khớp mô phỏng fixed-point INT8 **100/100** cho cả hai nhánh.

| Chỉ số JTAG | Dense | Pruned |
|---|---:|---:|
| Đúng ký tự theo nhãn | 99/100 | 99/100 |
| Đúng nguyên biển trong 12 biển demo | 11/12 | 11/12 |
| Chu kỳ lõi / ký tự | 870.198 | 577.998 |
| Thời gian lõi ở 50 MHz / ký tự | 17,404 ms | 11,560 ms |
| Tăng tốc lõi | 1,00× | **1,506×** so với dense |
| Trung vị nạp 1 KiB ký tự qua JTAG | 801 ms | 813 ms |
| Trung vị toàn lượt JTAG + tính toán | 860 ms | 876 ms |

Như vậy pruning **có tăng tốc phần lõi** theo chu kỳ thực đo và giảm BRAM, nhưng **không tăng tốc toàn tuyến JTAG** trong thử nghiệm này; thời gian truyền lấn át khoảng 12–17 ms tính toán. Chênh lệch 860/876 ms còn bị ảnh hưởng bởi giao thức/độ dao động truyền nên không diễn giải là pruned “chậm hơn” về lõi. Cần một giao tiếp streaming có băng thông cao và chuỗi xử lý nhiều ký tự trên FPGA để chứng minh nhận dạng biển số **thời gian thực**.

## 6. OS / WS / RS: phạm vi đã kiểm chứng

Trên 6 crop ký tự dựng chưa thấy (2 ký tự/loại biển) cho **mỗi checkpoint**, ba lịch tích chập output-stationary (OS), weight-stationary (WS), row-stationary (RS) đều cho argmax khớp PyTorch 6/6; sai số logit lớn nhất dưới `1×10⁻⁶` với OS/WS và dưới `1×10⁻⁶` với RS trong lượt này. Đây là kiểm chứng **chức năng CPU/FP32 của thứ tự vòng lặp**, không phải ba bitstream FPGA. RTL đang đo là **một MAC tuần tự, tích lũy đầu ra**, có truyền một ảnh ký tự qua 64 gói JTAG. Chưa xây ba mạng PE/buffer/bộ điều khiển OS–WS–RS riêng, nên **không có** số LE/M9K/Fmax/tốc độ FPGA riêng cho WS và RS và không được kết luận WS/RS nhanh/chậm hơn OS trên DE10-Lite.

Việc chọn OS/WS/RS là thiết kế dòng dữ liệu và tái sử dụng tensor khi **suy luận**; nó không thay đổi quy tắc tối ưu trọng số của hai lượt train. Để so sánh kiến trúc thực, bước tiếp theo là giữ cùng checkpoint INT8 và cùng tập ký tự, triển khai ba RTL độc lập với số PE, băng thông và bộ nhớ được định nghĩa rõ, rồi so post-fit và JTAG ở cùng điều kiện clock.

## 7. Kết luận theo mục tiêu đề tài

- **Đã chứng minh được:** LeNet-5 OCR 30 lớp nhận được ba nhóm biển ở mức ký tự; tỉa Conv2 50% giảm trọng số, MAC, M9K và **số chu kỳ lõi đo trên kit**, trong khi sai khác accuracy trên test thật nhỏ. Bản pruned đạt 96,32% ký tự nhưng chỉ 79,67% nguyên biển trên crop thật và 75,67% khi bắt đầu từ hộp biển đã chú thích.
- **Chưa đạt:** mục tiêu 95–99% đúng nguyên biển thật, nhận dạng ảnh camera trực tiếp, xử lý đầy đủ trên FPGA và streaming thời gian thực. Nạp `.sof` qua JTAG là phép đo tạm thời; không phải cấu hình flash lưu vĩnh viễn. Fmax vượt 50 MHz nhưng chưa timing sign-off do đường không ràng buộc.
- **Nguyên nhân sai lệch:** ảnh dựng dùng font/khung/nền hẹp hơn phân bố ảnh thật (giai đoạn dựng-only chỉ đúng 2/300 biển thật); ký tự tương tự nhau và biển xe máy 9 ký tự có nhiều cơ hội sai; phân đoạn hai hàng còn tạo sai độ dài hoặc bỏ ký tự; lượng tử hóa và truyền JTAG là các nguồn sai/độ trễ riêng. Cần cải thiện dữ liệu thật, đánh giá theo biển/nguồn camera độc lập, bộ phát hiện/rectification/segmentation trên luồng thực, và kiến trúc truyền ảnh không qua JTAG chậm.

Tệp đối chứng tại máy: `artifacts/three_layout_dense_pruned_20261009/metrics.json`, `history.csv`, `board_results.json`, `dataflow_functional.json`, `dense.pt`, `pruned.pt`; log Quartus và CSV JTAG tại `hardware/dataflow_study/rtl_eval/three_layout_20261009/`. Tệp `board_results_partial_50.csv` là lượt chạy dở dang với wrapper 50 ID, **không** dùng trong các số liệu kết luận ở trên.
