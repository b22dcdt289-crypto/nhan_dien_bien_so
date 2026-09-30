# Train lại LeNet-5 chỉ với 2.000 biển ô tô đã tạo

Ngày chạy: 01/10/2026. Đây là **lần train mới**, seed `20261003`, tách riêng hoàn toàn khỏi lần fine-tune bằng ảnh chụp thật. Nguồn ảnh duy nhất được đọc để train/validation/test là `data/car_frontal_synthetic_2x1000_v1/` (1.000 biển ô tô 1 hàng và 1.000 biển ô tô 2 hàng). Các ảnh này là **ảnh frontal dựng từ nhãn dữ liệu cũ**, không phải ảnh camera. Thử nghiệm còn truyền một đường dẫn ảnh thật **không tồn tại** cho tùy chọn `--real`; cờ `--synthetic-only` khiến chương trình không đọc đường dẫn đó. JSON ghi `synthetic_only: true`, `real_external_test_plates: 0` và không có kết quả ảnh thật.

## Thiết kế thí nghiệm

| Tập | 1 hàng | 2 hàng | Tổng |
|---|---:|---:|---:|
| Train | 800 | 800 | 1.600 biển / 12.800 ký tự |
| Validation | 100 | 100 | 200 biển / 1.600 ký tự |
| Test giữ riêng | 100 | 100 | 200 biển / 1.600 ký tự |

2.000 nhãn biển khác nhau; các tập chia theo chuỗi biển, không có cùng chuỗi ở hai tập. Mỗi biển của bộ này có **8 ký tự OCR**, bất kể một hay hai hàng. Mô hình dense và mô hình structured pruning dùng cùng split và augmentation (xoay ±4°, thay đổi sáng và nhiễu mức thấp). Batch 256, AdamW, learning rate `7e-4`, weight decay `1e-5`, cross-entropy. Dense học 12 epoch; sau epoch 4, lấy 8/16 filter Conv2 theo chuẩn L1 để tạo mô hình pruned, giảm đồng thời chiều vào FC1, rồi huấn luyện tới epoch 12. Checkpoint chọn theo accuracy ký tự validation, sau đó mới kiểm tra tập test giữ riêng. Không dùng ảnh thật ở bất kỳ giai đoạn nào trong lần chạy này.

## Kết quả đo trên **ảnh được tạo**

| Chỉ số | Dense, không pruning | Structured pruning Conv2 50% |
|---|---:|---:|
| Validation, epoch checkpoint tốt nhất | 1.600/1.600 ký tự = 100% | 1.600/1.600 = 100% |
| Test ký tự | 1.600/1.600 = **100%** | 1.600/1.600 = **100%** |
| Test biển ô tô 1 hàng | 100/100 = **100%** | 100/100 = **100%** |
| Test biển ô tô 2 hàng | 100/100 = **100%** | 100/100 = **100%** |
| Test toàn biển | 200/200 = **100%** | 200/200 = **100%** |
| Tách ký tự trên 200 ảnh test | 200/200 thành công | 200/200 thành công |
| Sai ký tự / sai toàn biển trên test | 0 / 0 | 0 / 0 |

Đây là số liệu **chỉ trên cùng họ ảnh dựng trực diện**. Tỉ lệ 100% không chứng minh mô hình đạt 100% với ảnh thật, góc nghiêng, motion blur, ánh sáng khác hoặc camera. Tập test chỉ có **18/30 lớp ký tự**; toàn bộ 2.000 biển có **22/30 lớp**. Các lớp `K,N,P,T,U,V,Y,Z` không xuất hiện trong cả bộ; `L,M,S,X` có trong bộ nhưng không xuất hiện ở test. Lớp `3` có 2.308 ký tự, trong khi `G` chỉ có 6 trên cả bộ. Vì vậy không thể tuyên bố accuracy cho đủ 30 lớp. Xem [phân bố từng lớp theo train/val/test](results/synthetic_only_2000_v2/class_distribution.csv) và [ma trận nhầm lẫn dense](results/synthetic_only_2000_v2/dense_synthetic_confusion.csv), [pruned](results/synthetic_only_2000_v2/pruned_synthetic_confusion.csv).

## Độ phức tạp, trọng số, tốc độ CPU

| Chỉ số | Dense | Pruned | Chênh lệch |
|---|---:|---:|---:|
| Conv2 filter | 16 | 8 | −50% tại Conv2 |
| Tổng weight | 63.150 | 37.950 | −39,90% |
| Tổng tham số gồm bias | 63.406 | 38.198 | −39,76% |
| FP32 lưu thô | 253.624 byte | 152.792 byte | −39,76% |
| INT8 weight + INT32 bias | 64.174 byte | 38.942 byte | −39,32% |
| MAC/ký tự | 418.200 | 274.200 | −34,43% |
| MAC/biển 8 ký tự | 3.345.600 | 2.193.600 | −34,43% |
| CPU PyTorch, 1 glyph, p50/1.000 lần xen kẽ | 0,1730 ms | 0,1715 ms | khoảng 1,009× |
| CPU PyTorch, batch 8 glyph, p50/1.000 lần | 0,24325 ms | 0,23575 ms | khoảng 1,032× |

Benchmark CPU chỉ tính forward trên tensor 32×32, **không** gồm tạo ảnh, tách ký tự, truyền ảnh hay FPGA. Tốc độ CPU không tăng theo tỷ lệ giảm MAC vì overhead framework và Conv1/FC2/FC3 vẫn giữ nguyên.

## Đo trực tiếp trên DE10-Lite với checkpoint của lần train này

Quartus Prime Lite 25.1 compile riêng hai bản frame-stream dense/pruned. Programmer báo `Configuration succeeded` với device JTAG ID `0x031050DD` cho **cả hai bitstream**. Sau mỗi lần nạp, Tcl gửi toàn bộ **1.600 crop 32×32 từ 200 biển synthetic test giữ riêng** (100 biển một hàng, 100 biển hai hàng) qua USB-Blaster/JTAG. Cả hai lượt trả đủ mẫu, sample sequence liên tục qua các lần quay vòng slot 7-bit và không timeout. Đây là số đo vật lý trên kit, không phải mô phỏng.

| Số đo | Dense | Pruned Conv2 50% | Cách đọc |
|---|---:|---:|---|
| Ký tự đúng trên board | **1.600/1.600 (100%)** | **1.600/1.600 (100%)** | Cùng 1.600 crop synthetic |
| Biển đúng trọn vẹn | **200/200 (100%)** | **200/200 (100%)** | 100 biển một hàng + 100 biển hai hàng mỗi bản |
| FPGA trùng fixed-point reference | **1.600/1.600** | **1.600/1.600** | Đối chiếu từng ký tự |
| Đúng biển theo bố cục một hàng | **100/100** | **100/100** | Ghép đúng 8 ký tự theo nhãn test |
| Đúng biển theo bố cục hai hàng | **100/100** | **100/100** | Ghép đúng 8 ký tự theo nhãn test |
| Chu kỳ lõi/ký tự, đọc probe FPGA | **870.198** | **577.998** | Không đổi qua 1.600 mẫu mỗi nhánh |
| Compute/ký tự ở clock 50 MHz | **17,40396 ms** | **11,55996 ms** | cycles ÷ 50 MHz; khớp phép đổi đơn vị của probe |
| Nạp ảnh JTAG, p50 / p95 | **1,001 / 1,006 ms** | **1,001 / 1,006 ms** | Trên 1.600 crop, CSV ghi millisecond nguyên |
| Cả nạp + OCR, p50 / p95 | **1,075 / 1,081 ms** | **1,060 / 1,066 ms** | Mỗi crop; gồm giao tiếp JTAG/Quartus |
| Quartus logic elements | **2.332 / 49.760** | **2.301 / 49.760** | Post-fit |
| Quartus total memory bits | **560.432 bit** | **358.832 bit** | Post-fit, giảm 35,97%; không đồng nhất với số block RAM |
| Registers / multiplier 9-bit | **1.068 / 1** | **1.068 / 1** | Post-fit; một multiplier MAC nối tiếp |
| Quartus Fmax chậm 85°C | **64,00 MHz** | **68,65 MHz** | Static timing estimate, không phải frequency đo bằng counter trên kit |

Tập này chỉ có **18/30 lớp ký tự** trong alphabet; ma trận confusion được lưu đủ 30×30 nhưng các lớp không có mẫu test không thể được đánh giá. 100% ở đây chỉ áp dụng cho bộ ảnh dựng frontal này, không chứng minh độ chính xác trên ảnh camera/biển thật. “Độ chính xác toàn biển” được tính bằng cách ghép đúng tám dự đoán ký tự theo thứ tự crop đã có sẵn; phép đo không gồm tìm biển, perspective correction hay phân loại/sắp dòng trên kit.

Từ probe, structured pruning giảm chu kỳ compute **33,58%** và tăng tốc lõi đo bằng cycles **1,506×**. MAC lý thuyết giảm **34,43%** (418.200 xuống 274.200 MAC/crop); tổng memory bits post-fit giảm **35,97%**. Dù vậy, lượt JTAG + OCR chỉ nhanh hơn khoảng **1,014×** theo p50 vì thời gian truyền ảnh (~1 giây/crop) lấn át compute (17,40/11,56 ms). Cả hai thiết kế dùng cùng một MAC nối tiếp; output/psum nằm trong thanh ghi `acc`, nên đây chỉ là baseline gần output-stationary. Không có phiên bản RTL riêng cho weight-stationary, row-stationary, PE song song hay pixel-stream + line-buffer trong phép đo này.

**Khuyến nghị theo phép đo hiện có:** dùng structured pruning như ứng viên giảm compute và tổng bộ nhớ nội chip cho DE10-Lite; giữ điều kiện phải thử lại trên ảnh camera thật trước khi kết luận accuracy. Không dùng JTAG frame streaming làm đường truyền sản phẩm realtime: phép đo trực tiếp cho thấy khoảng một giây mỗi crop. Giữ lõi MAC nối tiếp làm baseline. Chưa chọn WS/RS hoặc mạng multicast tái sử dụng dữ liệu theo số liệu phần cứng vì chưa có bản RTL riêng được nạp và đo; cần dựng hai biến thể đó mới so được công bằng.

Quartus báo compile thành công (dense 0 lỗi/32 cảnh báo; pruned 0 lỗi/33 cảnh báo). Fmax và slack là post-fit; STA vẫn cảnh báo thiết kế chưa fully constrained cho setup/hold. Không coi Fmax là phép đo tốc độ chạy của ứng dụng. Các lượt Signal Tap mất **28:45** dense và **28:16** pruned để gửi/nhận 1.600 crop; thời lượng lượt phụ thuộc giao tiếp PC–JTAG.

Raw file để kiểm chứng trên máy:

- [CSV kết quả board dense](rtl_eval/synthetic_only_fulltest_20261001/dense_stream/generated/board_results.csv) và [CSV pruned](rtl_eval/synthetic_only_fulltest_20261001/pruned_stream/generated/board_results.csv); kèm [manifest mẫu dense](rtl_eval/synthetic_only_fulltest_20261001/dense_stream/generated/manifest.json) và [manifest pruned](rtl_eval/synthetic_only_fulltest_20261001/pruned_stream/generated/manifest.json) để đối chiếu nhãn synthetic.
- [JSON tổng hợp hardware, gồm confusion matrix 30×30 và metrics theo lớp](results/synthetic_only_2000_v2/hardware_results.json).
- [Fit report dense](rtl_eval/synthetic_only_fulltest_20261001/dense_stream/output_files/OcrBench.fit.summary), [fit report pruned](rtl_eval/synthetic_only_fulltest_20261001/pruned_stream/output_files/OcrBench.fit.summary); timing tương ứng trong `OcrBench.sta.rpt`.

## Tệp kết quả và cách chạy lại

- Checkpoint của lần train này: [dense.pt](../../artifacts/car2000_synthetic_only_20261001_v2/dense.pt), [pruned.pt](../../artifacts/car2000_synthetic_only_20261001_v2/pruned.pt). SHA-256 của chúng ghi trong [summary.json](results/synthetic_only_2000_v2/summary.json).
- [Metrics đầy đủ](results/synthetic_only_2000_v2/metrics.json), [history theo epoch](results/synthetic_only_2000_v2/history.csv), [benchmark CPU xen kẽ](results/synthetic_only_2000_v2/cpu_interleaved.json). Tổng thời gian train trước phần đánh giá khoảng **23,66 giây** trên CPU với 4 threads; đây là thời gian cả hai nhánh, không phải thời gian train độc lập mỗi model.
- Mã train là [train_synthetic_car_compare.py](../../train_synthetic_car_compare.py), mã kiểm tra/phát hành kết quả là [publish_synthetic_only.py](publish_synthetic_only.py).

Trong VS Code, mở thư mục `D:\TOTNGHIEP\license_plate_ai`, chọn Python `.venv\Scripts\python.exe`, sau đó chạy ở Terminal tích hợp:

```powershell
.\.venv\Scripts\python.exe train_synthetic_car_compare.py --synthetic-only --audit-only --seed 20261003
.\.venv\Scripts\python.exe train_synthetic_car_compare.py --synthetic-only --data data/car_frontal_synthetic_2x1000_v1 --real data/NOT_USED_IN_SYNTHETIC_ONLY --output artifacts/car2000_synthetic_only_repeat --seed 20261003 --epochs 12 --prune-after 4 --threads 4
```

Chọn `--output` mới khi chạy lại để giữ nguyên checkpoint/số liệu hiện tại. **Không so trực tiếp 100% ở đây với độ chính xác ảnh thật**; muốn biết khả năng thực tế phải làm bộ test ảnh thật độc lập ở bước sau, không dùng nó để train trong thí nghiệm này.
