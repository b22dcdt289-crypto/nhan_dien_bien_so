# Đối chứng kernel 3×3 và 5×5 của bộ nhận dạng ký tự LeNet-style trên DE10-Lite

Ngày thí nghiệm: **09/10/2026**. Thiết bị: Terasic DE10-Lite, MAX 10 `10M50DAF484C7G`, clock board 50 MHz, Quartus Prime Lite 25.1std.0. Kết quả 3×3 bên dưới được huấn luyện mới, tổng hợp/mapping/place-and-route bằng Quartus, nạp `.sof` qua USB-Blaster và chạy trực tiếp trên kit. Bản 5×5 là đối chứng đã đo ngày 09/10/2026 với cùng tập dữ liệu, cùng cách chia tập và cùng quy trình train.

## 1. Giải thích kiến trúc

**Conv, pooling và fully connected (FC) là ba *loại tầng*, không có nghĩa LeNet chỉ có ba tầng và cũng không quyết định kích thước kernel.** Bài báo LeNet-5 gốc dùng các tầng tích chập C1/C3/C5 với kernel 5×5. Cấu hình đang làm là một biến thể LeNet-style đơn giản hóa; phép thử mới thay **cả Conv1 và Conv2 từ 5×5 sang 3×3**, giữ ảnh ký tự 32×32, 6 filter Conv1, 16 filter Conv2, average pooling, Tanh và bộ phân loại 120→84→30. Nhánh tỉa kênh chỉ giữ 8/16 filter Conv2; các cột tương ứng của FC1 được thu gọn thật, không phải mask số 0.

| Luồng kích thước | Kernel 5×5 | Kernel 3×3 |
|---|---|---|
| Ảnh đầu vào | 1×32×32 | 1×32×32 |
| Conv1 → AvgPool | 6×28×28 → 6×14×14 | 6×30×30 → 6×15×15 |
| Conv2 → AvgPool, dense | 16×10×10 → 16×5×5 | 16×13×13 → 16×6×6 |
| Đầu vào FC1, dense / tỉa | 400 / 200 | 576 / 288 |
| FC | 120 → 84 → 30 | 120 → 84 → 30 |

Vì feature map sau pool của 3×3 lớn hơn, FC1 **nhiều trọng số hơn** dù Conv ít MAC hơn. Do đó không thể suy từ “kernel nhỏ hơn” ra “mọi tài nguyên FPGA đều giảm”. Tài liệu gốc: [LeCun và cộng sự, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 1998](https://leon.bottou.org/papers/lecun-98h).

## 2. Dữ liệu và huấn luyện đối chứng

- Ảnh dựng: 1.000 biển ô tô một hàng, 1.000 biển ô tô hai hàng trong `data/car_frontal_synthetic_2x1000_v1/`; 1.000 biển xe máy hai hàng trong `data/motorcycle_frontal_synthetic_1000_v2/`. Theo từng nhóm: 800 train, 100 validation, 100 test. Biển ô tô có 8 ký tự; biển xe máy trong bộ này có 9 ký tự.
- Ảnh thật đã chuẩn bị và gán nhãn ký tự: `data/independent_chars_train1_structured_channel30_v1/`, nguồn từ bộ train(1) cũ. Mỗi nhóm 600 train, 50 validation, 100 test; tổng 1.800/150/300 biển. Tập test thật có 2.500 ký tự. Định danh biển không trùng giữa train/validation/test. **Đây không phải thí nghiệm chỉ train bằng 3.000 ảnh dựng**: sau pretrain ảnh dựng, hai nhánh đều fine-tune thêm ảnh thật. Không tuyên bố ảnh thật đã được kiểm tra thủ công toàn bộ.
- Seed `20261009`, cùng tệp manifest dữ liệu và cùng cách chia như 5×5. Pretrain 8 epoch; sau epoch 4 tỉa cấu trúc 50% filter Conv2 để tạo nhánh pruned; fine-tune hỗn hợp tối đa 10 epoch. Checkpoint chọn theo accuracy ký tự, rồi accuracy nguyên biển trên validation ảnh thật; test chỉ đánh giá sau khi chọn. 3×3 dense và pruned đều chọn epoch fine-tune thứ 10.
- Bộ test từ *crop ký tự đúng sẵn* tách biệt với phép đo từ *hộp biển chú thích rồi tự phân đoạn*. Không có phép đo nhận dạng tự động từ toàn khung camera.

Lệnh chạy lại (tại `D:\TOTNGHIEP\license_plate_ai`):

```powershell
.\.venv\Scripts\python.exe train_three_layout_compare.py --kernel-size 3 --output artifacts\three_layout_k3_dense_pruned_20261009 --threads 4
.\.venv\Scripts\python.exe hardware\dataflow_study\export_three_layout_hardware.py --kernel-size 3 --models artifacts\three_layout_k3_dense_pruned_20261009 --out-root hardware\dataflow_study\rtl_eval\three_layout_k3_20261009
```

## 3. Độ chính xác đo được

Mọi cột 5×5 và 3×3 đều đo trên cùng **300 biển/2.500 ký tự test thật**; mẫu số khác nhau ở số ký tự vì xe máy có 9 ký tự.

| Mô hình | Đúng ký tự trên crop chuẩn | Đúng nguyên biển trên crop chuẩn | Đúng nguyên biển từ hộp biển chú thích | Test ảnh dựng, nguyên biển |
|---|---:|---:|---:|---:|
| 5×5 dense | 2.401/2.500 = 96,04% | 236/300 = 78,67% | 224/300 = 74,67% | 299/300 = 99,67% |
| 5×5 tỉa Conv2 50% | 2.408/2.500 = 96,32% | 239/300 = 79,67% | 227/300 = 75,67% | 299/300 = 99,67% |
| **3×3 dense** | **2.415/2.500 = 96,60%** | **246/300 = 82,00%** | **236/300 = 78,67%** | **299/300 = 99,67%** |
| **3×3 tỉa Conv2 50%** | **2.409/2.500 = 96,36%** | **246/300 = 82,00%** | **235/300 = 78,33%** | **299/300 = 99,67%** |

Chi tiết 3×3 trên crop thật:

| Nhóm, 100 biển/nhóm | Dense: đúng ký tự; đúng biển | Tỉa: đúng ký tự; đúng biển | Dense / tỉa: đúng biển từ hộp chú thích |
|---|---:|---:|---:|
| Ô tô 1 hàng (8 ký tự) | 789/800; 93/100 | 787/800; 92/100 | 93/100; 92/100 |
| Ô tô 2 hàng (8 ký tự) | 773/800; 85/100 | 773/800; 85/100 | 83/100; 83/100 |
| Xe máy 2 hàng (9 ký tự) | 853/900; 68/100 | 849/900; 69/100 | 60/100; 60/100 |

Phân đoạn từ hộp biển tạo chuỗi đúng độ dài ở 297/300 biển. Sai số chỉnh sửa chuỗi trên 2.500 ký tự: dense 114/2.500 = 4,56%; tỉa 117/2.500 = 4,68%. Tỉ lệ ký tự tính *chỉ trên 2.296 ký tự của biển có chuỗi đúng độ dài* không được trình bày như accuracy đầu-cuối. Kết quả rất cao trên ảnh dựng nhưng thấp đáng kể trên ảnh thật cho thấy **chênh miền dữ liệu**; 82% nguyên biển chưa đáp ứng mục tiêu 95–99%. Chênh vài biển trên 300 là quan sát của lượt chạy này, chưa đủ kết luận 3×3 có ưu thế chính xác ổn định qua nhiều seed/bộ test độc lập.

## 4. Tính toán mô hình và tài nguyên vật lý

MAC/ký tự là phép đếm Conv + FC từ kích thước tensor, **không phải** số chu kỳ hoặc số DSP. Một biển 8/9 ký tự nhân tương ứng 8/9 lần phần OCR; chưa tính cắt biển, chỉnh phối cảnh, tách dòng, tách ký tự và truyền dữ liệu.

| Chỉ số | 5×5 dense | 5×5 tỉa | 3×3 dense | 3×3 tỉa |
|---|---:|---:|---:|---:|
| Trọng số | 63.150 | 37.950 | **82.638** | **47.646** |
| INT8 trọng số + INT32 bias, byte | 64.174 | 38.942 | **83.662** | **48.638** |
| MAC/ký tự | 418.200 | 274.200 | **276.336** | **168.768** |
| MAC/biển 8 ký tự | 3.345.600 | 2.193.600 | **2.210.688** | **1.350.144** |
| MAC/biển 9 ký tự | 3.763.800 | 2.467.800 | **2.487.024** | **1.518.912** |
| Logic elements post-fit / 49.760 | 2.318 | 2.334 | **2.391** | **2.345** |
| Combinational functions / 49.760 | 1.770 | 1.786 | **1.840** | **1.796** |
| M9K block RAM post-fit / 182 | 75 | 51 | **92** | **59** |
| Memory bits post-fit / 1.677.312 | 560.432 | 358.832 | **723.296** | **443.360** |
| DSP 9-bit multiplier / 288 | 1 | 1 | **1** | **1** |
| Fmax slow 85 °C, MHz | 66,53 | 65,69 | **62,68** | **65,92** |
| Chu kỳ xử lý/ký tự trên kit | 870.198 | 577.998 | **595.020** | **373.236** |
| Thời gian tính ở clock 50 MHz, ms/ký tự | 17,404 | 11,560 | **11,900** | **7,465** |
| Thời gian toàn lượt JTAG trung vị, ms/ký tự | 860 | 876 | **840** | **812** |

Ví dụ cách tính 3×3 dense: Conv1 `6×30×30×(1×3×3)=48.600`; Conv2 `16×13×13×(6×3×3)=146.016`; FC1 `120×(16×6×6)=69.120`; FC2 `120×84=10.080`; FC3 `84×30=2.520`. Tổng **276.336 MAC/ký tự**. Bản tỉa thay 16 bằng 8 ở Conv2 và FC1, thành **168.768 MAC/ký tự**. Trong RTL hiện tại chỉ có **một** datapath MAC tuần tự tích lũy theo đầu ra; vì vậy số chu kỳ không bằng chính xác số MAC và cấu hình tỉa **không** làm DSP giảm. 3×3 dense so với 5×5 dense giảm 33,92% MAC và 31,62% chu kỳ tính, nhưng tăng 17 M9K; 3×3 tỉa so với 5×5 tỉa giảm 38,45% MAC và 35,43% chu kỳ, nhưng tăng 8 M9K. Với hai bản 3×3, tỉa Conv2 giảm 38,93% MAC, 33 M9K và 37,27% chu kỳ so với dense; khác biệt timing JTAG nhỏ hơn vì thời gian truyền 1.024 pixel qua USB-Blaster chiếm phần lớn.

Clock 50 MHz có chu kỳ 20 ns. Quartus báo setup slack tại góc slow 85 °C dương: 3×3 dense **+4,047 ns**, tỉa **+4,830 ns**. Fmax post-fit tương ứng đều >50 MHz. Tuy nhiên báo cáo `timing_fully_constrained=false`: các đường chưa được ràng buộc toàn phần, nên đây **chưa phải chứng nhận timing sign-off** cho mọi đường và càng chưa chứng minh hệ thống camera thời gian thực.

## 5. Nạp và kiểm tra kit

Quartus Programmer phát hiện `USB-Blaster [USB-0]`, JTAG ID `0x031050DD`. Nạp `.sof` của 3×3 dense và 3×3 tỉa đều báo `Configuration succeeded`. Mỗi bản đã chạy **100 ảnh ký tự 32×32** chọn cố định từ **12 biển** (2 ảnh dựng + 2 ảnh thật đã crop cho mỗi nhóm). Cả hai bản: **100/100 dự đoán FPGA trùng tham chiếu INT8 phần mềm**, đúng nhãn thật **99/100 ký tự**, đúng nguyên biển **11/12** trong mẫu nhỏ này. Mẫu 12 biển quá nhỏ để suy rộng thành accuracy toàn bộ hệ thống; dùng bảng 300 biển thật ở mục 3 cho đánh giá mô hình.

Lượt đo kit truyền từng ảnh ký tự 32×32 từ máy tính qua JTAG rồi nhận lớp dự đoán. Laptop **vẫn làm khâu cấp crop và điều khiển JTAG**; đây chưa phải camera→FPGA→biển số độc lập. Tốc độ JTAG toàn lượt (khoảng 0,8 s/ký tự) **không** đạt thời gian thực, dù lõi OCR trên FPGA có thời gian tính 7,5–11,9 ms/ký tự. Không được gọi OS/WS/RS là ba phần cứng đã đo: ở lượt này các lịch OS/WS/RS chỉ được kiểm tra tương đương số học trong phần mềm; RTL là MAC nối tiếp và đầu vào được streaming qua JTAG. Để đánh giá camera thực phải triển khai cổng pixel/line buffer, phân đoạn biển trên FPGA, đường truyền ảnh khả dụng và đo thông lượng đầu-cuối.

## 6. Kết luận kỹ thuật và truy xuất số đo

Kernel 5×5 **không quá tải** DE10-Lite trong thiết kế này: cả 5×5 dense và pruned đều fit và đã chạy trên kit. Chuyển sang 3×3 giảm MAC/chu kỳ nhưng **tăng trọng số và M9K** do đầu vào FC1 rộng hơn; không nên chọn kernel chỉ theo phép đếm MAC. Nếu ưu tiên tốc độ lõi OCR, 3×3 tỉa đạt 373.236 chu kỳ/ký tự; nếu ưu tiên M9K trong hai bản tỉa, 5×5 dùng 51 thay vì 59 M9K. Cả hai còn thiếu accuracy nguyên biển thật mong muốn và chưa đạt một hệ thống camera độc lập thời gian thực.

- Số liệu train, confusion, history, checkpoint: `hardware/dataflow_study/results/three_layout_k3_20261009/`.
- Số liệu nạp và chạy kit: `board_results.json`; post-fit/Fmax lấy từ báo cáo Quartus của hai dự án ở `hardware/dataflow_study/rtl_eval/three_layout_k3_20261009/{dense_stream,pruned_stream}/output_files/` trên máy này. SHA-256 `.sof`: dense `cfe14b589170ffb5798c7da164bcbe7af33b4a603158df0a923c32f8f9663cff`; tỉa `96d3f374013aba167c72b120300bf88af8babcd608511167334d8b5f07e58e04`.
- Báo cáo đối chứng 5×5: `hardware/dataflow_study/BAO_CAO_3_LOAI_BIEN_20261009.md` và `hardware/dataflow_study/results/three_layout_20261009/`.

Không nạp đồng thời hai mô hình vào kit; sau thí nghiệm, `.sof` của bản 3×3 tỉa đang là cấu hình SRAM hiện hành và sẽ mất khi tắt nguồn hoặc nạp lại.
