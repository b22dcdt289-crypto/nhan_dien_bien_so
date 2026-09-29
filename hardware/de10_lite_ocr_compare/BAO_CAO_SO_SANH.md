# Đối chứng LeNet-5 dense, structured Conv2 pruning và JTAG frame streaming

Lần tổng hợp báo cáo: 2026-09-29T15:22:53. Cùng tập `train(1)` đã chuẩn bị: 180,768 crop train, 20,086 crop validation từ 2,505 biển. Dùng 30 lớp ký tự.

## Train (CPU laptop, không phải tài nguyên FPGA)

Cả hai nhánh xem dữ liệu 10 epoch: 5 epoch dense chung; sau đó dense tiếp tục 5 epoch, còn bản tỉa Conv2 từ 16 xuống 8 kênh và train tiếp 5 epoch. Không KD; cùng loss, optimizer, batch, seed và augmentation. Streaming chỉ là cách nạp ảnh khi suy luận, không phải kỹ thuật train.

| Mục | Dense | Tỉa kênh Conv2 |
|---|---:|---:|
| Tham số | 63,406 | 38,198 |
| MAC lý thuyết/ký tự | 418,200 | 274,200 |
| Best epoch | 10 | 10 |
| Validation crop ký tự (best) | 92.29% | 92.02% |
| Validation toàn biển trên crop | 83.79% (2099/2505) | 82.95% (2078/2505) |
| Train + validation (gồm 5 epoch chung) | 286.00 s | 268.77 s |
| CPU forward p50 batch 1 | 0.1819 ms | 0.1779 ms |
| CPU forward p50 batch 8 | 0.2932 ms | 0.2743 ms |

Thời gian toàn thí nghiệm (nạp data + train hai nhánh + đánh giá): 558.86 s; đỉnh RSS của **tiến trình chứa cả hai model và cache ảnh**: 859.5 MiB. Không gán con số RAM này cho riêng từng model. Chi tiết từng epoch trong `artifacts/compare_conv2_stream_20260929/history.csv`.

MAC/ký tự = Conv1 `6×25×28×28` + Conv2 `6×C2×25×10×10` + FC1 `C2×25×120` + FC2 `120×84` + FC3 `84×30`. Với `C2=16`: 117.600 + 240.000 + 48.000 + 10.080 + 2.520 = 418.200 MAC; với `C2=8`: 117.600 + 120.000 + 24.000 + 10.080 + 2.520 = 274.200 MAC. Không tính phép cộng bias, tanh và pooling vào MAC; đó là phép tính khác.

| Nhóm biển validation | Dense: ký tự đúng | Dense: biển đúng | Pruned: ký tự đúng | Pruned: biển đúng |
|---|---:|---:|---:|---:|
| Ô tô một hàng | 11203/11393 (98.33%) | 1347/1441 (93.48%) | 11183/11393 (98.16%) | 1337/1441 (92.78%) |
| Ô tô hai hàng | 5392/5711 (94.41%) | 613/723 (84.79%) | 5386/5711 (94.31%) | 609/723 (84.23%) |
| Xe máy hai hàng | 1943/2982 (65.16%) | 139/341 (40.76%) | 1914/2982 (64.19%) | 132/341 (38.71%) |

Tổng validation và từng nhóm này là kết quả **crop ký tự với phân đoạn dựa vào nhãn**, không phải OCR biển nguyên ảnh/camera. Mục tiêu 95,22% biển đúng trên validation toàn bộ chưa đạt, nhất là nhóm xe máy hai hàng.

## Kết quả sau fit và chạy trên DE10-Lite (50 crop từ 6 biển)

| Chỉ số | Dense ROM | Dense stream | Pruned ROM | Pruned stream |
|---|---:|---:|---:|---:|
| Ký tự đúng FPGA | 48/50 | 48/50 | 48/50 | 48/50 |
| Biển đúng toàn bộ | 5/6 | 5/6 | 5/6 | 5/6 |
| Khớp mô phỏng INT8 | 50/50 | 50/50 | 50/50 | 50/50 |
| LE Quartus | 1,749/49,760 | 2,356/49,760 | 1,721/49,760 | 2,324/49,760 |
| RAM bit Quartus | 961,840/1,677,312 | 560,432/1,677,312 | 760,240/1,677,312 | 358,832/1,677,312 |
| Phần tử nhân 9-bit | 1/288 | 1/288 | 1/288 | 1/288 |
| Chu kỳ/ký tự | 870,198 | 870,198 | 577,998 | 577,998 |
| Tính toán/ký tự ở 50 MHz | 17.404 ms | 17.404 ms | 11.560 ms | 11.560 ms |
| JTAG nạp ảnh p50 | 0.0 ms | 1000.0 ms | 0.0 ms | 1000.0 ms |
| JTAG toàn lượt p50 | 654.0 ms | 1074.0 ms | 656.0 ms | 1059.0 ms |
| Fmax post-fit slow 85°C | 63.58 MHz | 61.11 MHz | 62.13 MHz | 63.16 MHz |
| Setup slack @50 MHz | 4.271 ns | 3.635 ns | 3.905 ns | 4.168 ns |
| Trọng số INT8 | 63,150 byte | 63,150 byte | 37,950 byte | 37,950 byte |
| Ảnh test nhúng ROM | 51,200 byte | 0 byte | 51,200 byte | 0 byte |
| Bộ đệm ảnh stream | 0 byte | 1,024 byte | 0 byte | 1,024 byte |

Giảm MAC do tỉa kênh: 34.43%. Tăng tốc **lõi tính toán FPGA** pruned ROM so với dense ROM: 1.506×. Streaming chỉ thay đường nạp ảnh; không được gọi là tăng tốc train.

Trong cặp **cùng truyền ảnh**, pruning giảm RAM nội bộ 35.97% (560,432 → 358,832 bit), nhưng thời gian JTAG toàn lượt p50 chỉ đổi từ 1074.0 thành 1059.0 ms/ký tự: đường truyền chậm lấn át lợi ích lõi. So với Dense ROM, bản Pruned stream giảm 62.69% RAM nhưng tăng 32.88% LE vì có logic nhận JTAG và bộ đệm. Hai phép so này có phạm vi khác nhau.

Thời gian lõi cho biển N ký tự, nếu xử lý nối tiếp, xấp xỉ `N × chu kỳ/ký tự ÷ 50 MHz`; chưa tính cắt ảnh, giao tiếp, điều khiển. Với 8 ký tự: dense 139.232 ms, pruned 92.480 ms. Với giao tiếp JTAG thử nghiệm, thời gian thực tế lớn hơn đáng kể; không dùng phép tính lõi này để tuyên bố tốc độ hệ thống camera.

## Phạm vi và cách kiểm chứng

- Mở `comparison_full.json` để xem số đầy đủ, checksum, breakdown; `train_history.csv` để xem từng epoch.
- Mỗi cấu hình có `generated/board_results.csv` (raw JTAG), `generated/manifest.json` (nhãn và dự đoán tham chiếu, **chỉ lưu trên laptop**), `output_files/OcrBench.fit.summary` và `output_files/OcrBench.sta.rpt`.
- Streaming ở đây là nạp **một ảnh ký tự đã cắt** qua JTAG vào RAM 1 KiB; không phải camera pixel stream, không tăng tốc bộ MAC. Thời gian JTAG phụ thuộc USB-Blaster và thời gian chờ handshake trong Tcl, không đại diện cho giao tiếp triển khai cuối.
- ROM gắn sẵn 50 ảnh test nên RAM của nó lớn hơn; chênh lệch tài nguyên không phải chỉ do pruning. Tập FPGA 6 biển quá nhỏ để công bố accuracy thực tế.
- Validation dùng crop tham chiếu với số ký tự lấy từ nhãn nguồn, có nguy cơ trùng biển giữa train/validation. Fmax/slack chưa bảo đảm timing được ràng buộc đầy đủ.
