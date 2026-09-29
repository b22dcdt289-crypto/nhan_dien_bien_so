# Kết quả trực tiếp trên DE10-Lite: OCR ký tự (demo tham chiếu)

- Tập mẫu: 6 biển, 50 ký tự; chọn ngẫu nhiên cố định seed 20260929 từ validation `train(1)`.
- SHA-256 checkpoint: `d94c835c5b81e46eeeaa69ae6a7fdae14fa4da893fef36e12bbc0c164696f865`.
- FPGA đúng ký tự: **48/50 = 96.00%**; khoảng Wilson 95% 86.54%–98.90%.
- FPGA đúng nguyên biển trên crop tham chiếu: **5/6 = 83.33%**; khoảng Wilson 95% 43.65%–96.99%.
- FPGA trùng mô phỏng số nguyên: 50/50 ký tự.
- Hai lượt chạy liên tiếp trùng dự đoán và chu kỳ: 50/50 mẫu.

| Loại biển | Ký tự đúng | Biển đúng hoàn toàn |
|---|---:|---:|
| Hai hàng xe máy | 16/18 | 1/2 |
| Một hàng ô tô | 16/16 | 2/2 |
| Hai hàng ô tô | 16/16 | 2/2 |

| Thông số trực tiếp | Kết quả |
|---|---:|
| LE Quartus sau fit | 1,658/49,760 (3.33%) |
| RAM nội bộ sau fit | 760,240/1,677,312 bit (45.32%) |
| Phần tử nhân 9-bit sau fit | 1/288 |
| Fmax ước lượng sau fit (slow 85°C) | 65.93 MHz |
| Setup slack tại xung 50 MHz (slow 85°C) | 4.833 ns |
| Chu kỳ đọc từ bộ đếm FPGA | 577,998/ký tự |
| MAC lý thuyết của LeNet-5 đã tỉa Conv2 | 274.200/ký tự |
| Thời gian tính ở 50 MHz | 11.560 ms/ký tự |
| Ước tính 8 ký tự tuần tự, chỉ lõi OCR | 92.480 ms/biển |

RAM Quartus ở đây **bao gồm cả ảnh mẫu nhúng trong ROM**; không phải chỉ riêng trọng số mạng. Trọng số INT8 có 37,950 byte; 51,200 byte ảnh thử được đóng trong bitstream.

Giới hạn: đây là lõi phân loại crop ký tự 32×32. Ảnh đã được cắt theo nhãn trên PC và nhúng sẵn vào bitstream; FPGA chưa nhận ảnh camera hoặc tự phát hiện/cắt/sắp xếp biển. Tập chỉ 6 biển, chia validation ở mức frame và có thể trùng danh tính với train; các tỷ lệ trên không chứng minh độ chính xác 95% của hệ thống thực địa. Thời gian trên không tính tiền xử lý, truyền ảnh, JTAG hay camera. Quartus cảnh báo thiết kế chưa ràng buộc đầy đủ tất cả đường timing; các số Fmax/slack chỉ áp dụng cho đường xung đã phân tích.
