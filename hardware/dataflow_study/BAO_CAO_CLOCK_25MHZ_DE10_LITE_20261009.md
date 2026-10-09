# Kiểm chứng lõi OCR ở 25 MHz hiệu dụng trên DE10-Lite

Ngày đo: 09/10/2026. Thiết bị: DE10-Lite, MAX 10 `10M50DAF484C7G`; Quartus Prime Lite 25.1std.0; nạp và đọc kết quả qua USB-Blaster/JTAG.

## Phạm vi thay đổi và cách xác định tần số

Chân `MAX10_CLK1_50` tại P11 **vẫn nhận clock vật lý 50 MHz** (chu kỳ 20 ns). Trong `OcrBenchCoreK3.sv`, tham số `CORE_ENABLE_DIV2=1` cho FSM, các thanh ghi dữ liệu và RAM chỉ cập nhật ở mỗi cạnh lên thứ hai. Vì vậy tốc độ bước của lõi là **25 triệu bước/giây**, không phải một mạng clock 25 MHz riêng. Bộ đếm `cycles` ghi tất cả cạnh clock đầu vào khi lõi bận; thời gian tính thuần FPGA bằng `cycles / 50.000.000` giây. Ràng buộc SDC vẫn là 20 ns để kiểm tra các đường thanh ghi đúng theo clock vật lý 50 MHz. Không dùng PLL và không sửa dao động trên bo.

Hai checkpoint, thứ tự lớp, ảnh kiểm thử, trọng số INT8, bias, LUT và shift giống hệt phép đo 50 MHz ngày 09/10/2026 (đã so sánh SHA-256 từng tệp đầu vào). **Không huấn luyện lại mô hình** trong lần đổi clock này. Đầu vào JTAG là từng ảnh ký tự 32×32 đã phân đoạn từ 12 biển, tổng 100 ký tự; không phải phép đo hệ thống từ ảnh biển nguyên vẹn.

## Kết quả đo trực tiếp và hậu bố trí mạch

| Cấu hình | Bước lõi hiệu dụng | Chu kỳ 50 MHz/ký tự, đọc JTAG | Thời gian tính/ký tự | Đúng ký tự | Đúng toàn biển | LE | M9K | Bit RAM | DSP 9-bit | Fmax góc chậm 85°C | Setup slack ở 50 MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense cũ | 50 MHz | 595.020 | 11,9004 ms | 99/100 | 11/12 | 2.391 | 92 | 723.296 | 1 | 62,68 MHz | +4,047 ns |
| Dense mới | 25 MHz | 1.190.040 | 23,8008 ms | 99/100 | 11/12 | 2.403 | 92 | 723.296 | 1 | 63,60 MHz | +4,276 ns |
| Tỉa Conv2 cũ | 50 MHz | 373.236 | 7,46472 ms | 99/100 | 11/12 | 2.345 | 59 | 443.360 | 1 | 65,92 MHz | +4,830 ns |
| Tỉa Conv2 mới | 25 MHz | 746.472 | 14,92944 ms | 99/100 | 11/12 | 2.368 | 59 | 443.360 | 1 | 66,13 MHz | +4,878 ns |

Ví dụ, bản tỉa mới: `746.472 / 50.000.000 = 0,01492944 s/ký tự`. Số chu kỳ đầu vào đúng gấp đôi bản cũ; kết quả dự đoán của 100/100 ký tự giống hệt bản cũ. Bản tỉa ở 25 MHz tính một ký tự nhanh hơn bản dense cùng 25 MHz khoảng `23,8008 / 14,92944 = 1,594` lần. **Đổi 50 xuống 25 MHz không tăng tốc**: độ trễ tính của mỗi mô hình tăng đúng 2 lần. M9K và bit RAM không đổi; LE tăng 12 ở dense và 23 ở bản tỉa do logic cho phép clock và ánh xạ sau tổng hợp. Chênh lệch Fmax nhỏ là kết quả bố trí–định tuyến; Fmax không phải tần số lõi đang chạy.

Thời gian JTAG tổng hợp, gồm nạp 1 KiB ảnh qua 64 gói, giao tiếp PC, chờ và tính, có trung vị 1.128 ms (dense) và 1.134,5 ms (tỉa) trong lần 25 MHz. Phần lớn thời gian này do đường JTAG/PC, **không** đại diện cho độ trễ của đường truyền ảnh tối ưu hay khả năng xử lý camera thời gian thực. Không có phép đo công suất hoặc nhiệt độ trên kit, nên không kết luận tiết kiệm điện.

Quartus báo setup slack dương cho clock 50 MHz tại góc chậm 85°C và cả hai bitstream nạp thành công. Tuy nhiên Timing Analyzer cũng báo thiết kế **chưa được ràng buộc đầy đủ các đường I/O**; cần hoàn thiện SDC nếu dùng cho nghiệm thu timing toàn hệ thống. `.sof` được nạp vào cấu hình tạm thời của FPGA, mất sau khi mất nguồn; chưa ghi cấu hình không bay hơi. Hiện kit đang chạy bản **tỉa Conv2, 25 MHz hiệu dụng**.

## Vị trí kiểm chứng và tái lập

- RTL: `hardware/de10_lite_ocr_compare/OcrBenchCoreK3.sv`, `BenchVariantsK3_25.sv`; SDC clock vật lý: `hardware/de10_lite_ocr_compare/OcrBench.sdc`.
- Hai dự án Quartus: `hardware/dataflow_study/rtl_eval/three_layout_k3_25mhz_20261009/dense_stream/` và `pruned_stream/`; tệp `.sof` trong `output_files/` của từng dự án trên máy đo. Bản sao bitstream tỉa kênh đã đo để nạp trực tiếp từ GitHub: `hardware/dataflow_study/results/three_layout_k3_25mhz_20261009/OcrBench_pruned_stream_25MHz.sof`, SHA-256 `339c82e2a0f76d814312419c5bc79cbbdc773d802c7969af9e5a7fb2be42248a`.
- Tệp kết quả từng ký tự: `generated/board_results.csv` trong từng dự án; bản sao lưu để đưa lên GitHub ở `hardware/dataflow_study/results/three_layout_k3_25mhz_20261009/{dense_board_results.csv,pruned_board_results.csv}`. Cột `cycles` là **cạnh clock vật lý 50 MHz**, không phải số bước lõi.
- Bản tổng hợp có SHA-256 checkpoint/bitstream và kết quả theo loại biển: `hardware/dataflow_study/results/three_layout_k3_25mhz_20261009/board_results.json`.
- Bản sao báo cáo gốc Quartus hậu fit và timing của hai cấu hình: `dense_stream.fit.summary`, `dense_stream.sta.rpt`, `pruned_stream.fit.summary`, `pruned_stream.sta.rpt` cùng thư mục kết quả.
- Lệnh phân tích: `python hardware/dataflow_study/analyze_three_layout_board.py --run hardware/dataflow_study/rtl_eval/three_layout_k3_25mhz_20261009 --output <tep-json-moi> --core-enable-divisor 2`.
- Các tệp `weights.hex`, `biases.hex`, `shifts.hex`, `tanh.hex` được lưu trong thư mục `generated/` của từng dự án để biên dịch lại. `images.hex` chứa ảnh kiểm thử cục bộ, không đưa lên GitHub; số đo và dự đoán từng ký tự đã lưu trong hai CSV ở trên.

Nếu yêu cầu **một mạng clock nội bộ đúng 25 MHz** để đo bằng Signal Tap hoặc cấp cho khối phần cứng khác, cần thêm PLL/chia clock bằng tài nguyên chuyên dụng và ràng buộc generated clock; thiết kế này hiện dùng clock-enable đồng bộ. Hướng dẫn của Intel cũng khuyến nghị clock-enable đồng bộ thay cho chia/gating clock bằng logic thường: https://www.intel.com/programmable/technical-pdfs/683323.pdf .
