# Bản kiểm tra tốc độ trên máy tính

Cài Python 3.10–3.12 và tại thư mục gốc của gói chạy:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r esp32-fpga-ocr/requirements-pc.txt
.\.venv\Scripts\python.exe esp32-fpga-ocr/pc/benchmark.py esp32-fpga-ocr/pc/examples/synthetic_demo.png --warmup 5 --runs 30 --threads 1 --output ket_qua_toc_do.json
```

Thay đường dẫn ảnh mẫu bằng ảnh vùng biển đã cắt của bạn. Chương trình dùng **model chính 31 lớp FP32** và lặp một ảnh ROI có sẵn trong RAM; khởi động, đọc ảnh, camera, truyền UART, FPGA và giao diện không nằm trong thời gian. Kết quả gồm median, p95 và trung bình cho tách ảnh, forward mỗi ký tự, toàn bộ ký tự, quyết định và từ ROI đến kết quả. Suy luận từng ký tự được chạy tuần tự trên CPU; tốc độ suy ra từ median là chỉ số PC, không phải FPS đo trên FPGA hay camera. Một ảnh lặp lại không đánh giá độ chính xác.

Model này khác model 30 lớp INT8 đang sử dụng cho mạch độc lập. Vì vậy phép đo này không phải phép so sánh chuẩn cùng trọng số hoặc cùng biểu diễn số. Muốn đo FPGA, lấy số chu kỳ trong gói UART và dùng bộ đếm thời gian toàn tuyến; xem [UART_PROTOCOL.md](docs/UART_PROTOCOL.md).
