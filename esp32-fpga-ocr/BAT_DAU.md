# Hướng dẫn chạy dự án

## 1. Chạy demo nhận diện trên máy tính

Cài Python 3.10–3.12, sau đó mở PowerShell tại thư mục gốc của repository hoặc thư mục đã giải nén:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r esp32-fpga-ocr/requirements-pc.txt
python esp32-fpga-ocr/pc/recognize.py esp32-fpga-ocr/pc/examples/synthetic_demo.png
```

Nếu PowerShell chặn kích hoạt môi trường, có thể dùng trực tiếp `.\.venv\Scripts\python.exe` thay cho `python` mà không cần kích hoạt. Máy phải có quyền tải các thư viện Python; PyTorch có thể cần dung lượng lớn.

Chương trình trả JSON với chuỗi ứng viên, trạng thái nhận/từ chối, ký tự và ba điểm cao nhất. Ảnh mẫu tổng hợp có chuỗi `12A34567` (bỏ dấu gạch và chấm). Để thử ảnh khác, thay đường dẫn cuối bằng đường dẫn ảnh JPG/PNG đã cắt trọn vùng biển; ví dụ:

```powershell
python esp32-fpga-ocr/pc/recognize.py "C:\duong-dan\anh-bien.jpg"
```

Demo PC dùng model FP32 31 lớp, thuật toán tách ký tự và chính sách nhận/từ chối đã hiệu chuẩn trên tập tổng hợp. Chương trình nhận ảnh vùng biển đã cắt; không tự mở camera hoặc phát hiện biển trong toàn khung hình. Điểm model không phải tỷ lệ chính xác được bảo đảm. Xem [README](README.md) để biết phạm vi kết quả.

## 2. Chạy bản pruning trên PC

```powershell
python esp32-fpga-ocr/model/recognize_pruned_policy.py esp32-fpga-ocr/pc/examples/synthetic_demo.png
```

Model pruning 30 lớp dùng ký tự 32 × 32, khác model chính 31 lớp dùng 28 × 28. Chính sách quyết định của bản pruning còn thử nghiệm, chưa được hiệu chuẩn riêng. Xem [hướng dẫn pruning](model/PRUNING.md).

Ở lần kiểm tra trên ảnh mẫu này, bản pruning chạy thành công nhưng trả `22A34567` thay vì `12A34567`, đồng thời vẫn nhận kết quả. Điều này cho thấy ngưỡng của bản pruning chưa đủ tin cậy.

## 3. Mở firmware và FPGA

- ESP32-S3/OV3660: mở `esp32/PlateCamUART/PlateCamUART.ino` bằng Arduino IDE với gói ESP32 phù hợp, cấu hình đúng bo và PSRAM, sau đó biên dịch. Cần có phần cứng và cáp USB dữ liệu; cổng camera khác có thể cần điều chỉnh chân.
- FPGA: mở `fpga/standard/LiveDe10UART.qpf` bằng Quartus Prime cho DE10-Lite MAX 10 và tổng hợp; các tệp ROM `generated/` phải giữ nguyên cùng thư mục. Bản `fpga/experimental/` là phương án UART/tăng tốc riêng, không tương thích mặc định với firmware chuẩn.
- Nối hai kit và kiểm tra điện áp, chân, định dạng gói ở [UART_PROTOCOL.md](docs/UART_PROTOCOL.md). Bản phần cứng 30 lớp và phần PC 31 lớp là hai phiên bản khác nhau; không đưa trọng số 31 lớp vào bitstream 30 lớp.

Các bước phần cứng cần công cụ, bo mạch và kiểm thử thực tế; bản đóng gói không bảo đảm mọi phiên bản Arduino/Quartus sẽ biên dịch không thay đổi. Không bao gồm bộ dữ liệu train đầy đủ; các script nghiên cứu còn phụ thuộc dữ liệu và đường dẫn của môi trường phát triển.
