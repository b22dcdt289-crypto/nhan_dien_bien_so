# Bản hệ thống kit độc lập

```text
OV3660 → ESP32-S3: ROI, tách ký tự 28×28
       → UART → DE10-Lite: LeNet INT8 30 lớp, quyết định
       → UART → ESP32-S3: kết quả và giao diện web
```

- Firmware: `esp32/PlateCamUART/PlateCamUART.ino`.
- Mạch Quartus và ROM của model phần cứng: `fpga/standard/LiveDe10UART.qpf`, các tệp `.sv` và `generated/*.hex`.
- Kết nối, định dạng gói và giới hạn kiểm chứng: [docs/UART_PROTOCOL.md](docs/UART_PROTOCOL.md).
- Bản này dùng thiết kế **30 lớp, một hàng, 8 ký tự**; model 31 lớp 28×28 mới trên PC chưa chuyển sang RTL/bitstream tương ứng. Không nạp các tệp `generated31` vào mạch 30 lớp.
- `fpga/experimental/` là mạch tăng tốc thử nghiệm; tốc độ UART và firmware hiện tại chưa khớp, không dùng nó như bản kit độc lập đã kiểm chứng.

Cần Arduino IDE, ESP32-S3 với camera OV3660, Quartus, DE10-Lite, USB-Blaster và các dây UART/GND. File Quartus là mã nguồn cần được tổng hợp và kiểm tra trên bo; gói không chứa bitstream `.sof` đã xác minh cho mọi máy. Không có số đo camera-to-result mới trong gói này và không cam kết độ chính xác hay tốc độ khi chưa kiểm thử trên thiết bị đích.
