# Demo LeNet-5 trực tiếp trên DE10-Lite

Đây là **lõi OCR ký tự**, không phải hệ thống camera đọc biển hoàn chỉnh.
Ảnh từ tập validation `train(1)` được cắt thành từng ký tự 32×32 trên máy,
chọn ngẫu nhiên với seed cố định, lượng tử hóa và nhúng trong `.sof` cùng trọng số.
FPGA thực hiện Conv1, Tanh, AvgPool, Conv2, Tanh, AvgPool, ba lớp FC và chọn lớp;
máy chỉ ra lệnh chọn mẫu và đọc kết quả qua JTAG.

Mô hình là checkpoint `artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt`:
Conv1 giữ 6 kênh, Conv2 tỉa còn 8 kênh, 30 lớp đầu ra. Phần lượng tử hóa
sử dụng trọng số INT8 với số mũ nhị phân theo lớp, bias INT32, LUT Tanh và
trung bình 2×2. Trước khi dùng kết quả phần cứng, đối chiếu với mô phỏng
số nguyên trong `prepare_demo.py`.

## Chạy trong VS Code, không cần gõ lệnh

1. Cắm nguồn/USB-Blaster DE10-Lite, mở thư mục gốc project trong VS Code.
2. Chọn **Terminal → Run Task → DE10-Lite - Demo đầy đủ**. Có thể chạy từng
   bước 1/5 đến 5/5 để trình bày quá trình tổng hợp và nạp cho giảng viên.
3. Mở `BANG_KET_QUA_TRUC_TIEP.md` để xem tài nguyên, timing, cycles và accuracy.
   Quartus Programmer phải báo `Configuration succeeded` và bước 4/5 phải
   in đủ `SAMPLE=0` đến `SAMPLE=49` trước khi đọc báo cáo.

Đường dẫn Quartus trong `.vscode/tasks.json` là `E:\k\quartus\bin64` trên
máy hiện tại. Nếu cài ở nơi khác, sửa các tác vụ 2/5–4/5.

## Tệp và quyền riêng tư

- `prepare_demo.py`: cố định seed 20260929 và lấy 2 biển mỗi nhóm (một hàng,
  ô tô hai hàng, xe máy hai hàng) từ validation; xuất ROM và dự đoán đối chiếu.
- `OcrDemo.sv`, `.qpf`, `.qsf`, `.sdc`: lõi RTL và cấu hình Quartus.
- `run_demo.tcl`: dùng JTAG In-System Sources and Probes điều khiển từng ký tự,
  ghi class và cycles đọc trực tiếp từ FPGA.
- `analyze_demo.py`: đối chiếu dự đoán FPGA với nhãn/mô phỏng, đọc báo cáo fit
  và timing, xuất báo cáo tổng hợp. Bảng chi tiết từng biển và từng ký tự nằm
  ở `generated/plate_results_private.csv` và
  `generated/character_results_private.csv` để xem tại máy khi demo.
- `generated/`: ảnh, trọng số, tên/nhãn biển và CSV dự đoán từng ký tự; **chỉ
  lưu trên máy, không đưa lên GitHub**. `output_files/` cũng là file build cục bộ.

Vì dữ liệu gốc, ROM và bitstream không được đưa lên GitHub, người khác cần
checkpoint và thư mục `train(1)` tại cùng project để chạy bước 1/5 và biên dịch.

## Giới hạn quan trọng

Số accuracy trong demo chỉ trên crop ký tự tham chiếu từ 6 biển, không đại
diện cho biển ngẫu nhiên ngoài đời. Validation gốc chia ở mức frame, có nguy
cơ trùng biển giữa train/validation. FPGA chưa tự nhận ảnh camera, định vị
biển, sửa phối cảnh, phân hàng hoặc cắt ký tự. Mỗi lần chạy Task, host giữ
JTAG high/low để đồng bộ; thời gian Task toàn bộ **không phải** latency tính
trên FPGA. Latency lõi lấy từ bộ đếm chu kỳ trong RTL và giả định xung 50 MHz.

Để chứng minh pruning **tăng tốc so với dense trên cùng FPGA**, cần tổng hợp
và chạy thêm bản dense với cùng kiến trúc bộ nhân/bộ nhớ/giao tiếp; không suy
ra speedup FPGA từ MAC hoặc thời gian CPU. Để đạt nhận diện thực tế, cần có
giao tiếp ảnh vào, tiền xử lý/cắt ký tự trên phần cứng và bộ test tách biển
theo danh tính.
