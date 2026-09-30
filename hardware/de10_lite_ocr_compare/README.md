# Đối chứng dense / structured channel pruning / JTAG frame streaming

Thí nghiệm này tách **hai biến độc lập** trên cùng LeNet-5 OCR ký tự 32×32:

| Cấu hình | Conv2 | Ảnh thử vào FPGA |
|---|---:|---|
| `dense_rom` | 16 kênh | 50 crop nằm sẵn trong ROM của bitstream |
| `dense_stream` | 16 kênh | Laptop gửi từng crop qua USB-Blaster/JTAG vào RAM 1 KiB |
| `pruned_rom` | 8 kênh | ROM như bản dense |
| `pruned_stream` | 8 kênh | JTAG/RAM như bản dense stream |

Conv1 giữ 6 kênh. Structured pruning cắt nguyên **8 bộ lọc đầu ra Conv2**
và những cột đầu vào FC1 tương ứng; không chỉ đặt trọng số bằng 0. Mỗi nhánh
train trên cùng crop từ `train(1)`, cùng 10 lượt xem dữ liệu: 5 epoch dense
chung rồi 5 epoch tiếp tục riêng; mất bao lâu và accuracy từng epoch nằm ở
`artifacts/compare_conv2_stream_20260929/history.csv`. Streaming **không phải**
thủ thuật train: đây là đường nạp ảnh lúc suy luận.

`train_compare.py` tạo hai checkpoint trên CPU laptop; `export_compare.py`
lượng tử hóa trọng số INT8/bias INT32 và chọn đúng cùng 50 ký tự validation
với seed 20260929. `OcrBenchCore.sv` là cùng một RTL điều khiển MAC tuần tự
cho bốn bản; chỉ hai tham số Conv2 và nguồn ảnh khác. Các thư mục biến thể
có project Quartus riêng để lưu `.sof`, fit và timing report riêng. Với bản
stream, Tcl gửi **64 gói × 16 byte** cho mỗi crop qua JTAG, chờ FPGA xác nhận
từng gói rồi mới bắt đầu OCR.

## Xem và chạy lại trên máy hiện tại

1. Mở `BAO_CAO_SO_SANH.md` và `comparison_full.json` để đọc toàn bộ số đã đo.
2. Mở `train_history.csv` trong thư mục này để thấy loss, accuracy, thời gian
   và CPU giây của từng epoch. Checkpoint ở
   `artifacts/compare_conv2_stream_20260929/` chỉ lưu cục bộ.
3. Trong VS Code, chọn **Terminal → Run Task**. Chọn
   `DE10 Compare - Train lại (thư mục kết quả mới)` để train hai nhánh và nhập
   tên **thư mục chưa tồn tại kết quả** (ví dụ `artifacts\\compare_conv2_stream_manual_2`).
   Task này chỉ train, không thay bộ trọng số đã kiểm chứng của bản demo.
   Muốn đưa checkpoint mới xuống kit, chạy `export_compare.py --models` trỏ
   tới thư mục vừa tạo, rồi tổng hợp/nạp/chạy lại cả bốn cấu hình; không được
   lấy báo cáo phần cứng cũ để nhận xét checkpoint mới.
4. Các task `DE10 Compare - Tổng hợp...`, `Nạp...`, `Đo...` chạy riêng từng
   cấu hình. `DE10 Compare - Phần cứng đầy đủ` chạy tuần tự bốn cấu hình
   với **checkpoint đã train trước đó** và tổng hợp báo cáo. Không cần gõ
   lệnh thủ công khi muốn chạy lại phép đo hiện có.
5. Số FPGA gốc nằm tại `<variant>/generated/board_results.csv`; nhãn và
   ảnh cắt tại `<variant>/generated/manifest.json` và `images.hex`; tài nguyên
   và timing do Quartus sinh tại `<variant>/output_files/`.

`generated/` và `output_files/` **chỉ ở laptop, không đẩy lên GitHub** vì chứa
ảnh/nhãn biển hoặc bitstream gắn ảnh thử. Báo cáo công khai chỉ tổng hợp theo
nhóm, không liệt kê biển cụ thể.

## Thí nghiệm 6 biển từ đầu

`BAO_CAO_6_BIEN.md` là phép đối chứng riêng: chọn 2 ảnh ở mỗi nhóm từ
`train(1)` (6 biển, 49 crop), train **từ đầu** dense và tỉa 50% kênh Conv2.
Không dùng checkpoint tập lớn làm điểm xuất phát. Trên VS Code, chọn
**Terminal → Run Task → DE10 Compare - Train 6 biển (dense và pruning)**,
nhập tên thư mục kết quả mới. Script không ghi đè checkpoint cũ.
`micro6_train_history.csv` chứa số mỗi epoch và `micro6_public_metrics.json`
chứa ma trận nhầm lẫn, kết quả từng nhóm và phép đo CPU. Danh sách ảnh cụ thể,
trọng số và ảnh xuất INT8 nằm trong `artifacts/compare_micro6_verified_20260930/`
chỉ ở máy này. Bản 6 biển **chưa nạp lên FPGA**; tài nguyên Quartus trích dẫn
trong báo cáo là phép đo của cùng kiến trúc với trọng số tập lớn. Nếu train một
lượt mới ở thư mục khác, chỉ dùng báo cáo của lượt này sau khi chạy giả lập
INT8 và tổng hợp lại; không lấy số accuracy của lượt cũ gán cho checkpoint mới.

## Giới hạn đo

- Đây là **OCR ký tự đã được cắt sẵn**. FPGA chưa nhận camera, tìm biển,
  sửa phối cảnh, phân loại hàng hoặc cắt ký tự từ ảnh toàn cảnh.
- “Streaming” chỉ là **nạp từng frame ký tự qua JTAG vào RAM trước khi chạy**,
  chưa phải đường ống convolution pixel-stream có line buffer hay camera thật.
  Số chu kỳ lõi tách khỏi thời gian truyền; JTAG/Tcl không đại diện cho
  giao tiếp triển khai cuối cùng.
- Bản ROM chứa cả 50 ảnh test trong bitstream; bản streaming chỉ giữ RAM
  1 KiB. Vì vậy chênh lệch RAM bao gồm thiết kế bộ thử, không thể gán hết cho
  pruning.
- Validation chia theo frame và crop theo số ký tự từ nhãn; có thể trùng
  biển giữa train/validation. Mẫu FPGA chỉ 6 biển/50 ký tự, không chứng minh
  độ chính xác thực địa. Fmax là ước tính post-fit với cảnh báo một số đường
  timing chưa được ràng buộc đầy đủ.
