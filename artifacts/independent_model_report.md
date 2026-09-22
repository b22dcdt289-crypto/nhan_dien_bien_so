# Báo cáo pipeline độc lập không YOLOv5

## Kiến trúc

```text
Camera/frame
  -> contour + morphology
  -> perspective correction
  -> row classification
  -> sort từng dòng
  -> LeNet-5 OCR INT8/structured
```

Phương án này không dùng YOLOv5. Các bước trước LeNet-5 là xử lý ảnh hình học,
phù hợp để chuyển thành HDL hoặc mạch xử lý ảnh trên FPGA. Laptop chỉ được dùng
để chạy prototype và kiểm chứng thuật toán.

## Dữ liệu và độ chính xác

| Chỉ số | Kết quả |
|---|---:|
| Ảnh nguồn `train(1)` | 37.297 |
| Ảnh chấp nhận sau segmentation | 24.888 |
| Crop ký tự | 199.412 |
| Tỷ lệ bỏ qua tổng | 33,27% |
| Dense validation character accuracy | 90,43% |
| Structured validation character accuracy | 90,58% |
| Plate-level exact match | 78,02% |
| Một dòng character accuracy | 97,20% |
| Ô tô hai dòng character accuracy | 92,77% |
| Xe máy hai dòng character accuracy | 60,38% |

Kết quả structured hiện chưa đạt mục tiêu 95,22% trên toàn bộ pipeline. Nguyên
nhân chính là chất lượng crop và thứ tự ký tự của nhóm xe máy hai dòng, không
phải do MAC của LeNet-5.

Trong bước tạo crop huấn luyện, độ dài chuỗi trong tên file được dùng để loại
ảnh có số component không khớp nhãn, nhằm tránh tạo nhãn ký tự sai. Vì vậy,
đây là đánh giá OCR trên các ảnh đã qua bước tách ký tự thành công; khi chạy
camera không có nhãn độ dài, chương trình chỉ chấp nhận 8, 9 hoặc 10 ký tự và
bỏ qua frame có số component bất thường.

## MAC và tài nguyên mô hình

| Mô hình | Channel/neuron | MAC/ký tự | Giảm MAC |
|---|---|---:|---:|
| LeNet-5 dense | 6, 16, 120, 84 | 418.704 | 0% |
| LeNet-5 structured | 4, 10, 100, 70 | 212.920 | 49,15% |

| Độ dài biển | MAC dense | MAC structured |
|---:|---:|---:|
| 8 ký tự | 3.349.632 | 1.703.360 |
| 9 ký tự | 3.768.336 | 1.916.280 |
| 10 ký tự | 4.187.040 | 2.129.200 |

Checkpoint structured có 35.840 tham số, kích thước INT8 lý thuyết khoảng
0,034 MiB. Số ALM, DSP, memory bits, fMAX và latency phải được xác nhận lại
bằng Quartus sau khi ánh xạ kiến trúc sang HDL.

## Tỷ lệ bỏ qua theo điều kiện

| Điều kiện heuristic | Ngưỡng | Ảnh thuộc nhóm | Bị bỏ qua | Tỷ lệ bỏ qua |
|---|---:|---:|---:|---:|
| Mờ | Laplacian variance < 80 | 88 | 39 | 44,32% |
| Nghiêng | Góc ước lượng > 8° | 95 | 41 | 43,16% |

Hai tỷ lệ trên là thống kê theo heuristic của preprocessing. Cần kiểm thử
thêm bằng bộ dữ liệu có nhãn điều kiện ánh sáng/góc nghiêng để gọi đây là tỷ
lệ lỗi camera chính thức.
