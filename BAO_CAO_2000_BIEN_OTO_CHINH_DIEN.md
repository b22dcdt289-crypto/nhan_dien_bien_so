# Bộ ảnh biển ô tô chính diện: 1.000 hai hàng + 1.000 một hàng

## Kết quả đã tạo

Thư mục trên máy: `data/car_frontal_synthetic_2x1000_v1/`. Bản kê từng ảnh: `manifest.csv`; hai bảng xem mẫu: `review_two_row_car.jpg`, `review_one_row_car.jpg`.

| Nhóm | Số PNG | Kích thước | Chuỗi biển khác nhau | Đầu biển khác nhau |
|---|---:|---:|---:|---:|
| Ô tô hai hàng | 1.000 | 136 × 144, thang xám | 1.000 | 72 |
| Ô tô một hàng | 1.000 | 312 × 74, thang xám | 1.000 | 114 |

Hai nhóm không dùng trùng chuỗi biển. Tổng 2.000 nhãn và 2.000 ảnh có hash khác nhau; không có ảnh hoặc tệp nguồn bị thiếu. Dung lượng PNG là 26.581.489 byte (khoảng 25,35 MiB). Mỗi PNG đã được mở lại và đối chiếu pixel ngay sau khi ghi.

## Ảnh được tạo bằng cách nào

Đây là **ảnh dựng tổng hợp**, không phải ảnh camera được sửa phối cảnh. Cách này bảo đảm hình chữ nhật nhìn thẳng như ảnh mẫu, khung viền, hai dòng chữ rõ và chuỗi ký tự in ra khớp nhãn do chương trình đưa vào. Script `render_frontal_car_plates_2000.py` lấy 2.000 nhãn khác nhau từ dữ liệu `train(1)`, chỉ chọn chuỗi tám ký tự dạng hai số tỉnh + một chữ + năm chữ số; cân bằng chọn theo ba ký tự đầu. Mỗi dòng trong `manifest.csv` lưu nhãn gốc, cách trình bày và đường dẫn ảnh thật cung cấp nhãn.

Biển hai hàng hiển thị `30A` ở dòng trên và `345.60` ở dòng dưới; biển một hàng hiển thị `30A-345.60`. Ký tự `-` và `.` chỉ là dấu trình bày, **không** được tính là ký tự OCR của nhãn tám ký tự. Hình được dựng bằng phông Arial Narrow, nền sáng, chữ tối, viền và nhiễu nhẹ; không xoay hoặc nghiêng. 1.000 ảnh hai hàng dùng 450 nhãn từ bộ ảnh hai hàng đã tuyển và 550 nhãn từ bản chuẩn bị trước; 1.000 ảnh một hàng dùng 265 nhãn từ bộ tuyển một hàng và 735 nhãn bổ sung từ tên ảnh gốc.

Chạy lại trong thư mục dự án trên Windows: `.\\.venv\\Scripts\\python.exe render_frontal_car_plates_2000.py --output data/car_frontal_synthetic_run_moi`. Script không ghi đè thư mục kết quả đã có dữ liệu và cần các phông Arial Narrow trong `C:\\Windows\\Fonts`.

## Giới hạn khi sử dụng để train

- Nhãn được lấy từ tên tệp nguồn và **chưa được đối chiếu bằng mắt cho cả 2.000 ảnh nguồn**. Ảnh dựng bảo đảm vẽ đúng chuỗi đầu vào, nhưng không chứng minh nhãn gốc của ảnh chụp là đúng.
- Dù ảnh dựng giống bố cục mẫu, nó không mô phỏng đầy đủ phản quang, méo ống kính, vật che, bẩn, rung, mờ chuyển động, ánh sáng ban đêm hoặc phông chữ biển thật. Vì vậy, không dùng accuracy đo chỉ trên bộ tổng hợp để tuyên bố độ chính xác trên camera thực.
- Nên dùng bộ này để khởi tạo/augment cho LeNet-5, sau đó fine-tune và đánh giá trên ảnh biển thật tách riêng theo **chuỗi biển**. Không đưa ảnh dựng cùng nhãn vào validation/test thực tế.
- Bộ cắt thật hai hàng từ lần trước vẫn nằm ở `data/two_row_front_curated_2000_v2/`; không được gộp nhầm bộ đó với bộ ảnh dựng khi báo cáo số mẫu độc lập.

Ảnh PNG và `manifest.csv` được giữ cục bộ trong `data/` (đang bị `.gitignore` loại trừ). GitHub chỉ chứa script và báo cáo tổng hợp; không công khai hàng loạt mã biển lấy từ ảnh thật.
