# Tạo tập 2.000 ảnh biển số hai hàng từ dữ liệu cũ

## Kết quả thực tế

Tập cuối cùng nằm ở `data/two_row_front_curated_2000_v2/` trên máy thực hiện. Ảnh gốc và các tập validation/test không bị thay đổi. Đây là dữ liệu **cắt và hiệu chỉnh hình học**, không phải ảnh mới do AI sinh ra; không có bước huấn luyện hay đo accuracy trong công việc này.

| Nhóm | Ảnh PNG | Biển số khác nhau theo nhãn gốc | Kích thước trung vị (pixel) | Có sửa hình học | Cần rà soát thủ công theo chỉ báo |
|---|---:|---:|---:|---:|---:|
| Ô tô, hai hàng | 1.000 | 1.000 | 61 × 62 | 993 | 35 |
| Xe máy, hai hàng | 1.000 | 922 | 121 × 104 | 984 | 173 |

Tổng 2.000 ảnh chiếm 14.738.042 byte (khoảng 14,06 MiB). Đã đọc lại toàn bộ PNG và so sánh pixel với ảnh đầu vào của bước tuyển chọn: 2.000/2.000 khớp; không có đường dẫn thiếu, ảnh trùng hash hoặc tệp nguồn lặp.

## Nguồn và quy trình

- Nguồn ban đầu: `data/OCR/OCR/images/train(1)/detection/{two_rows,two_rows_label_xe_may}`.
- Bước chuẩn bị có sẵn: `prepare_two_row_frontal_dense.py` đọc bounding box từ tên tệp, cắt biển và dùng homography bốn góc khi thấy viền thích hợp; nếu không, dùng hình chữ nhật xoay hoặc giữ ROI. Nhãn ký tự lấy từ tên tệp nguồn.
- Script `curate_two_row_front_2000.py` chỉ lấy phần `train` của bản chuẩn bị; yêu cầu bộ phân đoạn tham chiếu tìm thấy hai hàng và số vùng ký tự khớp độ dài nhãn. Lọc ảnh quá nhỏ, mờ theo Laplacian, rồi xếp hạng bằng độ nét, tương phản, kích thước và góc của các đoạn thẳng gần ngang. Ưu tiên biển số khác nhau trước khi bổ sung khung hình thứ hai của cùng biển.
- Không vẽ lại ký tự, không thay nhãn, không nội suy để tạo chi tiết mới. Ảnh xuất là bản PNG thang xám giữ nguyên pixel từ bản đã hiệu chỉnh ở bước chuẩn bị.
- Bản kê `manifest.csv` trong thư mục kết quả có `original_source`, `prepared_source`, nhãn từ tên tệp, phương pháp sửa hình học, kích thước, điểm chất lượng, cờ `manual_review_recommended` và SHA-256 cho từng ảnh.
- Hai tấm `review_car_2row.jpg` và `review_motorcycle_2row.jpg` cho xem ngẫu nhiên 36 ảnh mỗi nhóm. Cần mở chúng trong VS Code hoặc trình xem ảnh trước khi dùng để huấn luyện.

Chạy lại từ thư mục dự án: `.\\.venv\\Scripts\\python.exe curate_two_row_front_2000.py --output data/two_row_front_curated_2000_run_moi`. Script không ghi đè thư mục đích đã có dữ liệu.

## Giới hạn quan trọng

"Góc trực diện" ở đây chỉ là **xấp xỉ sau hiệu chỉnh phối cảnh**. Không thể biến biển bị che, quá mờ, mất cạnh hoặc chụp nghiêng mạnh thành ảnh trực diện chân thực mà vẫn giữ thông tin ký tự. Bộ chỉ báo góc >15° xuất hiện ở 30 ảnh ô tô và 159 ảnh xe máy; cờ rà soát còn gồm ảnh không sửa được hình học. Kiểm tra tấm mẫu vẫn thấy một số ảnh bị nghiêng, nhòe hoặc sát mép, nên 2.000 ảnh **chưa được xác nhận thủ công là đều trực diện và có nhãn đúng**. Cờ tự động cũng không phát hiện hết lỗi.

Nhóm xe máy chỉ có 922 chuỗi biển khác nhau trong 1.000 ảnh: 78 ảnh là khung hình khác của chuỗi đã xuất hiện. Nếu yêu cầu 1.000 **biển xe máy khác nhau, đều rõ và thực sự chụp trực diện**, cần bổ sung ảnh gốc và kiểm tra nhãn bằng mắt. Việc coi 1.000 ảnh này là 1.000 biển độc lập sẽ thổi phồng kích thước tập dữ liệu. Nhãn gốc trong tên tệp chưa được xác minh thủ công cho toàn bộ mẫu. Khi chia tập để train/evaluate, phải chia theo chuỗi biển, không chia ngẫu nhiên theo ảnh, để tránh rò rỉ cùng một biển giữa train và validation.

Dữ liệu ảnh và nhãn chi tiết được giữ cục bộ trong `data/` (được `.gitignore` bỏ qua). GitHub chỉ lưu script và báo cáo tổng hợp này; không đưa ảnh biển số thực hoặc bảng nhãn chi tiết lên kho công khai.
