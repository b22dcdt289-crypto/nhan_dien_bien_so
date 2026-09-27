# Trình bày thí nghiệm LeNet-5 trong VS Code

## Chạy bằng giao diện, không cần nhập lệnh

1. Mở thư mục `D:\TOTNGHIEP\license_plate_ai` trong VS Code.
2. Bấm biểu tượng **Run and Debug** (hình tam giác/ký hiệu con bọ) ở thanh trái.
3. Trong ô chọn phía trên, chọn **Train LeNet-5 1 hang + 2 hang (dense)**.
4. Nhấn nút tam giác màu xanh hoặc **F5**. Xem log từng epoch ở tab **Debug Console** phía dưới. Cấu hình này dùng Python trong `.venv`, không dùng pruning và không cần nhập lệnh Terminal.
5. Chờ dòng `HOÀN TẤT`. Không nên đóng VS Code khi đang train.

Nếu chỉ muốn tạo lại biểu đồ/báo cáo từ số liệu đã train, chọn cấu hình **Cap nhat bao cao LeNet-5 (khong train)** và nhấn F5; thao tác này không chạy lại epoch.

## Mở kết quả để trình bày

Trong VS Code, nhấn `Ctrl+P`, gõ đường dẫn dưới đây rồi Enter:

| Nội dung | Tệp |
|---|---|
| Báo cáo tóm tắt tiếng Việt | `artifacts/vscode_demo/BANG_KET_QUA.md` |
| Biểu đồ validation theo epoch | `artifacts/vscode_demo/training_accuracy.png` |
| Log quá trình train | `artifacts/vscode_demo/live_train.log` |
| Lịch sử pha một hàng | `artifacts/vscode_demo/one_row_training_history.csv` |
| Lịch sử pha hai hàng | `artifacts/vscode_demo/two_row_dense_training_history.csv` |
| Số liệu gốc một hàng | `artifacts/vscode_demo/one_row_metrics.json` |
| Số liệu gốc hai hàng | `artifacts/vscode_demo/two_row_dense_metrics.json` |
| Ma trận nhầm lẫn hai hàng | `artifacts/vscode_demo/two_row_test_reference_after_confusion.csv` |
| So sánh từng lớp ký tự | `artifacts/vscode_demo/two_row_test_per_class_comparison.csv` |

Khi mở `BANG_KET_QUA.md`, có thể nhấn `Ctrl+Shift+V` để xem giao diện báo cáo. Khi mở biểu đồ PNG, VS Code hiển thị trực tiếp ảnh. Để xem ảnh crop mẫu trong thư mục dữ liệu bị ẩn khỏi tìm kiếm nhanh, dùng **File → Open File** (`Ctrl+O`) rồi chọn `D:\TOTNGHIEP\license_plate_ai\data\train1_frontal_1000_unseen_v3\review\frontal_train_examples.jpg` hoặc `D:\TOTNGHIEP\license_plate_ai\data\two_row_frontal_dense_v1\review\two_row_frontal_examples.jpg`.

## Cách đọc đúng số liệu

- **Accuracy ký tự trên crop tham chiếu**: mô hình nhận dạng khi đã có ảnh cắt đúng vị trí ký tự; không chứng minh khả năng tìm/tách biển tự động.
- **Exact toàn biển trên crop tham chiếu**: mọi ký tự trong cùng biển phải đúng. Bước này được dùng làm cổng validation tối thiểu 95% trước khi chuyển sang train hai hàng.
- **Exact từ ROI tự phân đoạn**: hệ thống tự tách các ký tự từ vùng biển; trường hợp bỏ qua hoặc sai số lượng ký tự vẫn tính là sai. Đây là số gần mục tiêu ứng dụng hơn, nhưng vẫn chưa tính phát hiện biển từ khung hình toàn cảnh.
- Các giá trị CPU và MAC là phép đo/ước lượng trong phần mềm. Không diễn giải chúng thành độ trễ hoặc tài nguyên DE10-Lite đã xác nhận, vì chưa có tổng hợp Quartus và chạy trên kit.
- Tập test hai hàng được tách theo chuỗi biển trong lần fine-tune hiện tại, nhưng checkpoint khởi tạo có thể từng thấy ảnh cùng nguồn ở lần train trước. Do đó đây chưa phải kiểm định trên nguồn độc lập tuyệt đối.

Không đưa ảnh biển gốc hoặc ID từng biển lên GitHub; thư mục dữ liệu thô được giữ trên máy.
