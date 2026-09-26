# Kết quả cắt dữ liệu và huấn luyện LeNet-5 trên 1.000 biển một hàng

## Quy trình

- Nguồn: `data/OCR/OCR/images/train(1)/detection/one_row` gồm 19.086 ảnh; tên file chứa chuỗi biển và hộp tọa độ đã gán nhãn.
- Cắt ROI biển theo hộp gán nhãn, hiệu chỉnh phối cảnh, bilateral denoise, CLAHE và unsharp mask; phân đoạn ký tự thành ảnh xám 32×32 cho LeNet-5.
- Tạo riêng `data/one_row_1000_curated_v4`; dữ liệu nguồn không bị sửa. Train có đúng 1.000 biển riêng biệt/7.901 crop ký tự, validation 300 biển, test ngẫu nhiên 500 biển.
- Tách theo chuỗi biển đã chuẩn hóa, nên cùng một biển không xuất hiện ở nhiều split. Test được chọn trước khi biết phân đoạn có thành công hay không; 500 ROI đều nằm trong mẫu số.
- Huấn luyện dense LeNet-5 từ khởi tạo ngẫu nhiên trong 20 epoch; không pruning. Chọn checkpoint theo accuracy ký tự validation.

## Rà soát nhãn

Đối chiếu ảnh gốc phát hiện 8 nhãn bất thường (tiền tố thừa hoặc ký tự thiếu)
và ghi từng ánh xạ sửa trong `data/one_row_1000_curated_v4/review/label_corrections.csv`.
Năm ảnh đã sửa được đưa vào đúng 1.000 ảnh train; ba ảnh còn lại không qua được
phân đoạn ký tự và không được đưa vào train. Các ảnh gốc được giữ nguyên.

Model OCR trước đây bất đồng với nhãn ở 109/1.687 crop mà nó đọc được. Đây chỉ
là cờ để rà soát, không phải bằng chứng nhãn sai: model đó từng được huấn luyện
trên nguồn rộng hơn. Các bất đồng có ảnh và chuỗi đối chiếu trong thư mục
`data/one_row_1000_curated_v4/review/contact_sheets`; nhãn train lấy từ annotation
đã kiểm chứng, không lấy dự đoán cũ làm ground truth.

## Kết quả test độc lập

| Phép đo | Kết quả |
|---|---:|
| Ảnh test ngẫu nhiên, mã biển không trùng train/validation | 500 |
| Phân đoạn được khi dùng độ dài nhãn chuẩn | 387/500 (77,40%) |
| Accuracy ký tự trên các ROI phân đoạn được theo nhãn chuẩn | 3.004/3.069 = **97,88%** (Wilson 95% CI: 97,31–98,33%) |
| Exact plate, tính cả 113 ROI phân đoạn không thành công là sai | 345/500 = **69,00%** (Wilson 95% CI: 64,81–72,90%) |
| Exact plate chỉ trong 387 ROI phân đoạn được | 345/387 = **89,15%** |
| Phân đoạn không biết trước số ký tự, trên ROI biển đã cắt chuẩn | 444/500 được thử (88,80% coverage); 56 bỏ qua |
| Exact plate không biết trước số ký tự, tính cả bỏ qua | 340/500 = **68,00%**; 340/444 = 76,58% trong số được thử |
| Sai khác số ký tự do phân đoạn runtime | 100/500 |
| Mean confidence lớn nhất softmax/ký tự | 96,84% (không phải accuracy đã hiệu chuẩn) |

Các số OCR ký tự điều kiện dùng hộp biển chuẩn từ datasheet. Con số exact 69%
đưa tất cả 500 ảnh ngẫu nhiên vào mẫu số và phạt phân đoạn lỗi. Phép chạy không
biết trước số ký tự gần với runtime hơn, nhưng vẫn bắt đầu từ ROI biển chuẩn;
**chưa đo detector cắt biển từ toàn khung hình/camera**.

## Accuracy theo vị trí ký tự

| Vị trí | Đúng / tổng ký tự có ở vị trí đó | Accuracy |
|---:|---:|---:|
| 1 | 378/387 | 97,67% |
| 2 | 381/387 | 98,45% |
| 3 | 367/387 | 94,83% |
| 4 | 381/387 | 98,45% |
| 5 | 384/387 | 99,22% |
| 6 | 380/387 | 98,19% |
| 7 | 383/387 | 98,97% |
| 8 | 348/358 | 97,21% |
| 9 | 2/2 | 100% (mẫu quá nhỏ, không suy rộng) |

Vị trí 3 là điểm yếu tương đối. Bảng precision/recall/F1 và support cho từng
ký tự 0–9, A–Z nằm trong `one_row_1000_metrics.json`; ma trận nhầm lẫn ở
`one_row_1000_metrics_confusion_matrix.csv`. Macro-F1 trên các ký tự có support
là 72,52%; thấp hơn accuracy tổng do các chữ cái hiếm có rất ít mẫu. Vì vậy
accuracy tổng không nên được dùng một mình để kết luận mọi ký tự đều tốt như nhau.

| Độ dài nhãn | Test | Phân đoạn theo nhãn | Exact toàn nhóm | Exact trong nhóm phân đoạn được | Character accuracy trong nhóm phân đoạn được |
|---:|---:|---:|---:|---:|---:|
| 7 | 63 | 29 | 22/63 = 34,92% | 22/29 = 75,86% | 96,55% |
| 8 | 433 | 356 | 322/433 = 74,36% | 322/356 = 90,45% | 98,03% |
| 9 | 4 | 2 | 1/4 = 25,00% | 1/2 = 50,00% | 88,89% |

Nhóm 9 ký tự chỉ có 4 biển test nên kết quả chưa có ý nghĩa suy rộng. Nhóm 7
ký tự có tỷ lệ phân đoạn thấp hơn; dữ liệu và thuật toán cần được mở rộng/điều
chỉnh cho các kiểu biển này.

## Cỡ mô hình và diễn giải

- LeNet-5 dense: 63.916 tham số, 418.704 MAC/ký tự.
- Theo số ký tự: 7 → 2.930.928 MAC; 8 → 3.349.632 MAC; 9 → 3.768.336 MAC.
- Không thể kết luận 97,88% là accuracy nhận dạng toàn ảnh: đây là accuracy ký
  tự có điều kiện sau khi đã có ROI và phân đoạn. Exact plate trên mẫu test ngẫu
  nhiên giảm xuống 69% khi tính các lỗi phân đoạn.
- 56 ảnh bị bỏ qua ở runtime chủ yếu vì phân đoạn trả về số ký tự không hỗ trợ
  (36 ảnh) hoặc thất bại (20 ảnh). Cần cải thiện row/count segmentation trước
  khi tuyên bố hiệu quả triển khai thực tế.

## Chạy lại

```powershell
.venv\Scripts\python.exe prepare_one_row_1000.py --output data/one_row_1000_curated_v4
.venv\Scripts\python.exe train_one_row_1000.py --data data/one_row_1000_curated_v4 --epochs 20
.venv\Scripts\python.exe make_ocr_review_sheets.py --data data/one_row_1000_curated_v4
```

Thử một ảnh ROI đã cắt:

```powershell
.venv\Scripts\python.exe recognize_independent.py data\one_row_1000_curated_v4\plates\test\test_0003.png --plate-crop --model artifacts/lenet5_dense_one_row_1000.pt
```

Các dự đoán/nhãn từng biển test nằm trong `artifacts/one_row_1000_test_predictions.csv`;
training history ở `artifacts/one_row_1000_training_history.csv`.
