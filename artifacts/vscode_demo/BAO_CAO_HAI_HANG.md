# Báo cáo huấn luyện LeNet-5 dense: biển một hàng → biển hai hàng

## Tóm tắt theo điều kiện đặt ra

Pha 1 dùng 1.000 biển ô tô một hàng, đạt 99.917% độ chính xác ký tự và 99.333% exact trên crop validation; đạt ngưỡng tối thiểu 95% nên mới chuyển sang pha hai hàng. Mức trên 99% vẫn được tính là đạt. Không dùng pruning.
Pha 2 fine-tune 12 epoch; epoch tốt nhất 11 có validation hai hàng 96.039% ký tự và 86.631% exact biển.

**Giới hạn pha 1:** số trên là validation có crop tham chiếu, không phải phép đo độc lập cuối cùng. Test 500 biển trước đó có 0 crop ký tự tham chiếu và segmenter không trả chuỗi 8 ký tự cho biển nào; do đó độ chính xác OCR/end-to-end trên test một hàng vẫn chưa được chứng minh.

## Dữ liệu hai hàng và xử lý ảnh

- Nguồn: `data/OCR/OCR/images/train(1)/detection/{two_rows,two_rows_label_xe_may}`; ảnh biển hai hàng ô tô và xe máy được gom theo chuỗi biển trước khi chia tập. Số frame đã lưu: 17,760; số danh tính biển duy nhất theo train/validation/test: {'train': 11125, 'val': 1721, 'test': 1720}.
- Chia nhóm xác định: train 76.7%, validation 11.6%, test 11.7%; giao nhau danh tính = 0 ở cả ba cặp. Đã loại khỏi train 312 frame hai hàng và 153 frame replay một hàng do trùng text biển với miền validation/test còn lại.
- Cắt ROI theo bbox nguồn → tìm tứ giác và warp phối cảnh khi đủ tin cậy → CLAHE + lọc nhiễu/làm nét nhẹ → phân đoạn và phân loại hàng → sắp từ trái sang phải ở hàng trên, rồi hàng dưới → LeNet-5 nhận dạng từng ký tự.
- Chỉnh hình học: homography 4 góc 17,375 (97.8%); min-area rotated rectangle 185; fallback bbox 200. Fallback được giữ và tính riêng, không tuyên bố đã chỉnh thẳng.
- Phân đoạn có số ký tự tham chiếu đúng và ra 2 hàng: {'two_row_car': 5070, 'two_row_motorcycle': 1573} (theo loại xe). Những biển thất bại vẫn được giữ trong test runtime để tính skip/coverage.
- Train hai hàng sau lọc identity: 5,034 biển có crop tham chiếu, 40,681 crop ký tự; đủ 30/30 lớp. Tỉ số lớp nhiều mẫu nhất/ít nhất có mẫu = 201.5:1.

## Nhận dạng ký tự và exact biển trên crop tham chiếu

| Tập / loại xe | Mô hình trước khi thêm hai hàng | Sau fine-tune hai hàng |
|---|---:|---:|
| Test hai hàng, ký tự (crop tham chiếu) | 95.563% (5,858/6,130) | 96.476% (5,914/6,130) |
| Test hai hàng, exact chuỗi (crop tham chiếu) | 82.609% (627/759) | 85.244% (647/759) |
| Macro precision / recall / F1 | 92.481% / 85.350% / 87.719% | 90.594% / 89.860% / 89.896% |
| Ô tô hai hàng, ký tự / exact crop | 96.968% / 87.458% | 97.247% / 88.983% |
| Xe máy hai hàng, ký tự / exact crop | 91.149% / 65.680% | 94.054% / 72.189% |

Đây là accuracy OCR khi đã có crop ký tự được căn chỉnh, không bao gồm lỗi tìm biển/phân đoạn tự do. Tập test được tách theo chuỗi biển nhưng là một phân hoạch mới của nguồn train(1); nó có thể trùng dữ liệu đã dùng để pretrain checkpoint cũ. Vì vậy so sánh trước/sau đo tác dụng fine-tune trên phân hoạch hiện tại, chưa phải đánh giá external-dataset độc lập.

## Pipeline tự chạy, không biết trước số ký tự

| Chỉ số trên test thô | Trước fine-tune | Sau fine-tune |
|---|---:|---:|
| Độ phủ đầu ra hợp lệ 2 hàng | 55.6% (1155/2079) | 55.6% (1155/2079) |
| Exact biển end-to-end, bỏ qua tính là sai | 28.090% (584/2079) | 29.004% (603/2079) |
| Ký tự đúng, chỉ khi số crop khớp nhãn | 95.975% | 96.858% |
| Chuỗi có số lượng ký tự khớp nhãn | 703/2079 | 703/2079 |

Histogram số crop sau phân đoạn của mô hình đã fine-tune: `{"8": 688, "3": 92, "1": 31, "7": 428, "6": 170, "9": 172, "4": 161, "2": 52, "5": 188, "10": 13}`; hàng phát hiện: `{"2": 1341, "1": 654}`. Phần này quyết định exact end-to-end; accuracy CNN cao không bù được cắt thiếu/thừa ký tự.
- Pipeline không phát output ở 924/2,079 frame (44.44%); nguyên nhân: `{"not_two_rows": 654, "segmentation_failed": 84, "unsupported_character_count": 186}`. Ngoài ra có 452 output lệch số lượng ký tự; các trường hợp này không được dùng cho accuracy ký tự căn chỉnh.
- Character accuracy runtime chỉ trên các chuỗi có số crop khớp nhãn: 96.858% (5,487/5,665 ký tự); chỉ 703/2,079 frame căn chỉnh đủ số lượng.

## Độ chính xác theo từng lớp ký tự

Recall ở đây là accuracy trong lớp; support các lớp không có dữ liệu test bằng 0 thì không thể đánh giá. Xem CSV để đối chiếu baseline/fine-tune theo từng lớp.

| Ký tự | Số mẫu test | Trước: recall | Sau: recall | Sau: precision | Sau: F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 564 | 96.63% | 96.63% | 97.32% | 96.98% |
| 1 | 555 | 96.04% | 95.68% | 96.02% | 95.85% |
| 2 | 613 | 97.39% | 97.39% | 97.87% | 97.63% |
| 3 | 966 | 96.89% | 97.62% | 95.16% | 96.37% |
| 4 | 459 | 95.21% | 96.08% | 97.57% | 96.82% |
| 5 | 300 | 97.67% | 98.33% | 98.01% | 98.17% |
| 6 | 815 | 96.81% | 96.93% | 97.17% | 97.05% |
| 7 | 309 | 92.56% | 96.44% | 96.13% | 96.28% |
| 8 | 313 | 95.53% | 97.12% | 96.20% | 96.66% |
| 9 | 468 | 94.87% | 96.37% | 98.69% | 97.51% |
| A | 363 | 96.69% | 96.69% | 96.69% | 96.69% |
| B | 54 | 77.78% | 85.19% | 88.46% | 86.79% |
| C | 112 | 96.43% | 97.32% | 96.46% | 96.89% |
| D | 23 | 86.96% | 91.30% | 84.00% | 87.50% |
| E | 24 | 91.67% | 91.67% | 100.00% | 95.65% |
| F | 8 | 100.00% | 100.00% | 88.89% | 94.12% |
| G | 11 | 100.00% | 100.00% | 84.62% | 91.67% |
| H | 17 | 100.00% | 100.00% | 100.00% | 100.00% |
| K | 12 | 66.67% | 83.33% | 76.92% | 80.00% |
| L | 14 | 78.57% | 92.86% | 81.25% | 86.67% |
| M | 32 | 84.38% | 93.75% | 88.24% | 90.91% |
| N | 32 | 81.25% | 84.38% | 93.10% | 88.52% |
| P | 13 | 92.31% | 92.31% | 85.71% | 88.89% |
| S | 11 | 81.82% | 90.91% | 100.00% | 95.24% |
| T | 4 | 50.00% | 50.00% | 66.67% | 57.14% |
| U | 5 | 40.00% | 40.00% | 66.67% | 50.00% |
| V | 8 | 62.50% | 100.00% | 100.00% | 100.00% |
| X | 9 | 88.89% | 100.00% | 90.00% | 94.74% |
| Y | 8 | 100.00% | 100.00% | 100.00% | 100.00% |
| Z | 8 | 25.00% | 37.50% | 60.00% | 46.15% |

## Loss, hard-example mining và augmentation

- Loss: weighted cross-entropy + λ=0.25 × kỳ vọng ma trận phạt nhầm lẫn M; OHEM top 25% được tăng trọng số, không bỏ phần còn lại.
- Tối ưu: AdamW, LR đầu 0.0001, cosine annealing, batch 256; hoàn thành 12 epoch, checkpoint chọn theo accuracy ký tự validation trước, exact biển làm tiêu chí phụ; epoch tốt nhất 11.
- Replay giữ kiến thức biển một hàng: 6,776 ký tự trong 847 biển train, trộn cùng ký tự hai hàng để hạn chế quên pha 1.
- Augmentation quan sát được trong train: `{"geometry:none": 0.550187538192469, "photometric:brightness_contrast": 0.23044018795962662, "photometric:none": 0.4498440693680595, "photometric:noise": 0.22023094590892808, "geometry:affine": 0.12032647098074749, "geometry:perspective": 0.07939116814519812, "photometric:blur": 0.0994847967633858, "geometry:rotation": 0.25009482268158545}`; gồm xoay/affine/perspective nhẹ, blur/noise/độ sáng-contrast.

## Tham số, MAC, bộ nhớ và tốc độ

- Dense LeNet-5 (Conv1=6, Conv2=16, FC=120/84, 30 lớp): **63,406 tham số**, **418,200 MAC/ký tự**.
- MAC theo độ dài: 7 ký tự = 2,927,400; 8 ký tự = 3,345,600; 9 ký tự = 3,763,800; 10 ký tự = 4,182,000.
- Trọng số thô: FP32 253,624 B (247.68 KiB); 8-bit lý thuyết 63,406 B (chưa phải mô hình lượng tử hóa kiểm định). Đỉnh activation riêng lẻ 4,704 giá trị = 18,816 B FP32/ký tự.
- CPU PyTorch 1 luồng, p50/p95 batch 1: trước 0.1915/0.2901 ms, sau 0.1898/0.2949 ms; batch 8 p50: trước 0.4777 ms, sau 0.4753 ms. Cùng topology/MAC nên đây không phải tăng tốc kiến trúc.
- Ước lượng số học nếu DE10-Lite chạy 50 MHz và một MAC/chu kỳ: 8 ký tự 66.912 ms; 9 ký tự 75.276 ms; 10 ký tự 83.640 ms. Chưa có RTL AI/Quartus synthesis nên không có fMAX, DSP/ALM/BRAM hoặc latency FPGA đo thật.

## Kết luận và giới hạn

Training fine-tune này tối ưu nhận dạng ký tự có crop tham chiếu trước, sau đó mới nhìn exact biển. Nếu exact/coverage runtime thấp hơn rõ rệt so với crop reference thì nút thắt là ROI, phân đoạn hoặc row-sort chứ không chỉ là LeNet-5. Các biển xe máy/ô tô có số ký tự khác nhau được so khớp theo chuỗi nhãn thực tế; không giả định mọi biển có cùng chiều dài.

Kết quả có thể bị ảnh hưởng bởi nhãn filename chưa kiểm tra thủ công toàn bộ, nhóm test mới có khả năng đã xuất hiện trong pretraining cũ, và các crop tham chiếu dùng số lượng ký tự nhãn để chọn phân đoạn. Confidence softmax chưa được calibration. Báo cáo không tuyên bố đạt 95–99% trên camera/FPGA nếu số liệu end-to-end chưa đạt.

Số liệu đầy đủ JSON: `artifacts/vscode_demo/two_row_dense_metrics.json`; báo cáo theo lớp: `artifacts/vscode_demo/two_row_test_per_class_comparison.csv`; lịch sử epoch: `artifacts/vscode_demo/two_row_dense_training_history.csv`. Ảnh từng biển/ID không được đưa lên GitHub.

### Số lượng theo split nguồn (trước lọc replay chéo)

| Split | Ô tô hai hàng | Xe máy hai hàng | Tổng frame |
|---|---:|---:|---:|
| train | 9,335 | 4,291 | 13,626 |
| val | 1,445 | 610 | 2,055 |
| test | 1,444 | 635 | 2,079 |

## Phụ lục: runtime theo loại xe/độ dài và confidence

| Nhóm test thô | Số frame | Độ phủ | Exact, skip tính sai | Exact trong output hợp lệ |
|---|---:|---:|---:|---:|
| Ô tô hai hàng | 1444 | 58.17% | 34.42% (497/1444) | 59.17% |
| Xe máy hai hàng | 635 | 49.61% | 16.69% (106/635) | 33.65% |
| 7 ký tự | 196 | 37.24% | 28.06% (55/196) | 75.34% |
| 8 ký tự | 1407 | 59.35% | 33.26% (468/1407) | 56.05% |
| 9 ký tự | 472 | 51.91% | 16.95% (80/472) | 32.65% |
| 10 ký tự | 4 | 50.00% | 0.00% (0/4) | 0.00% |

Confidence softmax trung bình trên crop tham chiếu: 0.9528; trung bình ký tự đúng 0.9627; ký tự sai 0.6794 nếu có. Confidence chưa calibration nên không được hiểu như xác suất đúng đã hiệu chuẩn.
Accuracy theo vị trí chuỗi 1-based trên crop tham chiếu: `{"1": {"correct": 744, "total": 759, "accuracy": 0.9802371541501976}, "2": {"correct": 731, "total": 759, "accuracy": 0.9631093544137023}, "3": {"correct": 711, "total": 759, "accuracy": 0.9367588932806324}, "4": {"correct": 722, "total": 759, "accuracy": 0.9512516469038208}, "5": {"correct": 737, "total": 759, "accuracy": 0.9710144927536232}, "6": {"correct": 738, "total": 759, "accuracy": 0.9723320158102767}, "7": {"correct": 738, "total": 759, "accuracy": 0.9723320158102767}, "8": {"correct": 669, "total": 685, "accuracy": 0.9766423357664233}, "9": {"correct": 123, "total": 131, "accuracy": 0.9389312977099237}, "10": {"correct": 1, "total": 1, "accuracy": 1.0}}`.

Activation được khảo sát trên 256 glyph validation; số liệu min/max/mean/std/quantile/zero fraction từng layer nằm trong JSON. Ma trận nhầm lẫn test 30×30 và ma trận cost M 30×30 được xuất CSV; tập test không tham gia xây M.

## Chạy lại từ VS Code

Trong thư mục gốc repo, dùng Python interpreter `.venv` rồi chạy lần lượt:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u prepare_two_row_frontal_dense.py
.\.venv\Scripts\python.exe -X utf8 -u train_two_row_dense.py --epochs 12 --batch-size 256
```
