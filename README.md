# Vietnamese license plate OCR with LeNet-5

This project trains a LeNet-5 character classifier from YOLO character boxes.

## Run

```powershell
.venv\Scripts\python.exe train_lenet5.py --data data/OCR/OCR --epochs 5
```

The best dense checkpoint is written to `artifacts/lenet5_chars.pt`.

## Trained checkpoints

The current OCR training run used 24,848 training character crops and 6,236
validation crops from the labelled OCR set. The best dense model reached
96.82% validation accuracy. After global unstructured pruning at 25% and
fine-tuning, `artifacts/lenet5_ocr_pruned25_final.pt` reached 96.89%.

## Structured pruning for DE10-Lite

`structured_prune_lenet5.py` removes complete convolution filters and fully
connected neurons. The deployed model uses channels `(4, 12, 90, 63)` instead
of `(6, 16, 120, 84)`, reducing theoretical MACs from 418,704 to 233,338
per character (about 44.3% fewer). A Vietnamese plate is treated as 8
characters whether it is printed on one row or two rows, so the OCR cost is
about 1.87 million MAC per plate; two rows only change layout/order handling.
After fine-tuning, validation accuracy was
95.22%. The MAC reduction is physical in the network dimensions and is
suitable for mapping to fewer FPGA multipliers.

```powershell
.venv\Scripts\python.exe structured_prune_lenet5.py --data data/OCR/OCR --init artifacts/lenet5_ocr_97target_final.pt --epochs 10 --output artifacts/lenet5_ocr_structured_pruned25.pt
```

The full LeNet-5 has 418,704 theoretical MACs per 32x32 character. With 25%
unstructured sparsity, the theoretical layer shape is unchanged; a sparse
accelerator can skip approximately 104,676 zero-weight MACs, leaving about
314,028 non-zero weight products. To reduce the physical MAC count on a
DE10-Lite, use structured channel pruning or the included `LeNet5Lite` model;
unstructured pruning alone does not remove DSP/multiplier instances.

Run the pruned OCR model on an image after the detector is available:

```powershell
.venv\Scripts\python.exe recognize_plate_yolo.py path\to\plate.jpg
```

The current dataset contains plate images and per-character bounding boxes. The model recognizes cropped characters; a separate plate detector/segmenter is required for end-to-end camera recognition.

## VNLP 37k-image retraining

```powershell
.venv\Scripts\python.exe prepare_vnlp_chars.py
.venv\Scripts\python.exe train_lenet5_folder.py --data data/VNLP_chars --epochs 5 --output artifacts/lenet5_vnlp_37k.pt
```

The corrected VNLP extraction is kept under `data/VNLP_chars_corrected` and
contains 37,297 source images; automated character segmentation accepts a
subset because images with ambiguous segmentation are rejected rather than
assigned noisy labels.

## Train with `images/train(1)`

The `train(1)` source contains 37,297 full plate images: 19,086 one-row,
12,618 two-row car plates, and 5,593 two-row motorcycle plates. The filename
contains the plate text and plate box, so `prepare_vnlp_chars.py` can generate
character folders:

```powershell
.venv\Scripts\python.exe prepare_vnlp_chars.py --root "data/OCR/OCR/images/train(1)/detection" --out data/VNLP_chars_train1
.venv\Scripts\python.exe finetune_structured_folder.py --data data/VNLP_chars_train1 --init artifacts/lenet5_ocr_structured_pruned25_final.pt --epochs 10 --lr 1e-4 --output artifacts/lenet5_train1_structured_pruned25.pt
```

This run accepted 13,779 images and generated 109,869 character crops. The
structured model reached 89.43% character validation accuracy. This is lower
than the 95.22% OCR-labelled baseline because automatic contour segmentation
rejects or misaligns many full-plate images; it is not a plate-level accuracy.

## Independent pipeline without YOLOv5

The independent prototype replaces YOLOv5 with classical image processing that
can be reimplemented in FPGA HDL: contour/morphology candidate detection,
perspective correction, row classification, left-to-right sorting within each
row, and LeNet-5 OCR. The laptop is used only as a reference runner during
development; a standalone FPGA implementation must implement these same stages
in hardware.

Prepare the full `train(1)` source and train the approximately 50% structured
LeNet-5 model:

```powershell
.venv\Scripts\python.exe train_independent_structured.py --mode all --clean --dense-epochs 3 --structured-epochs 8
```

Run the independent inference pipeline without YOLOv5:

```powershell
.venv\Scripts\python.exe recognize_independent.py path\to\frame.jpg
```

The completed run used all 37,297 source images, accepted 24,888 images and
created 199,412 character crops. Character validation accuracy was 90.58% and
plate-level exact match was 78.02%. By source type, one-row plates reached
97.20% character accuracy, two-row car plates 92.77%, and two-row motorcycle
plates 60.38%. The lower motorcycle result shows that the row/character
segmentation stage still needs targeted improvement; it should not be reported
as a 95%+ end-to-end system yet.

The compact model uses channels `(4, 10, 100, 70)`: 212,920 MAC per character,
about 49.15% fewer than dense LeNet-5. The resulting MAC counts are 1,703,360
for 8 characters, 1,916,280 for 9 characters and 2,129,200 for 10 characters.
The preparation skip rate was 33.27% overall; using Laplacian variance below
80 as a blur heuristic, the skip rate was 44.32% (39/88), and using an estimated
perspective angle above 8 degrees, it was 43.16% (41/95). These are preprocessing
heuristics, not camera ground-truth labels, and are stored in
`artifacts/independent_pipeline_metrics.json`.

During training-data preparation, the plate text encoded in each filename is
used only to reject a crop when the detected component count does not match the
known label. At inference, no text label is available: the prototype accepts
only 8, 9 or 10 detected characters and skips an ambiguous frame instead of
returning a fabricated plate string.

## Online labeled-image smoke test

`eval_online_labeled.py` evaluates public plate crops whose ground-truth text
is encoded before the first underscore in the filename, for example
`30A12345_0001_0.jpg`. It reports exact plate accuracy on attempted images,
end-to-end accuracy including skipped images, character position accuracy and
skip reasons:

```powershell
.venv\Scripts\python.exe eval_online_labeled.py path\to\online_test
```

The current reproducible smoke test used 69 public GitHub images and produced
`55/69 = 79.71%` end-to-end exact plate accuracy, `55/65 = 84.62%` exact
accuracy among attempted images, `93.45%` character-position accuracy and a
`5.80%` skip rate. This is an external one-row plate-crop smoke test, not a
DE10-Lite hardware accuracy result and not a replacement for a held-out
Vietnamese two-row test set. The detailed result is in
`artifacts/online_test_metrics.json`.

## Reproduce pruning

```powershell
.venv\Scripts\python.exe prune_lenet5_ocr.py --data data/OCR/OCR --init artifacts/lenet5_ocr_97target_final.pt --amount 0.25 --epochs 10 --output artifacts/lenet5_ocr_pruned25_final.pt
```
