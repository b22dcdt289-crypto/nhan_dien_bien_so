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

## Reproduce pruning

```powershell
.venv\Scripts\python.exe prune_lenet5_ocr.py --data data/OCR/OCR --init artifacts/lenet5_ocr_97target_final.pt --amount 0.25 --epochs 10 --output artifacts/lenet5_ocr_pruned25_final.pt
```
