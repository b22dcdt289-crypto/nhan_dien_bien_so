# Vietnamese license plate OCR with LeNet-5

This project trains a LeNet-5 character classifier from YOLO character boxes.

## Run

```powershell
.venv\Scripts\python.exe train_lenet5.py --data data/OCR/OCR --epochs 5
```

The best checkpoint is written to `artifacts/lenet5_chars.pt`.

The current dataset contains plate images and per-character bounding boxes. The model recognizes cropped characters; a separate plate detector/segmenter is required for end-to-end camera recognition.

## VNLP 37k-image retraining

```powershell
.venv\Scripts\python.exe prepare_vnlp_chars.py
.venv\Scripts\python.exe train_lenet5_folder.py --data data/VNLP_chars --epochs 5 --output artifacts/lenet5_vnlp_37k.pt
```
