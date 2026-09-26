from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from recognize_plate import segment_characters
from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes
from structured_prune_lenet5 import LeNet5Structured


def main():
    parser = argparse.ArgumentParser(description="Plate detector + LeNet-5 character OCR")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/lenet5_ocr_structured_pruned25_final.pt"))
    parser.add_argument("--detector", type=Path, default=Path("source/model/LP_detector.pt"))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    detector = torch.hub.load("yolov5", "custom", path=str(args.detector), source="local", force_reload=False)
    detector.conf = 0.25
    result = detector(image, size=640)
    boxes = result.pandas().xyxy[0].sort_values("confidence", ascending=False)
    if boxes.empty:
        raise RuntimeError("Detector không tìm thấy biển số trong ảnh.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, args.model)
    model_cls = LeNet5Structured if checkpoint.get("arch") == "LeNet5Structured" else LeNet5
    model = model_cls(len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for i, row in boxes.iterrows():
        x1, y1, x2, y2 = (max(0, int(row[k])) for k in ("xmin", "ymin", "xmax", "ymax"))
        plate = image[y1:y2, x1:x2]
        crops = segment_characters(plate)
        if not crops:
            print(f"box={x1},{y1},{x2},{y2} không tách được ký tự")
            continue
        batch = []
        for crop in crops:
            crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
            batch.append(torch.from_numpy(((crop - 0.5) / 0.5)).unsqueeze(0))
        with torch.no_grad():
            ids = model(torch.stack(batch).to(device)).argmax(1).cpu().tolist()
        text = "".join(CLASS_NAMES[index] for index in ids)
        print(f"plate={text} confidence={row['confidence']:.3f} box={x1},{y1},{x2},{y2}")


if __name__ == "__main__":
    main()
