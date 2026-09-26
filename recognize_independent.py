from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_independent_structured import LeNet5Structured50, perspective_correct, segment_characters
from train_lenet5 import CLASS_NAMES, LeNet5, require_compatible_classes


def find_plate_candidate(image: np.ndarray, return_box: bool = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    masks = [
        cv2.morphologyEx(cv2.Canny(gray, 50, 170), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))),
        cv2.morphologyEx(cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))),
    ]
    image_area = image.shape[0] * image.shape[1]
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / max(1, h)
            area = w * h
            if area < image_area * 0.0005 or ratio < 1.2 or ratio > 9.5:
                continue
            score = area * max(0.1, 1.0 - abs(ratio - 2.5) / 3.5)
            candidates.append((score, x, y, w, h))
    for _, x, y, w, h in sorted(candidates, reverse=True):
        crop = image[y : y + h, x : x + w]
        corrected, applied, angle = perspective_correct(crop)
        if corrected.size:
            if return_box:
                return corrected, applied, angle, (x, y, w, h)
            return corrected, applied, angle
    return None


def main():
    parser = argparse.ArgumentParser(description="Independent classical-CV + LeNet-5 license plate OCR")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/independent_lenet5_structured50_train1.pt"))
    parser.add_argument("--box", type=str, help="optional x,y,w,h plate box")
    parser.add_argument("--plate-crop", action="store_true", help="input image is already a cropped plate ROI")
    parser.add_argument("--enhancement", choices=("none", "clahe", "clahe_sharp"), default=None)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    if args.plate_crop:
        plate, applied, angle = perspective_correct(image)
    elif args.box:
        x, y, w, h = (int(value) for value in args.box.split(","))
        plate, applied, angle = perspective_correct(image[y : y + h, x : x + w])
    else:
        found = find_plate_candidate(image)
        if found is None:
            raise RuntimeError("Không tìm thấy biển số bằng contour/morphology. Hãy thử --box x,y,w,h.")
        plate, applied, angle = found
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    require_compatible_classes(checkpoint, args.model)
    enhancement = args.enhancement or checkpoint.get("enhancement", "none")
    segmented = segment_characters(plate, expected_count=None, enhancement=enhancement)
    if segmented is None:
        print("frame_skipped=1 reason=segmentation_failed")
        return
    crops, row_count, char_count = segmented
    if char_count not in (7, 8, 9, 10):
        print(f"frame_skipped=1 reason=ambiguous_character_count characters={char_count} rows={row_count}")
        return
    architecture = checkpoint.get("arch", "LeNet5Structured50")
    if architecture in {"LeNet5", "LeNet5_dense_no_pruning"}:
        model = LeNet5(len(CLASS_NAMES)).to(device)
        macs_per_character = 418200
        model_name = "LeNet5_dense_no_pruning"
    else:
        model = LeNet5Structured50(len(CLASS_NAMES)).to(device)
        macs_per_character = checkpoint.get("macs_per_character", LeNet5Structured50.macs())
        model_name = "LeNet5Structured50"
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch = []
    for crop in crops:
        pixels = crop.astype(np.float32) / 255.0
        batch.append(torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0))
    with torch.no_grad():
        predictions = model(torch.stack(batch).to(device)).argmax(1).cpu().tolist()
    text = "".join(CLASS_NAMES[index] for index in predictions)
    print(f"plate={text}")
    print(f"characters={char_count} rows={row_count} perspective_corrected={applied} angle_deg={angle:.2f}")
    print(f"plate_enhancement={enhancement}")
    print(f"macs_per_character={macs_per_character} macs_this_plate={macs_per_character * char_count}")
    print(f"detector=classical contour+morphology; ocr={model_name}; yolo=none")


if __name__ == "__main__":
    main()
