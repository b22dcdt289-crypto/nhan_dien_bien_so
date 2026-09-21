from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_lenet5 import CLASS_NAMES, LeNet5


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]], dtype=np.float32)


def warp_plate(image, contour):
    points = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    points = order_points(points)
    width = int(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    if height > width:
        points = np.roll(points, 1, axis=0)
        width, height = height, width
    if width < 80 or height < 15:
        return None
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def find_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    masks = []
    edges = cv2.Canny(gray, 60, 180)
    masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))))
    bright = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]
    masks.append(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))))
    candidates = []
    image_area = image.shape[0] * image.shape[1]
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / max(h, 1)
            if area < image_area * 0.0005 or ratio < 1.3 or ratio > 8.5:
                continue
            # Prefer plate-like rectangles while still allowing foreshortened views.
            ratio_score = max(0.15, 1.0 - abs(ratio - 3.0) / 3.0)
            candidates.append((area * ratio_score, contour))
    for _, contour in sorted(candidates, reverse=True, key=lambda item: item[0]):
        plate = warp_plate(image, contour)
        if plate is not None:
            return plate
    return None


def segment_characters(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary[:2, :] = binary[-2:, :] = 0
    binary[:, :2] = binary[:, -2:] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if h * 0.22 <= ch <= h * 0.95 and w * 0.015 <= cw <= w * 0.35:
            boxes.append((x, y, cw, ch))
    if len(boxes) < 3:
        return []
    centers = sorted(y + ch / 2 for _, y, _, ch in boxes)
    if centers[-1] - centers[0] > h * 0.22:
        split = (centers[0] + centers[-1]) / 2
        rows = [sorted([b for b in boxes if b[1] + b[3] / 2 <= split], key=lambda b: b[0]), sorted([b for b in boxes if b[1] + b[3] / 2 > split], key=lambda b: b[0])]
        boxes = rows[0] + rows[1]
    else:
        boxes = sorted(boxes, key=lambda b: b[0])
    return [(gray[max(0, y - 2):min(h, y + ch + 2), max(0, x - 2):min(w, x + cw + 2)]) for x, y, cw, ch in boxes]


def main():
    parser = argparse.ArgumentParser(description="Vietnamese license plate OCR with LeNet-5")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/lenet5_vnlp_37k_pruned25.pt"))
    parser.add_argument("--box", type=str, help="optional plate box x,y,w,h when automatic detection fails")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    if args.box:
        x, y, w, h = (int(v) for v in args.box.split(","))
        plate = image[y:y + h, x:x + w]
    else:
        plate = find_plate(image)
    if plate is None:
        raise RuntimeError("Không tìm thấy biển số. Hãy thử dùng --box x,y,w,h.")
    crops = segment_characters(plate)
    if not crops:
        raise RuntimeError("Không tách được ký tự biển số.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch = []
    for crop in crops:
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        batch.append(torch.from_numpy(((crop - 0.5) / 0.5)).unsqueeze(0))
    with torch.no_grad():
        predictions = model(torch.stack(batch).to(device)).argmax(1).cpu().tolist()
    text = "".join(CLASS_NAMES[index] for index in predictions)
    print(f"plate={text}")
    print(f"characters={len(crops)} model={args.model} pruning={checkpoint.get('pruning', 0)}")


if __name__ == "__main__":
    main()
