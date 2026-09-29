"""Export a reproducible, *reference-crop* LeNet demo for DE10-Lite.

The generated ROMs contain cropped 32x32 characters, not full camera images.
They are local-only because labels and source names can identify real plates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES


DATA = ROOT / "data/independent_chars_train1_structured_channel30_v1"
CHECKPOINT = ROOT / "artifacts/lenet5_conv2_pruned_kd_channel30_v1.pt"
LAYERS = (
    ("features.0", 150, 6),
    ("features.3", 1200, 8),
    ("classifier.0", 24000, 120),
    ("classifier.2", 10080, 84),
    ("classifier.4", 2520, 30),
)


def signed_hex(value: int, bits: int) -> str:
    return f"{value & ((1 << bits) - 1):0{bits // 4}x}"


def write_hex(path: Path, values: list[int], bits: int) -> None:
    path.write_text("\n".join(signed_hex(x, bits) for x in values) + "\n", encoding="ascii")


def load_samples(seed: int, plates_per_type: int) -> list[dict]:
    records = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    chosen = []
    for kind in ("one_row", "two_row_car", "two_row_motorcycle"):
        candidates = [
            item for item in records
            if item["split"] == "val" and item["type"] == kind
            and len(item["crop_files"]) == len(item["text"])
            and all((DATA / name).is_file() for name in item["crop_files"])
        ]
        if len(candidates) < plates_per_type:
            raise RuntimeError(f"Only {len(candidates)} valid plates of type {kind}")
        chosen.extend(rng.sample(candidates, plates_per_type))
    rng.shuffle(chosen)
    return chosen


def quantize_parameters(state: dict) -> tuple[dict, list[int], list[int]]:
    specs = {}
    weights: list[int] = []
    biases: list[int] = []
    for name, weight_count, bias_count in LAYERS:
        w = state[name + ".weight"].detach().cpu().numpy()
        b = state[name + ".bias"].detach().cpu().numpy()
        max_abs = float(np.max(np.abs(w)))
        exponent = max(0, math.floor(math.log2(127.0 / max_abs)))
        wq = np.rint(w * (1 << exponent)).astype(np.int32)
        if np.max(np.abs(wq)) > 127:
            raise AssertionError(f"Weight overflow in {name}")
        bq = np.rint(b * (1 << (exponent + 7))).astype(np.int64)
        if np.max(np.abs(bq)) >= (1 << 31):
            raise AssertionError(f"Bias overflow in {name}")
        if wq.size != weight_count or bq.size != bias_count:
            raise AssertionError(f"Unexpected tensor shape in {name}")
        specs[name] = {
            "weight_exponent": exponent,
            "weight_offset": len(weights),
            "bias_offset": len(biases),
            "weight_shape": list(w.shape),
        }
        weights.extend(int(x) for x in wq.flatten())
        biases.extend(int(x) for x in bq.flatten())
    return specs, weights, biases


def fixed_forward(image_q: torch.Tensor, specs: dict, weights: list[int], biases: list[int], lut: list[int]) -> torch.Tensor:
    device = image_q.device
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    bias_tensor = torch.tensor(biases, dtype=torch.float32, device=device)
    lut_tensor = torch.tensor(lut, dtype=torch.float32, device=device)

    def layer(name: str, x: torch.Tensor, tanh: bool) -> torch.Tensor:
        spec = specs[name]
        shape = spec["weight_shape"]
        start = spec["weight_offset"]
        bs = spec["bias_offset"]
        w = weight_tensor[start:start + math.prod(shape)].reshape(shape)
        b = bias_tensor[bs:bs + shape[0]]
        if len(shape) == 4:
            acc = F.conv2d(x, w, bias=b)
        else:
            acc = F.linear(x, w, bias=b)
        if not tanh:
            return acc
        # Hardware: arithmetic right shift, saturate LUT index to [-128,128].
        indices = torch.floor(acc / (1 << (spec["weight_exponent"] + 2))).to(torch.long)
        return lut_tensor[indices.clamp(-128, 128) + 128]

    def pool(x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = torch.ones((channels, 1, 2, 2), dtype=torch.float32, device=device)
        sums = F.conv2d(x, kernel, stride=2, groups=channels)
        return torch.floor((sums + 2.0) / 4.0).clamp(-128, 127)

    x = pool(layer("features.0", image_q, True))
    x = pool(layer("features.3", x, True))
    x = x.flatten(1)
    x = layer("classifier.0", x, True)
    x = layer("classifier.2", x, True)
    return layer("classifier.4", x, False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--plates-per-type", type=int, default=2)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "generated"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if list(checkpoint["classes"]) != CLASS_NAMES:
        raise ValueError("Unexpected class order")
    model = LeNet5Conv2Pruned().eval()
    model.load_state_dict(checkpoint["model"])
    specs, weights, biases = quantize_parameters(checkpoint["model"])
    lut = [int(np.rint(127.0 * math.tanh(index / 32.0))) for index in range(-128, 129)]
    plates = load_samples(args.seed, args.plates_per_type)
    pixels = []
    labels = []
    for plate in plates:
        for char, name in zip(plate["text"], plate["crop_files"]):
            with Image.open(DATA / name) as image:
                arr = np.asarray(image.convert("L"), dtype=np.uint8)
            if arr.shape != (32, 32):
                raise ValueError(f"Bad crop size: {name}: {arr.shape}")
            pixels.append(arr)
            labels.append(char)
    image_q = np.rint((np.stack(pixels).astype(np.float32) * (2.0 / 255.0) - 1.0) * 127.0).astype(np.int8)
    with torch.no_grad():
        x_float = torch.from_numpy(np.stack(pixels).astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
        x_quant = torch.from_numpy(image_q.astype(np.float32)).unsqueeze(1)
        fp32_idx = model(x_float).argmax(1).tolist()
        fixed_idx = fixed_forward(x_quant, specs, weights, biases, lut).argmax(1).tolist()
    fp32 = [CLASS_NAMES[index] for index in fp32_idx]
    fixed = [CLASS_NAMES[index] for index in fixed_idx]
    plate_rows = []
    cursor = 0
    for plate in plates:
        size = len(plate["text"])
        plate_rows.append({
            "type": plate["type"],
            "source": plate["source"],
            "truth": plate["text"],
            "fp32": "".join(fp32[cursor:cursor + size]),
            "fixed": "".join(fixed[cursor:cursor + size]),
            "sample_indices": list(range(cursor, cursor + size)),
        })
        cursor += size
    images = [int(v) for v in image_q.flatten()]
    write_hex(out / "images.hex", images, 8)
    write_hex(out / "weights.hex", weights, 8)
    write_hex(out / "biases.hex", biases, 32)
    write_hex(out / "tanh.hex", lut, 8)
    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "data_manifest": str(DATA / "manifest.json"),
        "seed": args.seed,
        "plates_per_type": args.plates_per_type,
        "plate_count": len(plates),
        "character_count": len(labels),
        "class_order": CLASS_NAMES,
        "layers": specs,
        "weight_bytes": len(weights),
        "bias_count": len(biases),
        "fp32_character_correct": sum(a == b for a, b in zip(fp32, labels)),
        "fixed_character_correct": sum(a == b for a, b in zip(fixed, labels)),
        "fp32_plate_correct": sum(row["fp32"] == row["truth"] for row in plate_rows),
        "fixed_plate_correct": sum(row["fixed"] == row["truth"] for row in plate_rows),
        "plates": plate_rows,
        "samples": [
            {"label": label, "fp32": fp32[i], "fixed": fixed[i]}
            for i, label in enumerate(labels)
        ],
    }
    (out / "demo_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "seed", "plate_count", "character_count", "weight_bytes", "fp32_character_correct",
        "fixed_character_correct", "fp32_plate_correct", "fixed_plate_correct",
    )}, indent=2))


if __name__ == "__main__":
    main()
