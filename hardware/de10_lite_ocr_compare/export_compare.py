"""Quantize the matched checkpoints and export identical validation crops to four builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware.de10_lite_ocr.prepare_demo import DATA, fixed_forward, load_samples, write_hex
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5


LAYERS = ("features.0", "features.3", "classifier.0", "classifier.2", "classifier.4")
VARIANTS = {
    "dense_rom": ("dense", False),
    "dense_stream": ("dense", True),
    "pruned_rom": ("conv2_pruned", False),
    "pruned_stream": ("conv2_pruned", True),
}


def quantize(state: dict) -> tuple[dict, list[int], list[int], list[int]]:
    specs: dict = {}
    weights: list[int] = []
    biases: list[int] = []
    shifts: list[int] = []
    for name in LAYERS:
        w = state[name + ".weight"].detach().cpu().numpy()
        b = state[name + ".bias"].detach().cpu().numpy()
        max_abs = float(np.max(np.abs(w)))
        if max_abs <= 0:
            raise ValueError(f"Zero weights in {name}")
        exponent = max(0, math.floor(math.log2(127.0 / max_abs)))
        wq = np.rint(w * (1 << exponent)).astype(np.int32)
        bq = np.rint(b * (1 << (exponent + 7))).astype(np.int64)
        if np.max(np.abs(wq)) > 127 or np.max(np.abs(bq)) >= (1 << 31):
            raise ValueError(f"INT8/INT32 overflow in {name}")
        shift = exponent + 2
        if shift > 31:
            raise ValueError(f"Shift exceeds 5-bit LUT control in {name}")
        specs[name] = {
            "weight_exponent": exponent, "weight_offset": len(weights),
            "bias_offset": len(biases), "weight_shape": list(w.shape),
        }
        weights.extend(int(value) for value in wq.flatten())
        biases.extend(int(value) for value in bq.flatten())
        shifts.append(shift)
    return specs, weights, biases, shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/compare_conv2_stream_20260929")
    parser.add_argument("--pruned-file", default="conv2_pruned.pt", help="Filename of the compact Conv2 checkpoint in --models.")
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--plates-per-type", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plates = load_samples(args.seed, args.plates_per_type)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for plate in plates:
        for character, relative in zip(plate["text"], plate["crop_files"]):
            with Image.open(DATA / relative) as image:
                arr = np.asarray(image.convert("L"), dtype=np.uint8)
            if arr.shape != (32, 32):
                raise ValueError(f"Expected 32x32 crop: {relative}")
            images.append(arr)
            labels.append(character)
    image_array = np.stack(images)
    image_q = np.rint((image_array.astype(np.float32) * (2 / 255.0) - 1) * 127).astype(np.int8)
    float_input = torch.from_numpy(image_array.astype(np.float32) / 127.5 - 1).unsqueeze(1)
    quant_input = torch.from_numpy(image_q.astype(np.float32)).unsqueeze(1)
    lut = [int(np.rint(127 * math.tanh(index / 32))) for index in range(-128, 129)]
    summaries = {}
    for architecture, cls in (("dense", LeNet5), ("conv2_pruned", LeNet5Conv2Pruned)):
        checkpoint_path = args.models / (args.pruned_file if architecture == "conv2_pruned" else "dense.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if list(checkpoint["classes"]) != CLASS_NAMES:
            raise ValueError("Class order differs between matched checkpoints")
        model = cls(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        specs, weights, biases, shifts = quantize(checkpoint["model"])
        expected_weights = 63150 if architecture == "dense" else 37950
        if len(weights) != expected_weights:
            raise AssertionError(f"Unexpected {architecture} weight count {len(weights)}")
        with torch.inference_mode():
            fp32_ids = model(float_input).argmax(1).tolist()
            fixed_ids = fixed_forward(quant_input, specs, weights, biases, lut).argmax(1).tolist()
        fp32 = [CLASS_NAMES[index] for index in fp32_ids]
        fixed = [CLASS_NAMES[index] for index in fixed_ids]
        plate_rows = []
        offset = 0
        for plate in plates:
            count = len(plate["text"])
            plate_rows.append({
                "type": plate["type"], "source": plate["source"],
                "truth": plate["text"],
                "fp32": "".join(fp32[offset:offset + count]),
                "fixed": "".join(fixed[offset:offset + count]),
                "sample_indices": list(range(offset, offset + count)),
            })
            offset += count
        summary = {
            "architecture": architecture, "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "data_manifest": str(DATA / "manifest.json"), "seed": args.seed,
            "plate_count": len(plates), "character_count": len(labels),
            "class_order": CLASS_NAMES, "layers": specs, "shifts": shifts,
            "weight_bytes": len(weights), "bias_count": len(biases),
            "fp32_character_correct": sum(a == b for a, b in zip(fp32, labels)),
            "fixed_character_correct": sum(a == b for a, b in zip(fixed, labels)),
            "fp32_plate_correct": sum(item["fp32"] == item["truth"] for item in plate_rows),
            "fixed_plate_correct": sum(item["fixed"] == item["truth"] for item in plate_rows),
            "plates": plate_rows,
            "samples": [{"label": label, "fp32": fp32[i], "fixed": fixed[i]}
                        for i, label in enumerate(labels)],
        }
        for variant, (variant_arch, streaming) in VARIANTS.items():
            if variant_arch != architecture:
                continue
            out = args.out_root / variant / "generated"
            out.mkdir(parents=True, exist_ok=True)
            write_hex(out / "images.hex", [int(value) for value in image_q.flatten()], 8)
            write_hex(out / "weights.hex", weights, 8)
            write_hex(out / "biases.hex", biases, 32)
            write_hex(out / "tanh.hex", lut, 8)
            write_hex(out / "shifts.hex", shifts, 8)
            (out / "manifest.json").write_text(json.dumps({**summary, "variant": variant,
                                                            "streaming_input": streaming},
                                                           ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[architecture] = {key: summary[key] for key in (
            "checkpoint_sha256", "plate_count", "character_count", "weight_bytes",
            "bias_count", "shifts", "fp32_character_correct", "fixed_character_correct",
            "fp32_plate_correct", "fixed_plate_correct")}
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
