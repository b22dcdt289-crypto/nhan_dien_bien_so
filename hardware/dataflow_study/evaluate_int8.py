"""Compare FP32 and exported INT8-reference LeNet on 400 real test plates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetune_mixed_car_compare import PREPARED, prepared_cars
from hardware.de10_lite_ocr.prepare_demo import fixed_forward
from hardware.de10_lite_ocr_compare.export_compare import quantize
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5
from train_synthetic_car_compare import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001"))
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001/int8_real_test.json"))
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--per-layout", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    plates = prepared_cars(args.prepared, "test", args.per_layout, args.seed, set())
    pixels = np.stack([crop for plate in plates for crop in plate["crops"]]).astype(np.float32)
    float_x = torch.from_numpy(pixels / 127.5 - 1.0).unsqueeze(1)
    quant_x = torch.from_numpy(np.rint((pixels / 127.5 - 1.0) * 127.0).astype(np.int8).astype(np.float32)).unsqueeze(1)
    lut = [int(np.rint(127.0 * math.tanh(index / 32.0))) for index in range(-128, 129)]
    summary = {"source": "identity-disjoint real prepared test glyph crops, label-count-assisted preprocessing",
               "plates": len(plates), "characters": len(pixels), "models": {}}
    for name, cls in (("dense", LeNet5), ("pruned", LeNet5Conv2Pruned)):
        path = args.models / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        specs, weights, biases, shifts = quantize(checkpoint["model"])
        fp32, fixed = [], []
        with torch.inference_mode():
            for start in range(0, len(pixels), args.batch):
                stop = start + args.batch
                fp32.extend(model(float_x[start:stop]).argmax(dim=1).tolist())
                fixed.extend(fixed_forward(quant_x[start:stop], specs, weights, biases, lut).argmax(dim=1).tolist())
        cursor = 0
        by_layout = {layout: {"plates": 0, "fp32_exact": 0, "int8_exact": 0,
                              "characters": 0, "fp32_correct": 0, "int8_correct": 0, "agreement": 0}
                     for layout in ("one_row_car", "two_row_car")}
        for plate in plates:
            truth = plate["record"]["label"]
            layout = plate["record"]["layout"]
            a = "".join(CLASS_NAMES[index] for index in fp32[cursor:cursor + 8])
            b = "".join(CLASS_NAMES[index] for index in fixed[cursor:cursor + 8])
            group = by_layout[layout]
            group["plates"] += 1
            group["fp32_exact"] += int(a == truth)
            group["int8_exact"] += int(b == truth)
            group["characters"] += 8
            group["fp32_correct"] += sum(x == y for x, y in zip(a, truth))
            group["int8_correct"] += sum(x == y for x, y in zip(b, truth))
            group["agreement"] += sum(x == y for x, y in zip(a, b))
            cursor += 8
        total = {key: sum(group[key] for group in by_layout.values()) for key in next(iter(by_layout.values()))}
        summary["models"][name] = {"checkpoint_sha256": sha(path), "weight_bytes": len(weights),
                                   "bias_count": len(biases), "shifts": shifts,
                                   "total": total, "by_layout": by_layout}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
