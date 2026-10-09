"""Functional Conv1/Conv2 loop-order verification for WS, OS and RS.

This is a CPU reference, not an FPGA implementation or a hardware speed
benchmark. It checks that changing the convolution schedule preserves LeNet-5
predictions before pursuing three distinct RTL PE arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetune_mixed_car_compare import PREPARED, prepared_cars
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5
from train_three_layout_compare import LAYOUTS, generated_plates
from lenet5_k3 import LeNet5K3, LeNet5K3Pruned


def output_stationary(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Keep one output psum local until all input channels and taps are done."""
    channels, height, width = x.shape
    outputs, inputs, kernel, _ = w.shape
    assert inputs == channels
    result = np.empty((outputs, height - kernel + 1, width - kernel + 1), dtype=np.float32)
    for oc in range(outputs):
        for oy in range(result.shape[1]):
            for ox in range(result.shape[2]):
                patch = x[:, oy:oy + kernel, ox:ox + kernel]
                result[oc, oy, ox] = np.float32(b[oc] + np.sum(patch * w[oc], dtype=np.float32))
    return result


def weight_stationary(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hold each filter tap while sweeping its use over output positions."""
    outputs, inputs, kernel, _ = w.shape
    result = np.broadcast_to(b[:, None, None],
                             (outputs, x.shape[1] - kernel + 1,
                              x.shape[2] - kernel + 1)).copy()
    height, width = result.shape[1:]
    for oc in range(outputs):
        for ic in range(inputs):
            for ky in range(kernel):
                for kx in range(kernel):
                    result[oc] += w[oc, ic, ky, kx] * x[ic, ky:ky + height, kx:kx + width]
    return result


def row_stationary(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reuse a 5-tap filter row and sliding input row across output columns."""
    outputs, inputs, kernel, _ = w.shape
    height, width = x.shape[1] - kernel + 1, x.shape[2] - kernel + 1
    result = np.broadcast_to(b[:, None, None], (outputs, height, width)).copy()
    for oc in range(outputs):
        for ic in range(inputs):
            for ky in range(kernel):
                filter_row = w[oc, ic, ky]
                for oy in range(height):
                    input_row = x[ic, oy + ky]
                    row_psum = np.zeros(width, dtype=np.float32)
                    for kx in range(kernel):
                        row_psum += filter_row[kx] * input_row[kx:kx + width]
                    result[oc, oy] += row_psum
    return result


SCHEDULES = {"output_stationary": output_stationary,
             "weight_stationary": weight_stationary,
             "row_stationary": row_stationary}


def reference_forward(model: torch.nn.Module, image: torch.Tensor, schedule) -> tuple[list[np.ndarray], torch.Tensor]:
    with torch.inference_mode():
        conv1 = model.features[0]
        first = schedule(image.numpy(), conv1.weight.numpy(), conv1.bias.numpy())
        pool1 = model.features[2](torch.tanh(torch.from_numpy(first).unsqueeze(0)))
        conv2 = model.features[3]
        second = schedule(pool1[0].numpy(), conv2.weight.numpy(), conv2.bias.numpy())
        pool2 = model.features[5](torch.tanh(torch.from_numpy(second).unsqueeze(0)))
        logits = model.classifier(pool2.flatten(1))
    return [first, second], logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001"))
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001/dataflow_functional.json"))
    parser.add_argument("--three-layout", action="store_true", help="Use six held-out generated glyphs across car/motorcycle layouts")
    parser.add_argument("--kernel-size", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    if args.three_layout:
        plates_by_split = generated_plates(Path("data/car_frontal_synthetic_2x1000_v1"),
                                           Path("data/motorcycle_frontal_synthetic_1000_v2"), 20261009)
        plates = [next(plate for plate in plates_by_split["test"] if plate["record"]["layout"] == layout)
                  for layout in LAYOUTS]
        source = "6 held-out generated glyph crops, 2 per car-one-row/car-two-row/motorcycle-two-row"
    else:
        plates = prepared_cars(args.prepared, "test", 1, 20261002, set())
        source = "4 held-out prepared real glyph crops, two one-row and two two-row"
    samples = [torch.from_numpy(crop.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
               for plate in plates for crop in plate["crops"][:2]]
    result = {"source": source,
              "input_count": len(samples), "hardware_implementation": False, "models": {}}
    model_classes = (("dense", LeNet5K3), ("pruned", LeNet5K3Pruned)) if args.kernel_size == 3 else (
        ("dense", LeNet5), ("pruned", LeNet5Conv2Pruned))
    for name, cls in model_classes:
        checkpoint = torch.load(args.models / f"{name}.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("kernel_size", 5) != args.kernel_size:
            raise RuntimeError("Checkpoint and requested convolution kernel differ")
        model = cls(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        schedules = {key: {"conv1_max_abs_error": 0.0, "conv2_max_abs_error": 0.0,
                           "logit_max_abs_error": 0.0, "argmax_agreement": 0}
                     for key in SCHEDULES}
        for image in samples:
            with torch.inference_mode():
                expected_conv1 = model.features[0](image.unsqueeze(0))
                expected_conv2 = model.features[3](model.features[2](torch.tanh(expected_conv1)))
                expected = model(image.unsqueeze(0))
            for key, schedule in SCHEDULES.items():
                convs, logits = reference_forward(model, image, schedule)
                row = schedules[key]
                row["conv1_max_abs_error"] = max(row["conv1_max_abs_error"],
                    float(np.max(np.abs(convs[0] - expected_conv1[0].numpy()))))
                row["conv2_max_abs_error"] = max(row["conv2_max_abs_error"],
                    float(np.max(np.abs(convs[1] - expected_conv2[0].numpy()))))
                row["logit_max_abs_error"] = max(row["logit_max_abs_error"],
                    float(torch.max(torch.abs(logits - expected))))
                row["argmax_agreement"] += int(logits.argmax().item() == expected.argmax().item())
        result["models"][name] = schedules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
