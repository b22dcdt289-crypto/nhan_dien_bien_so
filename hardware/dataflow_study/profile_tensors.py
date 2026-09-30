"""Aggregate weight, bias and activation distributions for new checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetune_mixed_car_compare import PREPARED, prepared_cars
from hardware.de10_lite_ocr_compare.export_compare import LAYERS, quantize
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5


def stats(tensor: torch.Tensor) -> dict:
    array = tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return {"count": int(array.size), "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std()),
            "p01": float(np.quantile(array, 0.01)), "p99": float(np.quantile(array, 0.99)),
            "exact_zero_fraction": float(np.mean(array == 0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001"))
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001/tensor_profile.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plates = prepared_cars(args.prepared, "test", 2, 20261002, set())
    crops = np.stack([crop for plate in plates for crop in plate["crops"][:2]]).astype(np.float32)
    inputs = torch.from_numpy(crops / 127.5 - 1.0).unsqueeze(1)
    summary = {"activation_source": "8 held-out prepared real glyph crops (4 plates, 2 per layout)",
               "activation_sample_count": len(crops), "not_hardware_measured": True, "models": {}}
    for name, cls in (("dense", LeNet5), ("pruned", LeNet5Conv2Pruned)):
        checkpoint = torch.load(args.models / f"{name}.pt", map_location="cpu", weights_only=False)
        model = cls(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        weights, biases, activations = {}, {}, {}
        hooks = []
        for layer, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights[layer] = stats(module.weight)
                biases[layer] = stats(module.bias)
                hooks.append(module.register_forward_hook(
                    lambda _module, _input, output, key=layer: activations.__setitem__(key, stats(output))))
        with torch.inference_mode():
            model(inputs)
        for hook in hooks:
            hook.remove()
        specs, w_int8, _b_int32, shifts = quantize(checkpoint["model"])
        distribution = {"minimum": min(w_int8), "maximum": max(w_int8),
                        "exact_zero_fraction": sum(value == 0 for value in w_int8) / len(w_int8),
                        "saturated_127_count": sum(abs(value) == 127 for value in w_int8)}
        summary["models"][name] = {"layer_weights_fp32": weights, "layer_biases_fp32": biases,
                                   "layer_output_activations_fp32": activations,
                                   "exported_weights_int8": distribution,
                                   "weight_exponents_by_layer": {layer: specs[layer]["weight_exponent"] for layer in LAYERS},
                                   "tanh_shifts_by_layer": dict(zip(LAYERS, shifts))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(crops),
                      "int8_distributions": {key: value["exported_weights_int8"] for key, value in summary["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
