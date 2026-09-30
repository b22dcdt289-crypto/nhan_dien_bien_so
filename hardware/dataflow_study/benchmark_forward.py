"""Interleaved CPU forward benchmark for the two new OCR checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware.de10_lite_ocr_compare.train_compare import bench_forward, digest
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001/cpu_interleaved.json"))
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = {"dense": LeNet5(len(CLASS_NAMES)), "pruned": LeNet5Conv2Pruned(len(CLASS_NAMES))}
    digests = {}
    for name, model in models.items():
        path = args.models / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["classes"] != CLASS_NAMES:
            raise RuntimeError(f"Incompatible classes in {path}")
        model.load_state_dict(checkpoint["model"])
        digests[name] = digest(path)
    measurements = bench_forward(models, args.threads, args.repeats)
    result = {"device": "CPU", "torch_version": torch.__version__, "threads": args.threads,
              "input_shape_per_glyph": [1, 32, 32], "randomized_interleaved_order": True,
              "model_sha256": digests, "latency": measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
