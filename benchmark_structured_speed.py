from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from train_independent_structured import LeNet5Structured50
from train_lenet5 import LeNet5


DEFAULT_DENSE = Path("artifacts/independent_lenet5_dense_enhanced_train1.pt")
DEFAULT_STRUCTURED = Path("artifacts/independent_lenet5_structured50_enhanced_train1.pt")


def load_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError(f"Checkpoint has no saved class order: {checkpoint_path}")
    return checkpoint, list(classes)


def lenet_macs(channels: tuple[int, int, int, int], class_count: int):
    c1, c2, h1, h2 = channels
    return c1 * 25 * 28 * 28 + c1 * c2 * 25 * 10 * 10 + c2 * 25 * h1 + h1 * h2 + h2 * class_count


def load_model(checkpoint_path: Path, model_type: str, class_count: int, checkpoint):
    model = LeNet5(class_count) if model_type == "dense" else LeNet5Structured50(class_count)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def summarize(samples_ns: list[int], batch_size: int):
    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    median_ms = float(np.median(samples_ms))
    return {
        "iterations": len(samples_ns),
        "median_ms_per_forward": median_ms,
        "mean_ms_per_forward": float(np.mean(samples_ms)),
        "std_ms_per_forward": float(np.std(samples_ms, ddof=1)) if len(samples_ms) > 1 else 0.0,
        "p90_ms_per_forward": float(np.percentile(samples_ms, 90)),
        "p95_ms_per_forward": float(np.percentile(samples_ms, 95)),
        "min_ms_per_forward": float(np.min(samples_ms)),
        "max_ms_per_forward": float(np.max(samples_ms)),
        "characters_per_second_from_median": batch_size / (median_ms / 1000.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Matched CPU forward-latency benchmark for dense and structured LeNet-5.")
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--structured", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/structured_speed_benchmark.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/structured_speed_benchmark_raw.csv"))
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 100 or args.warmup < 1 or args.threads < 1:
        raise ValueError("Use at least 100 measured iterations, one warm-up, and one CPU thread.")
    if not args.dense.is_file() or not args.structured.is_file():
        raise FileNotFoundError("Both dense and structured checkpoints must exist.")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260926)
    dense_checkpoint, dense_classes = load_checkpoint(args.dense)
    structured_checkpoint, structured_classes = load_checkpoint(args.structured)
    if dense_classes != structured_classes:
        raise ValueError("Dense and structured checkpoints have different class orders; refusing a mismatched comparison.")
    class_count = len(dense_classes)
    dense, _ = load_model(args.dense, "dense", class_count, dense_checkpoint)
    structured, _ = load_model(args.structured, "structured", class_count, structured_checkpoint)
    models = {"dense": dense, "structured_50": structured}
    batches = (1, 7, 8, 9, 10)
    raw_rows = []
    results = {}

    with torch.inference_mode():
        for batch_size in batches:
            inputs = torch.randn(batch_size, 1, 32, 32)
            for model in models.values():
                for _ in range(args.warmup):
                    model(inputs)

            samples = {name: [] for name in models}
            for iteration in range(args.repeats):
                order = ("dense", "structured_50") if iteration % 2 == 0 else ("structured_50", "dense")
                for name in order:
                    start_ns = time.perf_counter_ns()
                    models[name](inputs)
                    elapsed_ns = time.perf_counter_ns() - start_ns
                    samples[name].append(elapsed_ns)
                    raw_rows.append({
                        "batch_characters": batch_size,
                        "iteration": iteration,
                        "model": name,
                        "latency_ns": elapsed_ns,
                    })

            dense_stats = summarize(samples["dense"], batch_size)
            structured_stats = summarize(samples["structured_50"], batch_size)
            dense_median = dense_stats["median_ms_per_forward"]
            structured_median = structured_stats["median_ms_per_forward"]
            results[str(batch_size)] = {
                "dense": dense_stats,
                "structured_50": structured_stats,
                "speedup_dense_over_structured_by_median": dense_median / structured_median,
                "latency_reduction_percent_by_median": 100.0 * (1.0 - structured_median / dense_median),
            }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("batch_characters", "iteration", "model", "latency_ns"))
        writer.writeheader()
        writer.writerows(raw_rows)

    dense_macs = lenet_macs((6, 16, 120, 84), class_count)
    structured_macs = lenet_macs(LeNet5Structured50.channels, class_count)
    report = {
        "benchmark": "PyTorch CPU model-forward latency only; excludes image I/O, rectification, segmentation, and formatting",
        "device": "CPU",
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": args.threads,
        "warmup_forwards_per_model_and_batch": args.warmup,
        "measured_forwards_per_model_and_batch": args.repeats,
        "input_shape_per_character": [1, 32, 32],
        "class_count": class_count,
        "class_order": dense_classes,
        "batch_sizes_characters": list(batches),
        "dense_model": {
            "checkpoint": args.dense.as_posix(),
            "parameters": sum(t.numel() for t in dense.state_dict().values()),
            "macs_per_character": dense_macs,
            "validation_character_accuracy": dense_checkpoint.get("val_acc"),
        },
        "structured_model": {
            "checkpoint": args.structured.as_posix(),
            "channels": list(LeNet5Structured50.channels),
            "parameters": sum(t.numel() for t in structured.state_dict().values()),
            "macs_per_character": structured_macs,
            "validation_character_accuracy": structured_checkpoint.get("val_acc"),
        },
        "mac_reduction_percent": 100.0 * (1.0 - structured_macs / dense_macs),
        "results_by_batch_size": results,
        "raw_measurements_csv": args.output_csv.as_posix(),
        "limitations": "This host CPU benchmark is not DE10-Lite/FPGA latency. A structured model is physically smaller, but board speedup depends on HDL mapping, parallelism, memory bandwidth, clock rate, and Quartus timing closure.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
