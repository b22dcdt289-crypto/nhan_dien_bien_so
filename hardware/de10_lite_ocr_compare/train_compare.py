"""Matched dense vs Conv2-channel-pruned LeNet-5 training on prepared train(1).

This compares ten dataset passes per branch. Epochs 1-5 are one shared dense
checkpoint; epochs 6-10 continue both dense and physically compact Conv2=8
models with identical data order, augmentations, loss, optimizer and schedule.
Streaming is a hardware input mode and is not a training technique.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned, compact_teacher_to_student
from train_independent_structured import FolderChars
from train_lenet5 import CLASS_NAMES, LeNet5, run_epoch
from train_matched_channel_pruning import evaluate_plates, model_macs, read_plate_validation_batches


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProcessSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_rss = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.1):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> "ProcessSampler":
        self.peak_rss = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def save_model(path: Path, model: nn.Module, name: str, epoch: int, val_acc: float, seed: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    channels = [6, 16, 120, 84] if name == "dense" else [6, 8, 120, 84]
    torch.save({
        "model": model.state_dict(), "classes": CLASS_NAMES,
        "arch": name, "channels": channels, "epoch": epoch,
        "val_acc": val_acc, "seed": seed,
        "macs_per_character": model_macs(tuple(channels), len(CLASS_NAMES)),
        "data_root": "data/independent_chars_train1_structured_channel30_v1",
    }, path)


def bench_forward(models: dict[str, nn.Module], threads: int, repeats: int) -> dict:
    torch.set_num_threads(threads)
    rng = random.Random(1741)
    results: dict[str, dict] = {}
    for batch in (1, 8):
        tensor = torch.randn(batch, 1, 32, 32)
        for model in models.values():
            model.eval()
            with torch.inference_mode():
                for _ in range(100):
                    model(tensor)
        times = {name: [] for name in models}
        with torch.inference_mode():
            for _ in range(repeats):
                order = list(models)
                rng.shuffle(order)
                for name in order:
                    start = time.perf_counter_ns()
                    models[name](tensor)
                    times[name].append((time.perf_counter_ns() - start) / 1e6)
        for name, values in times.items():
            ordered = sorted(values)
            results.setdefault(name, {})[str(batch)] = {
                "repeats": repeats,
                "median_ms_per_batch": statistics.median(values),
                "p95_ms_per_batch": ordered[int(0.95 * (len(ordered) - 1))],
                "min_ms_per_batch": ordered[0],
                "max_ms_per_batch": ordered[-1],
                "threads": threads,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/independent_chars_train1_structured_channel30_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/compare_conv2_stream_20260929")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prune-after", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bench-repeats", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.prune_after < args.epochs:
        raise ValueError("prune-after must lie between 1 and epochs-1")
    args.out.mkdir(parents=True, exist_ok=True)
    out_paths = [args.out / name for name in (
        "dense.pt", "conv2_pruned.pt", "history.csv", "training_metrics.json"
    )]
    existing = [str(path) for path in out_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite experiment outputs: " + ", ".join(existing))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    loss_fn = nn.CrossEntropyLoss()
    process = psutil.Process()
    with ProcessSampler() as sampler:
        experiment_start = time.perf_counter()
        load_start = time.perf_counter()
        train_set = FolderChars(args.data, "train", augment=True)
        val_set = FolderChars(args.data, "val", augment=False)
        data_load_seconds = time.perf_counter() - load_start
        if not train_set or not val_set:
            raise RuntimeError("Training or validation split is empty")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        dense = LeNet5(len(CLASS_NAMES)).to(device)
        dense_opt = torch.optim.AdamW(dense.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pruned = None
        pruned_opt = None
        selected_channels = None
        best = {"dense": (-1.0, 0), "conv2_pruned": (-1.0, 0)}
        history: list[dict] = []
        for epoch in range(1, args.epochs + 1):
            model_order = ["dense"] if epoch <= args.prune_after else ["dense", "conv2_pruned"]
            for name in model_order:
                model = dense if name == "dense" else pruned
                optimizer = dense_opt if name == "dense" else pruned_opt
                assert model is not None and optimizer is not None
                # Identical shuffle and random rotation stream for each continuation branch.
                random.seed(args.seed + 1000 * epoch)
                torch.manual_seed(args.seed + 1000 * epoch)
                cpu_before = process.cpu_times()
                start = time.perf_counter()
                train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
                train_seconds = time.perf_counter() - start
                cpu_after = process.cpu_times()
                start = time.perf_counter()
                val_loss, val_acc = run_epoch(model, val_loader, loss_fn, None, device, False)
                val_seconds = time.perf_counter() - start
                record = {
                    "epoch": epoch, "branch": name,
                    "phase": "shared_pretraining" if epoch <= args.prune_after else "continuation",
                    "train_loss": train_loss, "train_character_accuracy": train_acc,
                    "val_loss": val_loss, "val_character_accuracy": val_acc,
                    "train_seconds": train_seconds, "val_seconds": val_seconds,
                    "train_images_per_second": len(train_set) / train_seconds,
                    "train_cpu_seconds": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
                    "process_peak_rss_bytes_so_far": sampler.peak_rss,
                }
                history.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if val_acc > best[name][0] and (name == "dense" and epoch > args.prune_after or name == "conv2_pruned"):
                    best[name] = (val_acc, epoch)
                    target = args.out / ("dense.pt" if name == "dense" else "conv2_pruned.pt")
                    # Best checkpoint is replaced only by a demonstrably better epoch.
                    if target.exists():
                        target.unlink()
                    save_model(target, model, name, epoch, val_acc, args.seed)

            if epoch == args.prune_after:
                pruned = LeNet5Conv2Pruned(len(CLASS_NAMES)).to(device)
                selected_channels = compact_teacher_to_student(dense, pruned)
                # Both branches have all remaining layers trainable: pruning is the only model difference.
                pruned.features[0].weight.requires_grad_(True)
                pruned.features[0].bias.requires_grad_(True)
                # Reset continuation optimizer state in both branches.
                dense_opt = torch.optim.AdamW(dense.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                pruned_opt = torch.optim.AdamW(pruned.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                _, post_prune_val_acc = run_epoch(pruned, val_loader, loss_fn, None, device, False)
                print(json.dumps({"event": "prune", "epoch": epoch,
                                  "conv2_channels_kept": selected_channels,
                                  "immediate_val_accuracy": post_prune_val_acc}), flush=True)

        # Evaluate best checkpoints, not whichever model happens to be last in RAM.
        for name, model in (("dense", dense), ("conv2_pruned", pruned)):
            checkpoint = torch.load(args.out / (name + ".pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        val_records = [item for item in manifest if item["split"] == "val"]
        plate_samples = read_plate_validation_batches(args.data, val_records)
        validation = {
            "dense": evaluate_plates(dense, plate_samples, device),
            "conv2_pruned": evaluate_plates(pruned, plate_samples, device),
        }
        cpu_benchmark = bench_forward({"dense": dense, "conv2_pruned": pruned}, args.threads, args.bench_repeats)
        elapsed_seconds = time.perf_counter() - experiment_start

    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    shared = [item for item in history if item["phase"] == "shared_pretraining"]
    dense_extra = [item for item in history if item["phase"] == "continuation" and item["branch"] == "dense"]
    pruned_extra = [item for item in history if item["branch"] == "conv2_pruned"]
    def duration(items: list[dict]) -> float:
        return sum(item["train_seconds"] + item["val_seconds"] for item in items)
    dense_channels = (6, 16, 120, 84)
    pruned_channels = (6, 8, 120, 84)
    metrics = {
        "experiment": "Matched ten-pass dense vs structured Conv2 output-channel pruning; streaming is separately evaluated on FPGA",
        "source": "data/OCR/OCR/images/train(1)/detection via prepared character crops",
        "data_root": str(args.data), "manifest_sha256": digest(args.data / "manifest.json"),
        "split": {"train_character_crops": len(train_set), "validation_character_crops": len(val_set),
                  "validation_plates": len(val_records), "method": "frame-level, possible repeated plate identities"},
        "training": {"seed": args.seed, "epochs_per_branch": args.epochs,
                     "shared_dense_pretraining_epochs": args.prune_after,
                     "continuation_epochs_per_branch": args.epochs - args.prune_after,
                     "loss": "cross_entropy", "optimizer": "AdamW", "lr": args.lr,
                     "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                     "augmentation": "55% chance random rotation uniformly [-6,6] degrees on train crops only",
                     "device": "CPU", "torch_version": torch.__version__, "torch_threads": args.threads,
                     "data_load_seconds": data_load_seconds, "experiment_wall_seconds": elapsed_seconds,
                     "shared_train_plus_val_seconds": duration(shared),
                     "dense_branch_train_plus_val_seconds": duration(shared) + duration(dense_extra),
                     "pruned_branch_train_plus_val_seconds": duration(shared) + duration(pruned_extra),
                     "process_peak_rss_bytes": sampler.peak_rss,
                     "post_prune_val_character_accuracy": post_prune_val_acc,
                     "selected_conv2_channel_indices": selected_channels},
        "models": {
            name: {"checkpoint": str(args.out / (name + ".pt")),
                   "checkpoint_sha256": digest(args.out / (name + ".pt")),
                   "channels": list(channels),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "macs_per_character": model_macs(channels, len(CLASS_NAMES)),
                   "best_epoch": best[name][1], "best_val_crop_character_accuracy": best[name][0],
                   "validation_plate_metrics": validation[name], "cpu_forward": cpu_benchmark[name]}
            for name, model, channels in (("dense", dense, dense_channels), ("conv2_pruned", pruned, pruned_channels))
        },
        "evaluation_scope": "Pre-cropped reference characters with filename-guided segmentation; not full-frame or live camera recognition",
    }
    (args.out / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(args.out / "training_metrics.json"),
                      "dense": {"char": validation["dense"]["character_accuracy"],
                                "plate": validation["dense"]["exact_plate_accuracy"]},
                      "pruned": {"char": validation["conv2_pruned"]["character_accuracy"],
                                 "plate": validation["conv2_pruned"]["exact_plate_accuracy"]}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
