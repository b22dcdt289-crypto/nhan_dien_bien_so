from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from train_lenet5 import CLASS_NAMES, LeNet5, run_epoch
from train_lenet5_folder import FolderChars


def apply_global_pruning(model: nn.Module, amount: float):
    parameters = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    return parameters


def sparsity(model: nn.Module):
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()
            total += weight.numel()
            zeros += int((weight == 0).sum())
    return zeros / total, zeros, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/VNLP_chars_full"))
    parser.add_argument("--init", type=Path, default=Path("artifacts/lenet5_vnlp_37k_full.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_vnlp_37k_pruned25.pt"))
    parser.add_argument("--prune", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.prune < 1:
        raise ValueError("--prune must be between 0 and 1")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = FolderChars(args.data / "train", augment=True)
    val = FolderChars(args.data / "val", augment=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    if args.init.exists():
        checkpoint = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded={args.init}", flush=True)
    apply_global_pruning(model, args.prune)
    current, zeros, total = sparsity(model)
    print(f"device={device} train_chars={len(train)} val_chars={len(val)} sparsity={current:.4f} zeros={zeros}/{total}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        vl, va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        if va >= best:
            best = va
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Save a clean copy while keeping masks active in the training model.
            saved_model = copy.deepcopy(model).cpu()
            for module in saved_model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.remove(module, "weight")
            torch.save({"model": saved_model.state_dict(), "classes": CLASS_NAMES, "val_acc": va, "pruning": args.prune, "sparsity": sparsity(saved_model)[0]}, args.output)
    print(f"saved={args.output} best_val_acc={best:.4f} final_sparsity={sparsity(model)[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
