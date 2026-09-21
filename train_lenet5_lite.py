from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_lenet5 import CLASS_NAMES, run_epoch
from train_lenet5_folder import FolderChars


class LeNet5Lite(nn.Module):
    """Hardware-oriented LeNet: fewer channels/FC units, 36 character classes."""

    def __init__(self, num_classes: int = 36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(4, 12, 5), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(12 * 5 * 5, 64), nn.ReLU(),
            nn.Linear(64, 48), nn.ReLU(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def macs(model: nn.Module):
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total += module.out_channels * module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 28 * 28 if module.in_channels == 1 else module.out_channels * module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 10 * 10
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/VNLP_chars_corrected"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_lite_int8_ready.pt"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = FolderChars(args.data / "train", augment=True)
    val = FolderChars(args.data / "val", augment=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5Lite(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    print(f"device={device} train_chars={len(train)} val_chars={len(val)} macs={macs(model)}", flush=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        vl, va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        scheduler.step()
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        if va >= best:
            best = va
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "val_acc": va, "macs": macs(model), "quantization": "INT8-ready"}, args.output)
    print(f"saved={args.output} best_val_acc={best:.4f} macs={macs(model)}", flush=True)


if __name__ == "__main__":
    main()
