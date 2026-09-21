from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_lenet5 import CLASS_NAMES, CharacterCropDataset, run_epoch
from train_lenet5_lite import LeNet5Lite, macs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/OCR/OCR"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_lite_ocr_int8_ready.pt"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = CharacterCropDataset(args.data, "train", augment=True)
    val = CharacterCropDataset(args.data, "val", augment=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5Lite(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    print(f"device={device} train_crops={len(train)} val_crops={len(val)} macs={macs(model)}", flush=True)
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
