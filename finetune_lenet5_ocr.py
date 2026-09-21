from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_lenet5 import CLASS_NAMES, CharacterCropDataset, LeNet5, run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/OCR/OCR"))
    parser.add_argument("--init", type=Path, default=Path("artifacts/lenet5_chars.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_ocr_97target.pt"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = CharacterCropDataset(args.data, "train", augment=True)
    val = CharacterCropDataset(args.data, "val", augment=False)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=0)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    best = float(checkpoint.get("val_acc", 0.0))
    print(f"device={device} train_crops={len(train)} val_crops={len(val)} start_val_acc={best:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        vl, va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        if va >= best:
            best = va
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "val_acc": va}, args.output)
    print(f"saved={args.output} best_val_acc={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
