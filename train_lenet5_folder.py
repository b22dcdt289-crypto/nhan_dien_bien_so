from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_lenet5 import CLASS_NAMES, LeNet5, run_epoch


class FolderChars(Dataset):
    def __init__(self, root: Path, augment: bool = False):
        self.items = []
        self.augment = augment
        for label, char in enumerate(CLASS_NAMES):
            self.items.extend((p, label) for p in (root / char).glob("*.png"))
        self.cache = []
        for path, label in self.items:
            with Image.open(path) as image:
                self.cache.append((np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pixels, label = self.cache[index]
        image = Image.fromarray(pixels, mode="L")
        if self.augment and random.random() < 0.5:
            image = image.rotate(random.uniform(-4, 4), resample=Image.Resampling.BILINEAR, fillcolor=0)
        data = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(data).unsqueeze(0), label


class LazyFolderChars(Dataset):
    """Folder dataset that avoids caching a very large crop set in RAM."""

    def __init__(self, root: Path, augment: bool = False):
        self.items = []
        self.augment = augment
        for label, char in enumerate(CLASS_NAMES):
            self.items.extend((p, label) for p in (root / char).glob("*.png"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            if self.augment and random.random() < 0.5:
                image = image.rotate(random.uniform(-4, 4), resample=Image.Resampling.BILINEAR, fillcolor=0)
            data = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(data).unsqueeze(0), label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/VNLP_chars"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_vnlp_37k.pt"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = FolderChars(args.data / "train", augment=True)
    val = FolderChars(args.data / "val", augment=False)
    if not train or not val:
        raise RuntimeError("Character folders are empty. Run prepare_vnlp_chars.py first.")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device} train_chars={len(train)} val_chars={len(val)}", flush=True)
    best = 0.0
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
