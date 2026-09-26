from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from structured_prune_lenet5 import LeNet5Structured
from train_lenet5 import CLASS_NAMES, require_compatible_classes, run_epoch
from train_lenet5_folder import LazyFolderChars


def main():
    p = argparse.ArgumentParser(description='Fine-tune the structured LeNet on a folder dataset')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = LazyFolderChars(args.data / 'train', augment=True)
    val = LazyFolderChars(args.data / 'val', augment=False)
    if not train or not val:
        raise RuntimeError(f'Character folders are empty: {args.data}')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5Structured(len(CLASS_NAMES)).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    require_compatible_classes(ckpt, args.init)
    model.load_state_dict(ckpt['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    print(f'device={device} train_chars={len(train)} val_chars={len(val)} data={args.data}', flush=True)
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        vl, va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(f'epoch {epoch:02d}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}', flush=True)
        if va >= best:
            best = va
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'classes': CLASS_NAMES,
                        'val_acc': va, 'arch': 'LeNet5Structured',
                        'channels': model.channels, 'macs': model.macs(),
                        'data': str(args.data)}, args.output)
    print(f'saved={args.output} best_val_acc={best:.4f} macs={model.macs()}', flush=True)


if __name__ == '__main__':
    main()
