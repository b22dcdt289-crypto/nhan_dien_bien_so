from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_lenet5 import CLASS_NAMES, CharacterCropDataset, LeNet5, run_epoch


class LeNet5Structured(nn.Module):
    """Physically smaller LeNet-5: channel/neuron pruning is baked into shapes."""

    def __init__(self, num_classes=36, c1=4, c2=12, h1=90, h2=63):
        super().__init__()
        self.channels = (c1, c2, h1, h2)
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, 5), nn.Tanh(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c2 * 5 * 5, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def macs(self):
        c1, c2, h1, h2 = self.channels
        return 1 * c1 * 25 * 28 * 28 + c1 * c2 * 25 * 10 * 10 + c2 * 25 * h1 + h1 * h2 + h2 * 36


def top_indices(weight, count, dim=0):
    score = weight.detach().abs().sum(dim=tuple(i for i in range(weight.ndim) if i != dim))
    return torch.topk(score, count).indices.sort().values


def transfer(dense: LeNet5, small: LeNet5Structured):
    c1, c2, h1, h2 = small.channels
    conv1, conv2 = dense.features[0], dense.features[3]
    fc1, fc2, fc3 = dense.classifier[0], dense.classifier[2], dense.classifier[4]
    i1 = top_indices(conv1.weight, c1)
    i2 = top_indices(conv2.weight, c2)
    ih1 = top_indices(fc1.weight[:, i2.repeat_interleave(25)], h1)
    ih2 = top_indices(fc2.weight[:, ih1], h2)
    with torch.no_grad():
        small.features[0].weight.copy_(conv1.weight[i1])
        small.features[0].bias.copy_(conv1.bias[i1])
        small.features[3].weight.copy_(conv2.weight[i2][:, i1])
        small.features[3].bias.copy_(conv2.bias[i2])
        input_cols = i2.repeat_interleave(25)
        small.classifier[0].weight.copy_(fc1.weight[ih1][:, input_cols])
        small.classifier[0].bias.copy_(fc1.bias[ih1])
        small.classifier[2].weight.copy_(fc2.weight[ih2][:, ih1])
        small.classifier[2].bias.copy_(fc2.bias[ih2])
        small.classifier[4].weight.copy_(fc3.weight[:, ih2])
        small.classifier[4].bias.copy_(fc3.bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/OCR/OCR'))
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = CharacterCropDataset(args.data, 'train', augment=True)
    val = CharacterCropDataset(args.data, 'val', augment=False)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=0)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model = LeNet5Structured(len(CLASS_NAMES)).to(device)
    if ckpt.get('arch') == 'LeNet5Structured':
        model.load_state_dict(ckpt['model'])
        print(f'resumed={args.init}', flush=True)
    else:
        dense = LeNet5(len(CLASS_NAMES)).to(device)
        dense.load_state_dict(ckpt['model'])
        transfer(dense, model)
    print(f'device={device} train={len(train)} val={len(val)} channels={model.channels} macs={model.macs()}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, loss_fn, opt, device, True)
        vl, va = run_epoch(model, val_loader, loss_fn, opt, device, False)
        print(f'epoch {epoch:02d}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}', flush=True)
        if va >= best:
            best = va
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'classes': CLASS_NAMES,
                        'val_acc': va, 'arch': 'LeNet5Structured',
                        'channels': model.channels, 'macs': model.macs()}, args.output)
    print(f'saved={args.output} best_val_acc={best:.4f} macs={model.macs()}', flush=True)


if __name__ == '__main__':
    main()
