from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Standard domestic plate serial letters from Circular 79/2024/TT-BCA:
# A-H, K-N, P, S-V, X-Z. Together with digits this is a 30-class OCR alphabet.
CLASS_NAMES = list("0123456789ABCDEFGHKLMNPSTUVXYZ")
SOURCE_CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_INDEX = {character: index for index, character in enumerate(CLASS_NAMES)}


def require_compatible_classes(checkpoint: dict, checkpoint_path: Path | str = "checkpoint") -> None:
    stored_classes = checkpoint.get("classes")
    if stored_classes != CLASS_NAMES:
        stored_count = len(stored_classes) if stored_classes is not None else "unknown"
        raise ValueError(
            f"{checkpoint_path} has {stored_count} classes and a different class order; "
            f"this model requires {len(CLASS_NAMES)} classes ({''.join(CLASS_NAMES)}). "
            "Retrain or explicitly migrate the checkpoint before using it."
        )


class CharacterCropDataset(Dataset):
    """Reads plate images + YOLO character boxes and returns 32x32 crops."""

    def __init__(self, root: Path, split: str, augment: bool = False, max_items: int | None = None):
        self.image_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        self.items: list[tuple[Path, int, tuple[float, float, float, float]]] = []
        image_files = sorted(self.image_dir.glob("*.jpg"))
        for image_path in image_files:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                source_cls = int(float(parts[0]))
                if not 0 <= source_cls < len(SOURCE_CLASS_NAMES):
                    continue
                character = SOURCE_CLASS_NAMES[source_cls]
                # Source YOLO labels use the old digit+ A-Z (36-class) ordering.
                # Drop symbols outside the current plate vocabulary and remap
                # every retained symbol to its new contiguous class index.
                if character not in CLASS_TO_INDEX:
                    continue
                cls = CLASS_TO_INDEX[character]
                box = tuple(float(x) for x in parts[1:5])
                self.items.append((image_path, cls, box))

        if max_items is not None:
            random.Random(42).shuffle(self.items)
            self.items = self.items[:max_items]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label, (xc, yc, w, h) = self.items[index]
        with Image.open(image_path) as source:
            image = source.convert("L")
            width, height = image.size
            left = max(0, int((xc - w / 2) * width))
            top = max(0, int((yc - h / 2) * height))
            right = min(width, max(left + 1, int((xc + w / 2) * width)))
            bottom = min(height, max(top + 1, int((yc + h / 2) * height)))
            crop = image.crop((left, top, right, bottom))
            crop = ImageOps.pad(crop, (32, 32), color=0, centering=(0.5, 0.5))
            if self.augment and random.random() < 0.7:
                angle = random.uniform(-4.0, 4.0)
                crop = crop.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            pixels = np.asarray(crop, dtype=np.float32) / 255.0
            pixels = (pixels - 0.5) / 0.5
            tensor = torch.from_numpy(pixels).unsqueeze(0)
        return tensor, label


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def run_epoch(model, loader, loss_fn, optimizer, device, training: bool):
    model.train(training)
    total_loss = total_correct = total_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/OCR/OCR"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lenet5_chars.pt"))
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_set = CharacterCropDataset(args.data, "train", augment=True, max_items=args.max_train)
    val_set = CharacterCropDataset(args.data, "val", augment=False, max_items=args.max_val)
    if not train_set or not val_set:
        raise RuntimeError(f"Dataset not found or empty: {args.data}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeNet5(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device} train_crops={len(train_set)} val_crops={len(val_set)} classes={len(CLASS_NAMES)}")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "val_acc": val_acc}, args.output)
    print(f"saved={args.output} best_val_acc={best_acc:.4f}")
    (args.output.parent / "classes.json").write_text(json.dumps(CLASS_NAMES), encoding="utf-8")


if __name__ == "__main__":
    main()
