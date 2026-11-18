"""Classify one already segmented 32x32 grayscale character image."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

CLASSES = '0123456789ABCDEFGHKLMNPSTUVXYZ'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image', type=Path, help='32x32 grayscale glyph prepared with the training preprocessing')
parser.add_argument('--model', type=Path, default=Path(__file__).with_name('lenet5_structured50_30class.ts'))
args = parser.parse_args()
image = Image.open(args.image).convert('L')
if image.size != (32, 32):
    raise ValueError('Expected a segmented and normalized 32x32 glyph; do not pass a full plate.')
pixels = np.asarray(image, dtype=np.float32) / 255.0
tensor = torch.from_numpy((pixels - 0.5) / 0.5)[None, None]
model = torch.jit.load(str(args.model), map_location='cpu').eval()
with torch.inference_mode():
    values, indices = model(tensor).softmax(1).topk(3, dim=1)
for rank, (value, index) in enumerate(zip(values[0], indices[0]), 1):
    print(f'{rank}: {CLASSES[int(index)]} score={float(value):.4f}')
