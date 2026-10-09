"""LeNet-style 3x3 ablation with valid convolutions on 32x32 glyphs.

Shape: 32 -> Conv3 30 -> AvgPool 15 -> Conv3 13 -> AvgPool 6.
This is a 3x3 variant, not the canonical 5x5 LeNet-5 from the 1998 paper.
"""

from __future__ import annotations

import torch
from torch import nn

from train_independent_structured import top_indices
from train_lenet5 import CLASS_NAMES


class LeNet5K3(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES), conv2_channels: int = 16):
        super().__init__()
        self.conv2_channels = conv2_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, conv2_channels, kernel_size=3),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv2_channels * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


class LeNet5K3Pruned(LeNet5K3):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__(num_classes, conv2_channels=8)


def compact_k3(teacher: LeNet5K3, student: LeNet5K3Pruned) -> list[int]:
    """Select Conv2 output filters and the matching 6x6 FC input columns."""
    selected = top_indices(teacher.features[3].weight, 8).tolist()
    indices = torch.tensor(selected, dtype=torch.long)
    columns = (indices[:, None] * 36 + torch.arange(36)[None, :]).reshape(-1)
    with torch.no_grad():
        for index in (0,):
            student.features[index].weight.copy_(teacher.features[index].weight)
            student.features[index].bias.copy_(teacher.features[index].bias)
        student.features[3].weight.copy_(teacher.features[3].weight[indices])
        student.features[3].bias.copy_(teacher.features[3].bias[indices])
        student.classifier[0].weight.copy_(teacher.classifier[0].weight[:, columns])
        student.classifier[0].bias.copy_(teacher.classifier[0].bias)
        for index in (2, 4):
            student.classifier[index].weight.copy_(teacher.classifier[index].weight)
            student.classifier[index].bias.copy_(teacher.classifier[index].bias)
    return selected


def model_size_k3(model: LeNet5K3) -> dict:
    c2 = model.conv2_channels
    layers = (
        ("features.0", model.features[0], 6 * 30 * 30 * 9),
        ("features.3", model.features[3], c2 * 13 * 13 * 6 * 9),
        ("classifier.0", model.classifier[0], 120 * c2 * 36),
        ("classifier.2", model.classifier[2], 120 * 84),
        ("classifier.4", model.classifier[4], 84 * len(CLASS_NAMES)),
    )
    layer_counts = {name: {"weight_shape": list(layer.weight.shape),
                           "weights": layer.weight.numel(), "biases": layer.bias.numel(),
                           "macs": macs}
                    for name, layer, macs in layers}
    weights = sum(row["weights"] for row in layer_counts.values())
    biases = sum(row["biases"] for row in layer_counts.values())
    macs = sum(row["macs"] for row in layer_counts.values())
    return {"weights": weights, "biases": biases, "parameters": weights + biases,
            "fp32_bytes": 4 * (weights + biases),
            "int8_weights_int32_bias_bytes": weights + 4 * biases,
            "macs_per_character": macs, "macs_per_8_character_plate": macs * 8,
            "macs_per_9_character_plate": macs * 9, "layers": layer_counts}
