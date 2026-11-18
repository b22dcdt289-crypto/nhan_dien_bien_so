"""Export the trained 31-class K3 FP32 model for portable PC inference."""
from pathlib import Path
import torch
from torch import nn

class LeNetK3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 6, 3), nn.Tanh(), nn.AvgPool2d(2),
                                      nn.Conv2d(6, 16, 3), nn.Tanh(), nn.AvgPool2d(2))
        self.classifier = nn.Sequential(nn.Linear(400, 120), nn.Tanh(), nn.Linear(120, 84),
                                        nn.Tanh(), nn.Linear(84, 31))
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

folder = Path(__file__).resolve().parent
ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
model = LeNetK3().eval()
model.load_state_dict(ck['model'])
torch.manual_seed(27)
x = torch.rand(8, 1, 28, 28) * 2 - 1
with torch.inference_mode():
    traced = torch.jit.trace(model, x)
    traced.save(str(folder / 'lenet_k3_31class.ts'))
    torch.testing.assert_close(model(x), torch.jit.load(str(folder / 'lenet_k3_31class.ts'))(x), rtol=0, atol=0)
print('31-class export verified: 8 inputs, 248 identical logits')
