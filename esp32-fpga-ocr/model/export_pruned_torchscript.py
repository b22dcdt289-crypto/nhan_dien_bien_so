"""Export the trained 30-class structured50 model as a standalone TorchScript module."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_independent_structured import LeNet5Structured50
from train_lenet5 import CLASS_NAMES, require_compatible_classes

checkpoint_path = ROOT / 'artifacts/lenet5_structured50_channel30_matched_v1.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
require_compatible_classes(checkpoint, checkpoint_path)
model = LeNet5Structured50(len(CLASS_NAMES)).eval()
model.load_state_dict(checkpoint['model'])
torch.manual_seed(27)
sample = torch.rand(8, 1, 32, 32) * 2 - 1
with torch.inference_mode():
    traced = torch.jit.trace(model, sample)
    torch.testing.assert_close(model(sample), traced(sample), rtol=0, atol=0)
    output = Path(__file__).resolve().parent / 'lenet5_structured50_30class.ts'
    traced.save(str(output))
    torch.testing.assert_close(model(sample), torch.jit.load(str(output))(sample), rtol=0, atol=0)
print(f'saved={output}; exact_match=8x30 logits; input=N,1,32,32 range [-1,1]')
