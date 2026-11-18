"""Recognize an already-cropped Vietnamese license plate ROI on a PC."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from segment_enhanced import segment

CLASSES = '0123456789ABCDEFGHKLMNPSTUVXYZO'

def decide(logits, policy):
    z = np.asarray(logits, dtype=np.float64) / policy['temperature']
    p = np.exp(z - z.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    n = len(p); details = []
    for pos, scores in enumerate(p):
        unrestricted = n == 9 and pos == 3
        allowed = np.arange(31) if unrestricted else np.arange(10, 31) if pos == 2 else np.arange(10)
        raw = int(scores.argmax()); work = scores.copy()
        if not unrestricted and scores[0] + scores[30] >= .8 and min(scores[0], scores[30]) >= .12:
            work[30 if pos == 2 else 0] = scores[0] + scores[30]
        order = allowed[np.argsort(-work[allowed], kind='stable')]
        selected = int(order[0]); score = float(work[selected]); margin = float(score - work[order[1]])
        ratio = score / max(float(scores[raw]), 1e-12)
        accepted = bool(score >= policy['minimum_score'] and margin >= policy['minimum_allowed_margin'] and
                        (raw == selected or ratio >= policy['minimum_alternative_ratio']))
        top = np.argsort(-scores, kind='stable')[:3]
        details.append({'position': pos + 1, 'raw': CLASSES[raw], 'candidate': CLASSES[selected],
            'accepted': accepted, 'decision_score': score, 'margin': margin,
            'top3': [{'character': CLASSES[int(i)], 'score': float(scores[i])} for i in top]})
    return {'raw': ''.join(d['raw'] for d in details), 'candidate': ''.join(d['candidate'] for d in details),
            'accepted': all(d['accepted'] for d in details), 'characters': details}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='image of an already cropped plate')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None: parser.error('Cannot read image')
    glyphs, info = segment(image)
    if glyphs is None:
        print(json.dumps({'accepted': False, 'reason': 'segmentation_failed', 'details': info}, ensure_ascii=False, indent=2)); return
    model = torch.jit.load(str(ROOT / 'model/lenet_k3_31class.ts'), map_location='cpu').eval()
    policy = json.loads((ROOT / 'model/training-results/calibration_freeze.json').read_text(encoding='utf-8'))['fp32_policy']
    with torch.inference_mode(): logits = model(torch.from_numpy(glyphs).float().unsqueeze(1) / 127.5 - 1).numpy()
    result = decide(logits, policy); result['segmentation'] = info
    result['note'] = 'PC FP32; ROI input; confidence is not guaranteed accuracy; no FPGA used'
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
