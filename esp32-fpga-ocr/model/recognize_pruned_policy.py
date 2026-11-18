"""PC plate OCR using the pruned network and a cautious decision policy."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_independent_structured import perspective_correct, segment_characters

CLASSES = '0123456789ABCDEFGHKLMNPSTUVXYZ'

def decide(logits, temperature=0.6098777047334674, min_score=0.4,
           min_margin=0.05, min_ratio=1.0):
    """Experimental transfer of 31-class policy; parameters not calibrated here."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(CLASSES) or len(logits) not in (8, 9):
        raise ValueError('Expected 8 or 9 glyphs with 30 logits each')
    if not np.isfinite(logits).all() or temperature <= 0:
        raise ValueError('Invalid logits or temperature')
    z = logits / temperature
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    details = []
    for pos, row in enumerate(p):
        allowed = np.arange(30) if len(p) == 9 and pos == 3 else np.arange(10, 30) if pos == 2 else np.arange(10)
        order = allowed[np.argsort(-row[allowed], kind='stable')]
        raw, selected = int(row.argmax()), int(order[0])
        score = float(row[selected]); margin = float(score - row[order[1]])
        ratio = score / max(float(row[raw]), 1e-12)
        accepted = bool(score >= min_score and margin >= min_margin and (raw == selected or ratio >= min_ratio))
        top = np.argsort(-row, kind='stable')[:3]
        details.append({'position': pos + 1, 'raw': CLASSES[raw], 'candidate': CLASSES[selected],
            'accepted': accepted, 'score': score, 'allowed_margin': margin, 'alternative_ratio': ratio,
            'top3': [{'character': CLASSES[int(i)], 'score': float(row[i])} for i in top]})
    return {'raw': ''.join(d['raw'] for d in details), 'candidate': ''.join(d['candidate'] for d in details),
            'accepted': all(d['accepted'] for d in details), 'characters': details}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='already cropped plate ROI')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('lenet5_structured50_30class.ts'))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    plate, applied, angle = perspective_correct(image)
    segmented = segment_characters(plate, expected_count=None, enhancement='clahe_sharp')
    if segmented is None:
        print(json.dumps({'accepted': False, 'reason': 'segmentation_failed'})); return
    crops, rows, count = segmented
    if count not in (8, 9):
        print(json.dumps({'accepted': False, 'reason': 'unsupported_character_count', 'count': count})); return
    batch = np.stack([(c.astype(np.float32) / 255 - .5) / .5 for c in crops])[:, None]
    model = torch.jit.load(str(args.model), map_location='cpu').eval()
    with torch.inference_mode():
        result = decide(model(torch.from_numpy(batch)).numpy())
    result.update({'rows': rows, 'perspective_corrected': applied, 'angle_deg': angle,
      'policy_status': 'experimental: thresholds transferred from different model; not calibrated for pruning',
      'model_classes': 30, 'class_O_available': False})
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
