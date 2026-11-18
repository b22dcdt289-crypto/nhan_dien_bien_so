"""Measure CPU recognition latency on repeated cropped plate images."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recognize import ROOT, decide, segment

def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {'median_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95)),
            'mean_ms': float(statistics.mean(values)), 'samples': len(values)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path, help='already cropped plate ROI')
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.threads < 1: p.error('invalid runs, warmup or threads')
    torch.set_num_threads(args.threads); cv2.setNumThreads(args.threads)
    image = cv2.imread(str(args.image))
    if image is None: p.error('Cannot read image')
    model = torch.jit.load(str(ROOT / 'model/lenet_k3_31class.ts'), map_location='cpu').eval()
    policy = json.loads((ROOT / 'model/training-results/calibration_freeze.json').read_text(encoding='utf-8'))['fp32_policy']
    plate, inference, segmentation, decision, per_glyph = [], [], [], [], []
    final = None
    with torch.inference_mode():
        for iteration in range(args.warmup + args.runs):
            start = time.perf_counter_ns()
            glyphs, info = segment(image)
            t_segment = time.perf_counter_ns()
            if glyphs is None: raise RuntimeError(f"Segmentation failed: {info.get('reason')}")
            tensors = torch.from_numpy(glyphs).float().unsqueeze(1) / 127.5 - 1
            glyph_timings = []; outputs = []
            for tensor in tensors:
                a = time.perf_counter_ns(); outputs.append(model(tensor.unsqueeze(0))); b = time.perf_counter_ns()
                glyph_timings.append((b-a)/1e6)
            t_inference = time.perf_counter_ns()
            final = decide(torch.cat(outputs).numpy(), policy)
            end = time.perf_counter_ns()
            if iteration >= args.warmup:
                segmentation.append((t_segment-start)/1e6); inference.append((t_inference-t_segment)/1e6)
                decision.append((end-t_inference)/1e6); plate.append((end-start)/1e6); per_glyph.extend(glyph_timings)
    result = {'model': '31-class FP32 LeNet K3', 'device': 'CPU', 'platform': platform.platform(),
       'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(), 'torch_version': torch.__version__,
       'threads': args.threads, 'runs': args.runs, 'warmup': args.warmup, 'glyphs_per_plate': len(final['characters']),
       'candidate': final['candidate'], 'accepted': final['accepted'], 'segmentation': stats(segmentation),
       'glyph_forward': stats(per_glyph), 'all_glyphs_with_tensor_preparation': stats(inference),
       'decision': stats(decision), 'roi_to_result': stats(plate),
       'note': 'Same already-cropped in-memory ROI repeated; excludes image load, camera, UART and FPGA. FPS derived from latency is not camera throughput or accuracy.'}
    result['roi_per_second_from_median_latency'] = 1000.0/result['roi_to_result']['median_ms']
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output: args.output.write_text(text + '\n', encoding='utf-8')
    print(text)

if __name__ == '__main__': main()
