# Dense vs. structured-pruned LeNet-5: accuracy, MAC, and measured speed

## Compared models and data

This is the matched enhanced `train(1)` pair:

- Dense: `artifacts/independent_lenet5_dense_enhanced_train1.pt`
- Structured 50%: `artifacts/independent_lenet5_structured50_enhanced_train1.pt`

Both checkpoints use the same saved 36-class order, 32×32 grayscale inputs,
and the same enhanced crop dataset/split. This is more valid than comparing
the latest 1,000-plate dense run, which is a different 30-class task. Dense
trained for five epochs; structured transferred selected channels from dense
and fine-tuned for eight epochs. The unequal training schedule matters when
interpreting accuracy, but the forward-latency test below uses the final fixed
architectures and identical input tensors.

These historical checkpoints have 36 outputs. Both saved checkpoints have an
identical 36-symbol ordering, which the benchmark checks before loading. The
current general-purpose OCR code also contains a newer 30-class vocabulary;
do not load this historical pair through that 30-class recognizer without an
explicit class-order migration or matching model definition.

The preparation accepted 25,286 of 37,297 source frames and generated 202,748
character crops. The shared validation split has 2,529 labeled plates. Source
and crop acceptance details are in `artifacts/enhanced_crop_training_comparison.md`.

## Accuracy on the shared validation data

The first exact-match score uses the dataset-provided character length to
select a segmentation candidate. The runtime ROI evaluation does not give the
segmenter that expected count; skipped plates count as incorrect. It still
starts from an already-cropped plate ROI, not a full frame.

| Metric | Dense, no pruning | Structured 50% | Difference |
|---|---:|---:|---:|
| Character accuracy on prepared crops | 91.43% | 91.28% | −0.16 percentage point |
| Exact plate, length-assisted | 2,043/2,529 = 80.78% | 2,031/2,529 = 80.31% | −0.47 pp |
| Exact plate, runtime ROI path | 1,772/2,489 = 71.19% | 1,759/2,489 = 70.67% | −0.52 pp |
| Runtime ROI skipped | 315/2,489 = 12.66% | 315/2,489 = 12.66% | no change |
| Runtime count mismatch | 111/2,489 = 4.46% | 111/2,489 = 4.46% | no change |

| Plate group | Dense character / exact | Structured character / exact |
|---|---:|---:|
| One row (1,488) | 98.00% / 91.80% | 97.94% / 91.33% |
| Two-row car (685) | 92.22% / 80.44% | 92.15% / 80.15% |
| Two-row motorcycle (356) | 65.08% / 35.39% | 64.44% / 34.55% |

By labeled string length, exact match was 71.01% vs. 67.63% for length 7
(207 plates), 86.80% vs. 86.80% for length 8 (2,060), 41.54% vs. 39.62%
for length 9 (260), and 0% vs. 0% for length 10 (only 2 samples). These
subsets are unbalanced; in particular, two 10-character plates are not enough
to estimate performance reliably. The dominant accuracy limitation is
segmentation and the difficult two-row motorcycle group, not MAC count alone.

## Full MAC and parameter calculations

Input is 32×32. After `conv5 → pool2 → conv5 → pool2`, the dense feature map
is `16×5×5`; the structured map is `10×5×5`. There are 36 output classes.
Each multiply-accumulate (MAC) in the formulas counts one weight×activation
product accumulated into an output.

| Layer | Dense MAC calculation | Dense MAC | Structured MAC calculation | Structured MAC |
|---|---|---:|---|---:|
| Conv1 | 6×5×5×28×28 | 117,600 | 4×5×5×28×28 | 78,400 |
| Conv2 | 16×6×5×5×10×10 | 240,000 | 10×4×5×5×10×10 | 100,000 |
| FC1 | (16×5×5)×120 | 48,000 | (10×5×5)×100 | 25,000 |
| FC2 | 120×84 | 10,080 | 100×70 | 7,000 |
| FC3 | 84×36 | 3,024 | 70×36 | 2,520 |
| **Total per character** | sum of layers | **418,704** | sum of layers | **212,920** |

```text
MAC reduction = (418,704 − 212,920) / 418,704 × 100 = 49.1478%
MAC ratio     = 418,704 / 212,920 = 1.9665×
```

The 1.9665× figure is a ratio of arithmetic work, **not measured latency**.
It would only be an ideal speedup if the implementation were compute-bound
and sustained the same MAC throughput without other bottlenecks.

| Characters on plate | Dense MAC | Structured MAC |
|---:|---:|---:|
| 7 | 2,930,928 | 1,490,440 |
| 8 | 3,349,632 | 1,703,360 |
| 9 | 3,768,336 | 1,916,280 |
| 10 | 4,187,040 | 2,129,200 |

| Resource proxy | Dense | Structured 50% | Reduction |
|---|---:|---:|---:|
| Parameters | 63,916 | 35,840 | 43.93% |
| Saved PyTorch checkpoint | 261,013 bytes | 149,061 bytes | 42.89% |

Checkpoint bytes include serialization metadata and are not the FPGA's final
weight-memory size. Structured pruning removes channels/neurons, so these
smaller layer dimensions are physical in the model graph; this differs from
unstructured zero weights, which may not make ordinary dense hardware faster.

## Measured CPU model-forward latency

The benchmark feeds the same preallocated CPU tensor to both models, with
shape `N×1×32×32`. It warms each model for 250 forwards per batch size, then
measures 2,500 forwards for every model and batch. Model order alternates each
iteration. The run uses PyTorch inference mode and one CPU thread. It measures
the model forward plus CPU framework dispatch, but excludes image loading,
perspective correction, segmentation, tensor stacking, output decoding, and
any laptop-to-FPGA transport.

Host: Intel Core i5-12500H, 16 logical CPUs; PyTorch 2.14.0 CPU build. The
recognizer batches the character crops from one plate into a single model
call; thus batch sizes 7–10 are the realistic cases. Values are medians and
95th-percentile latency (p95) across 2,500 measured calls.

| Characters per call | Dense p50 (ms) | Structured p50 (ms) | Dense p95 (ms) | Structured p95 (ms) | Speedup by p50 | Latency reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.27750 | 0.24915 | 0.60038 | 0.56351 | 1.114× | 10.22%* |
| 7 | 0.95560 | 0.81235 | 1.15525 | 0.97711 | 1.176× | 14.99% |
| 8 | 1.04915 | 0.88365 | 1.25016 | 1.06365 | 1.187× | 15.77% |
| 9 | 1.17750 | 0.98610 | 1.57754 | 1.35821 | 1.194× | 16.25% |
| 10 | 1.22985 | 1.02140 | 1.50305 | 1.27036 | 1.204× | 16.95% |

For a typical 8-character plate:

```text
speedup = dense median / structured median
        = 1.04915 ms / 0.88365 ms = 1.187×

latency reduction = (1 − structured median / dense median) × 100
                  = (1 − 0.88365 / 1.04915) × 100 = 15.77%
```

**Conclusion on this laptop:** for realistic plate-sized batches, this
benchmark shows about 15–17% lower median model latency, a modest but
measurable speedup. That is smaller than the 49.15% MAC reduction because
framework/call overhead and other CPU costs do not shrink with the channels.
The batch-1 difference is less robust and should not be used as the main
claim. CPU scheduling produced occasional outliers, so the raw measurements
are retained and p50/p95 are reported instead of relying on a single timing.

## What data proves speedup, and what to inspect on DE10-Lite

- **MAC count** says how much arithmetic the graph needs; it does not by
  itself prove the model runs faster.
- **Measured latency on the same target** is the direct speed evidence. Keep
  device, input, batch size, precision, thread/parallelism, warm-up, and
  preprocessing identical. Calculate `speedup = T_dense/T_structured` and
  `latency reduction = (1−T_structured/T_dense)×100%`.
- **For DE10-Lite**, synthesize both HDL builds in Quartus using the same
  constraints. Compare DSP blocks, ALMs, memory bits, fMAX and timing slack;
  then measure cycles on identical plate inputs. Board latency is
  `cycles / clock frequency`; board speedup is `T_dense/T_structured`.
  If the design reuses one MAC engine, fewer MACs may reduce cycles. If it
  computes in parallel, pruning may save DSPs/area while latency stays similar.

No Verilog, Quartus synthesis, or DE10-Lite board timing is included in this
experiment. So the result establishes a CPU speedup for this PyTorch forward
benchmark only; FPGA speedup remains unmeasured.

## Reproduce and inspect raw data

```powershell
.venv\Scripts\python.exe benchmark_structured_speed.py --repeats 2500 --warmup 250
```

- Benchmark code: `benchmark_structured_speed.py`
- Environment and summary: `artifacts/structured_speed_benchmark.json`
- Raw timings, 25,000 rows: `artifacts/structured_speed_benchmark_raw.csv`
- Matched accuracy/failure details: `artifacts/enhanced_crop_training_comparison.md`
