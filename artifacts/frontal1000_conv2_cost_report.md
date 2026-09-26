# 1,000 near-frontal one-row plates — Conv2-only structured-pruned LeNet-5

## Dataset and protocol

- Source: `data/OCR/OCR/images/train(1)/detection/one_row`; train/validation/test plates: {'train': 1000, 'val': 300, 'test': 500}.
- Test set: 500 unseen, eight-character identities absent from previous model manifest; near-frontal image/ROI gates passed. Some do not pass current reference character segmentation and count as end-to-end failures.
- Preparation audit: scanned 19,086 frames; 9,772 passed image gates (9,206 unique labels, 6,358 segmentable 8-character identities).
- ROI selection filters: width×height ≥120×28 px, aspect ratio 3.7–5.8, Laplacian blur ≥80, grayscale contrast std ≥25, roll angle ≤4°. Images are real crops enhanced with CLAHE + mild sharpening, not AI super-resolution.
- Train characters: 8,000; supported classes: 19/30; balance max/min(nonzero): 669.000.
- Source-label QA: 7 labels were corrected only after visually checking the plate crops; substitutions by position are in the JSON. Uncertain disagreements were not auto-corrected.
- The test identities are absent from the previous model's manifest. Runtime character accuracy is reported only where predicted segmentation count aligns with the label; end-to-end plate exact counts skips and count mismatches as failures.

## Accuracy on the same unseen test plates

| Model | Runtime character accuracy (count aligned) | Full-plate exact (end-to-end; skips wrong) | Runtime segmentation coverage | Reference segmentation coverage | MAC/char | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Previous Conv2 student | N/A (0 chars; 0/500 plates) | 0.000% (0/500) | 67.6% | 0.0% | 274,200 | 38,198 |
| Fine-tuned on 1,000 | N/A (0 chars; 0/500 plates) | 0.000% (0/500) | 67.6% | 0.0% | 274,200 | 38,198 |

Character accuracy on raw new test plates is only defined for plates where the automatic segmenter returns a single row and exactly 8 glyphs; coverage is shown above. The 300-plate validation split below has reference character crops, but its identities occur in the previous model's manifest, so treat it as a tuning/diagnostic set, not an independent test.
- The new test segmenter produced exactly 8 aligned glyphs on 0/500 plates; therefore unseen-test character accuracy and confidence are **not measurable** here (N/A, not 0% classifier accuracy). Full-plate exact remains 0/500 because count mismatches/skips fail end-to-end.
- Test segmentation counts (previous/adapted share this weight-independent stage): `{"7": 262, "6": 39, "9": 66, "5": 15, "1": 12, "10": 10, "2": 7, "3": 5, "11": 2, "4": 4, "12": 1}`; attempted 338/500, skipped 162/500. The dominant failure is outputting 7 instead of 8 glyphs.
- Validation mean max-softmax confidence: previous 97.208%, adapted 98.414%; not calibrated and validation identities were seen by the prior model.
- Focused confusion submatrix source: `validation_reference_crops_fallback`. Off-diagonal confusion-cost M range: 1.010–1.574; full M is exported separately.

| Model | Validation character accuracy (reference crops) | Validation exact plate (reference crops) | Reference segmentation coverage |
|---|---:|---:|---:|
| Previous Conv2 student | 99.833% (2,400 chars) | 98.667% (296/300) | 100.0% |
| Fine-tuned on 1,000 | 99.875% (2,400 chars) | 99.000% (297/300) | 100.0% |

## Training, augmentation, and hard examples

- Cost-sensitive loss: weighted cross-entropy + 0.25 × expected confusion-cost penalty, with M derived only from the previous model's validation confusion matrix and Laplace smoothing.
- OHEM: top 25% hardest examples add an equally weighted auxiliary mean; discarded samples: 0.0%.
- Augmentation observed over 160,000 sample presentations: `{"geometry:none": 0.55096875, "photometric:brightness_contrast": 0.2301, "photometric:none": 0.4494, "photometric:noise": 0.2208, "geometry:affine": 0.11955, "geometry:perspective": 0.078675, "photometric:blur": 0.0997, "geometry:rotation": 0.25080625}` (fractions per training glyph presentation).
- Test confidence: N/A because no test plate produced an aligned 8-glyph sequence; see validation confidence above and raw distributions in JSON.

## Compute and resource estimates

- LeNet-5 Conv2-only topology (Conv1=6 fixed, Conv2=8): **274,200 MAC/character**, **2,193,600 MAC/8-character plate**, **38,198 parameters**.
- FP32 raw weights: 152,792 bytes; INT8 raw-weight estimate: 38,198 bytes (not an exported/validated quantized model).
- CPU software latency: see benchmark in JSON; the 8-character batch p50 is 0.3830 ms (not FPGA timing).
- Interleaved CPU batch-8 p50: previous 0.3811 ms → adapted 0.3830 ms; measured delta +0.5%. Same topology/MAC, so this is not a structural speedup.
- At the existing SDC target 50 MHz and an assumed single MAC/cycle: 274,200 cycles/character = 5.484 ms; 8 characters = 2,193,600 cycles = 43.872 ms. This is an operation-count estimate, not a measured design result.
- Quartus synthesis/STA and AI RTL are not available in this environment; ALM/DSP/BRAM, achieved fMAX, pipeline cycles, and hardware latency are therefore **not measured**. Current DE10-Lite top is a heartbeat demo, not this classifier.

## Interpretation

Fine-tuning changes weights, not the structured topology: MACs and parameter count remain the same as the previous Conv2-pruned student. Any measured CPU latency delta is benchmark variation/weight-dependent runtime noise, not a structural speedup. The model itself classifies a pre-cropped glyph; the current classical segmentation stage is part of end-to-end coverage and is counted explicitly.

Full aggregated metrics: `artifacts\frontal1000_conv2_cost_metrics.json`. Confusion submatrix: `artifacts/frontal1000_confusion_submatrix.csv`; full confusion-cost matrix M: `artifacts/frontal1000_confusion_cost_M.csv`; class/augmentation distributions: `artifacts/frontal1000_class_distribution.csv` and `artifacts/frontal1000_augmentation_distribution.csv`.
