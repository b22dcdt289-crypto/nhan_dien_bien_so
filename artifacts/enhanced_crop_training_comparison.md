# Enhanced crop preprocessing and dense/pruned LeNet-5 comparison

## Method

The source `train(1)` filenames provide labeled plate bounding boxes. The
preparation script crops those annotated regions, applies perspective correction,
then uses mild bilateral denoising, CLAHE local contrast normalization, and a
small unsharp-mask step before Otsu/adaptive-threshold character segmentation.
The same preprocessing is applied to the training and validation crops. The
existing data directory was retained; enhanced crops are in
`data/independent_chars_train1_enhanced`.

The split is fixed by the preparation script and contains 2,529 validation
plates from 25,286 accepted source images. Dense LeNet-5 trained for five
epochs. The 50%-structured model was initialized by transferring the strongest
channels and fine-tuned for eight epochs. Both use the same enhanced crop set
and validation split.

## OCR result on prepared, labeled character crops

| Metric | Dense, no pruning | Structured pruning 50% | Difference |
|---|---:|---:|---:|
| Character accuracy | 91.43% | 91.28% | -0.16 pp |
| Exact plate accuracy | 80.78% (2,043/2,529) | 80.31% (2,031/2,529) | -0.47 pp |
| MAC per character | 418,704 | 212,920 | -49.15% |
| MAC for 8 characters | 3,349,632 | 1,703,360 | -49.15% |
| MAC for 9 characters | 3,768,336 | 1,916,280 | -49.15% |
| MAC for 10 characters | 4,187,040 | 2,129,200 | -49.15% |

| Plate group | Dense character / exact | Pruned character / exact | Dense → pruned exact change |
|---|---:|---:|---:|
| One row (1,488) | 98.00% / 91.80% | 97.94% / 91.33% | -0.47 pp |
| Two-row car (685) | 92.22% / 80.44% | 92.15% / 80.15% | -0.29 pp |
| Two-row motorcycle (356) | 65.08% / 35.39% | 64.44% / 34.55% | -0.84 pp |

| Ground-truth string length | Plates | Dense exact | Pruned exact |
|---:|---:|---:|---:|
| 7 | 207 | 71.01% | 67.63% |
| 8 | 2,060 | 86.80% | 86.80% |
| 9 | 260 | 41.54% | 39.62% |
| 10 | 2 | 0% | 0% |

This source split includes 207 labels of length 7 and only two of length 10;
the 10-character result is too small to generalize. Keep those counts visible
in the report rather than describing the source as a uniform 8-character set.

## Actual segmentation without the known character count

At inference, the filename label cannot tell the segmenter how many glyphs to
find. The following rerun starts with the annotated plate ROI, calls the
segmenter without `expected_count`, and applies the same supported-length rule
as the recognizer. The ROI location is still ground truth, so this measures
segmentation plus OCR and does not measure full-scene plate localization.

| Metric | Dense enhanced | Structured 50% enhanced |
|---|---:|---:|
| Exact plate, all 2,489 source validation records | 71.19% (1,772) | 70.67% (1,759) |
| Character-position accuracy on attempted plates | 90.43% | 90.30% |
| Skipped | 315 (12.66%) | 315 (12.66%) |
| Detected length disagreed with label | 111 (4.46%) | 111 (4.46%) |

The difference between 80.78% and 71.19% for the dense model is mainly the
segmentation protocol: the training-data preparation uses the known filename
length to select/reject a segmentation candidate, while runtime does not know
that count. This is an oracle-assisted validation step and must not be
presented as full end-to-end accuracy. Full-scene plate-detector accuracy is
measured separately below; this ROI evaluation still crops using the dataset's
ground-truth box.

## Full-frame plate localization check

The current CPU-only contour/morphology proposal was compared with the labeled
boxes on the same 2,529 validation frames. It finds a proposal on almost every
frame, but the box placement is poor: recall alone is misleading because a
proposal may cover only part of the plate or include much of the vehicle.

| Plate group | Found | Recall | Mean IoU | IoU ≥ 0.50 | IoU ≥ 0.75 |
|---|---:|---:|---:|---:|---:|
| One row (1,488) | 1,488 | 100.00% | 0.227 | 18.28% | 9.54% |
| Two-row motorcycle (356) | 355 | 99.72% | 0.119 | 8.99% | 8.71% |
| Two-row car (685) | 685 | 100.00% | 0.066 | 4.67% | 1.61% |

Therefore there is not yet a trustworthy full-frame end-to-end accuracy number.
The crop/OCR scores above assume a correctly localized plate ROI; the current
proposal stage needs replacing or substantially improving before real-camera
performance can be claimed. The source dataset's annotated box is used to make
training crops and to isolate OCR evaluation, not as an inference-time input.

## Crop acceptance and enhancement ablation

The enhanced preparation accepted 25,286 of 37,297 source images (67.80%),
versus 24,888 (66.73%) before enhancement. Skip rates by source group were:

| Group | Enhanced accepted | Enhanced skip rate | Previous skip rate |
|---|---:|---:|---:|
| One row | 14,494 / 19,086 | 24.06% | 24.08% |
| Two-row motorcycle | 3,478 / 5,593 | 37.82% | 40.46% |
| Two-row car | 7,314 / 12,618 | 42.04% | 43.98% |

An ablation using the previous dense checkpoint on ground-truth plate ROIs
found that applying CLAHE at inference reduced exact plate accuracy from
79.47% to 76.54% and raised character-count mismatches from 91 to about 230.
That checkpoint had been trained without this enhancement. This confirms that
sharpening/contrast processing must be used consistently at training and
inference; it also does not prove that enhancement itself improves OCR.

## Why accuracy remains limited

1. **Character segmentation is still the largest runtime gap.** Validation
   data preparation has the ground-truth label length and can reject a wrong
   count. Runtime must infer the count, causing 12.66% skips and 4.46% length
   mismatches in this ROI-only evaluation.
2. **Full-frame localization is not accurate enough.** The contour proposal's
   IoU≥0.5 rate is only 4.67% for two-row cars and 8.99% for motorcycles, so
   nearly-perfect proposal recall does not mean the plate was correctly cut.
3. **Two-row motorcycle plates are difficult.** Their exact result is about
   35% with either model, versus about 92% on one-row plates. Small glyphs,
   tight spacing, row separation, and merged/split connected components lead
   to misordered or missing characters.
4. **Contrast enhancement cannot recreate detail that is absent.** CLAHE can
   amplify background texture and compression noise; sharpening can thicken
   strokes or join neighboring glyphs. Blur and viewpoint distortion require
   better source resolution and geometric correction, not stronger sharpening
   alone.
5. **The classifier confuses visually similar glyphs.** In the dense enhanced
   validation results, common substitutions include `1↔2`, `9→2`, `6→3`, and
   `3→1`. The current OCR is a per-character 36-class classifier and has no
   plate-format or sequence correction.
6. **Training duration and test coverage are limited.** This run used five
   dense epochs and eight pruning fine-tuning epochs; the accepted random
   validation split is not an independent camera-condition test. Blur/angle
   subgroups are small.

## Reproducibility

```powershell
.venv\Scripts\python.exe train_independent_structured.py --mode all --source "data/OCR/OCR/images/train(1)/detection" --data data/independent_chars_train1_enhanced --metrics artifacts/independent_pipeline_enhanced_metrics.json --enhancement clahe_sharp --clean --dense-epochs 5 --structured-epochs 8 --dense-output artifacts/independent_lenet5_dense_enhanced_train1.pt --output artifacts/independent_lenet5_structured50_enhanced_train1.pt
```

The detailed prediction for every accepted validation plate is in
`dense_enhanced_plate_results.csv` and `structured_enhanced_plate_results.csv`.
Use the enhanced structured checkpoint in the Python recognizer with:

```powershell
.venv\Scripts\python.exe recognize_independent.py path\to\frame.jpg --model artifacts/independent_lenet5_structured50_enhanced_train1.pt
```
