# Mixed-source perspective LeNet-5 experiment — 1,000 training plates

## Run configuration

- Architecture: dense LeNet-5, 30 output classes (`0–9`, `A–Z` subset); randomly initialized; no pruning and no YOLO.
- Training: 40 epochs on CPU; select checkpoint by validation runtime exact-plate accuracy (validation plate ROI, no target character count supplied).
- Split: 1,000 train / 300 validation / 500 test plate identities, kept disjoint.
- Train sources: 949 filename-labeled one-row crops from `train(1)` + 51 filtered and visually reviewed pseudo-labeled `CarLongPlate` crops from `train`.
- Augmentation: Python/OpenCV perspective and small roll transforms for training only; 1,767/2,000 transformed views kept, 233 rejected by segmentation/count validation. No source image was modified or deleted.

## Held-out results

| Measurement | Result |
|---|---:|
| Character accuracy (conditional on ground-truth-length segmentation) | 3,915 / 3,971 = **98.59%** |
| Raw character accuracy (before plate-format constraint) | **98.46%** |
| Exact plate string (ground-truth-length segmentation) | 472 / 500 = **94.40%** |
| 95% Wilson interval for exact plate score | **92.03–96.10%** |
| Runtime ROI exact string (segmentation infers count; skips count as failures) | 471 / 500 = **94.20%** |
| Runtime ROI coverage / skips | 500 / 500 attempted; 0 skips |
| Runtime ROI character-count mismatches | 4 |
| LeNet-5 parameters | 63,406 |
| Raw FP32 parameter storage (excluding checkpoint metadata) | about 247.7 KiB |
| MAC per classified character | **418,200** |
| MAC for 7 / 8 / 9 characters | **2,927,400 / 3,345,600 / 3,763,800** |

The 96% exact-plate gate was not met. Therefore the conditional 3,000-plate
stage was not started. These measurements begin with already-cropped plate
ROIs; they are not full-frame localization, camera, or DE10-Lite measurements.
The character score uses segmentation supplied with the ground-truth length,
so the runtime ROI exact score is the more representative OCR-path result.

### Position accuracy on the 500 test ROIs

| Position | Correct / total | Accuracy |
|---:|---:|---:|
| 1 | 496 / 500 | 99.20% |
| 2 | 493 / 500 | 98.60% |
| 3 | 491 / 500 | 98.20% |
| 4 | 495 / 500 | 99.00% |
| 5 | 494 / 500 | 98.80% |
| 6 | 489 / 500 | 97.80% |
| 7 | 492 / 500 | 98.40% |
| 8 | 465 / 471 | 98.73% |

Exact-match by labeled string length was 26/29 (89.66%) for seven-character
samples and 446/471 (94.69%) for eight-character samples. This test set is
strongly dominated by eight-character labels; uncommon letters are too sparse
for stable per-letter claims (for example D=3, N=7, M=17).

## Data audit and why samples fail

Primary source: `data/OCR/OCR/images/train(1)/detection/one_row`. Of 19,086
scanned frames, 7,189 failed quality or geometry/count filters: 2,079 aspect
ratio, 1,495 resolution, 3,525 segmentation/count, 74 blur, 3 contrast, 6
roll angle, and 7 row/count. A further 1,954 frames were excluded because
their identities belonged to the prior experiment. Remaining repeated views
were deduplicated by plate identity; one crop per identity was retained.

Legacy source: `data/OCR/OCR/images/train` held 3,066 mixed-category images,
including 414 `CarLongPlate` candidates. Of those candidates, 269 failed the
minimum pseudo-label confidence, 33 were not one row or had an unsupported
length, 10 disagreed under the two preprocessing variants, 6 had an annotation
box-count mismatch, 4 failed aspect filtering, and 2 failed segmentation.
Ninety candidate annotations were audited and all 90 disagreed with teacher
OCR text. This shows those annotations are not safe to treat as labels without
manual review; it does not prove that teacher OCR is ground truth. After
deduplication, previous-identity exclusion, and visual review, 51 legacy
images were used in training. One visually reviewed pseudo-label correction
is stored only in the local ignored data folder, not in the public source.

Main causes limiting exact plate score:

1. One wrong character makes the entire string incorrect, even when per-character accuracy is high.
2. Character segmentation/thresholding can split, merge, or omit strokes; runtime count differed from truth on four ROIs.
3. Several visually similar symbols are easily confused under blur, low resolution, glare, or skew.
4. Labels and test support are uneven: rare letters are scarcely represented, and the legacy annotation audit found systematic disagreement.
5. Perspective augmentation synthesizes geometry but cannot replace real oblique, blurred, night, reflective, or motorcycle/two-row samples.

The most frequent test-set character confusions include `6→5`, `7→1`,
`7→3`, and `3→6` (3 examples each); `A→N`, `3→5`, `3→0`, `2→7`, and `0→4`
occurred twice each. These are observations from this 500-ROI test split, not
universal confusion rates.

## Local paths and reproduction

- Prepared split and plate crops: `data/one_row_combined_frontal_1000_perspective_v1/`
- Perspective preview: `data/one_row_combined_frontal_1000_perspective_v1/review/perspective_examples.png`
- Checkpoint: `artifacts/lenet5_dense_mixed_frontal_1000_perspective_bestformat_v2.pt`
- Aggregate metrics: `artifacts/mixed_frontal_1000_perspective_bestformat_v2_metrics.json`
- Confusion matrix: `artifacts/mixed_frontal_1000_perspective_bestformat_v2_metrics_confusion_matrix.csv`
- Training history: `artifacts/mixed_frontal_1000_perspective_bestformat_v2_history.csv`
- Per-plate predictions remain local and are excluded from Git.

Re-train from the existing prepared data:

```powershell
.venv\Scripts\python.exe train_one_row_1000.py --data data/one_row_combined_frontal_1000_perspective_v1 --epochs 40 --output artifacts/lenet5_dense_mixed_frontal_1000_perspective_bestformat_v2.pt --metrics artifacts/mixed_frontal_1000_perspective_bestformat_v2_metrics.json --test-csv artifacts/mixed_frontal_1000_perspective_bestformat_v2_test_predictions.csv --history-csv artifacts/mixed_frontal_1000_perspective_bestformat_v2_history.csv
```
