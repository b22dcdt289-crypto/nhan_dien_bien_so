"""Export the synthetic-only 2,000-plate rerun without any real-photo data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_lenet5 import CLASS_NAMES
from train_synthetic_car_compare import sha, split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/car_frontal_synthetic_2x1000_v1"))
    parser.add_argument("--run", type=Path, default=Path("artifacts/car2000_synthetic_only_20261001_v2"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "hardware/dataflow_study/results/synthetic_only_2000_v2")
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    if metrics["run"]["synthetic_only"] is not True or metrics["run"]["real_external_test_plates"] != 0:
        raise ValueError("Refusing to publish a run that read real-photo test data")
    manifest = args.data / "manifest.csv"
    if sha(manifest) != metrics["run"]["data_manifest_sha256"]:
        raise ValueError("Manifest checksum differs from training record")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    split = split_rows(rows, metrics["run"]["seed"])
    if len(rows) != 2000 or [len(split[key]) for key in ("train", "val", "test")] != [1600, 200, 200]:
        raise ValueError("Unexpected dataset or split size")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "history.csv", "dense_synthetic_confusion.csv",
                 "pruned_synthetic_confusion.csv", "cpu_interleaved.json"):
        shutil.copy2(args.run / name, args.output / name)
    hardware_path = args.run / "hardware_results.json"
    if hardware_path.is_file():
        shutil.copy2(hardware_path, args.output / "hardware_results.json")
    counts = {key: Counter("".join(row["label"] for row in group)) for key, group in split.items()}
    with (args.output / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["character", "total_2000", "train_1600", "validation_200", "test_200",
                         "dense_test_correct", "pruned_test_correct"])
        dense = metrics["models"]["dense"]["synthetic_plate_to_text"]["confusion_matrix_aligned_only"]
        pruned = metrics["models"]["pruned"]["synthetic_plate_to_text"]["confusion_matrix_aligned_only"]
        for i, character in enumerate(CLASS_NAMES):
            if sum(dense[i]) != counts["test"][character] or sum(pruned[i]) != counts["test"][character]:
                raise ValueError(f"Confusion support mismatch for {character}")
            writer.writerow([character, sum(counts[key][character] for key in counts),
                             counts["train"][character], counts["val"][character],
                             counts["test"][character], dense[i][i], pruned[i][i]])
    summary = {
        "training_and_evaluation_source": "2,000 generated frontal car plates only",
        "real_photos_read": False,
        "run": metrics["run"],
        "checkpoint_sha256": {name: metrics["models"][name]["checkpoint_sha256"]
                              for name in ("dense", "pruned")},
        "synthetic_test": {name: metrics["models"][name]["synthetic_plate_to_text"]["total"]
                           for name in ("dense", "pruned")},
        "model_size": {name: metrics["models"][name]["model_size"]
                       for name in ("dense", "pruned")},
        "tested_class_count": sum(counts["test"][character] > 0 for character in CLASS_NAMES),
        "all_dataset_class_count": sum(sum(counts[key][character] for key in counts) > 0
                                       for character in CLASS_NAMES),
    }
    if hardware_path.is_file():
        summary["hardware_test"] = json.loads(hardware_path.read_text(encoding="utf-8"))
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tested_classes": summary["tested_class_count"],
                      "classes_present_in_all_2000": summary["all_dataset_class_count"]}))


if __name__ == "__main__":
    main()
