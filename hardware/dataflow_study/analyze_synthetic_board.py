"""Audit the real DE10-Lite JTAG runs for the synthetic-only checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_lenet5 import CLASS_NAMES


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    proportion = successes / trials
    return (proportion + z * z / (2 * trials) -
            z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))) / (1 + z * z / trials)


def fit_metrics(root: Path, name: str) -> dict:
    base = root / f"{name}_stream"
    fit_path = base / "output_files/OcrBench.fit.summary"
    sta_path = base / "output_files/OcrBench.sta.rpt"
    fit = fit_path.read_text(encoding="utf-8", errors="replace")
    sta = sta_path.read_text(encoding="utf-8", errors="replace")

    def resource(label: str) -> tuple[int, int]:
        match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*([\d,]+)\s*/\s*([\d,]+)", fit, re.M)
        if not match:
            raise ValueError(f"Quartus result is missing {label}: {fit_path}")
        return tuple(int(value.replace(",", "")) for value in match.groups())

    fmax = re.search(r";\s*([\d.]+) MHz\s*;\s*[\d.]+ MHz\s*;\s*MAX10_CLK1_50", sta)
    slack = re.search(r"Worst-case setup slack is\s*([\d.-]+)", sta)
    if "Fitter Status : Successful" not in fit or not fmax or not slack:
        raise ValueError(f"Quartus reports are not complete: {base}")
    return {
        "logic_elements": dict(zip(("used", "available"), resource("Total logic elements"))),
        "dedicated_registers": dict(zip(("used", "available"), resource("Dedicated logic registers"))),
        "memory_bits": dict(zip(("used", "available"), resource("Total memory bits"))),
        "9bit_multipliers": dict(zip(("used", "available"), resource("Embedded Multiplier 9-bit elements"))),
        "fmax_mhz_slow_85c": float(fmax.group(1)),
        "setup_slack_ns_slow_85c": float(slack.group(1)),
        "timing_fully_constrained": "Design is not fully constrained" not in sta,
        "sof_bytes": (base / "output_files/OcrBench.sof").stat().st_size,
        "sof_sha256": sha(base / "output_files/OcrBench.sof"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("hardware/dataflow_study/rtl_eval/synthetic_only_2000_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/car2000_synthetic_only_20261001_v2/hardware_results.json"))
    args = parser.parse_args()
    manifests = {name: json.loads((args.run / f"{name}_stream/generated/manifest.json").read_text(encoding="utf-8"))
                 for name in ("dense", "pruned")}
    baseline = manifests["dense"]
    for name, manifest in manifests.items():
        if manifest["real_photos_read"] or manifest["dataset_manifest_sha256"] != baseline["dataset_manifest_sha256"]:
            raise ValueError("Board runs do not share the generated-only dataset")
        if manifest["character_count"] != manifest["plate_count"] * 8:
            raise ValueError(f"Invalid glyph/plate counts in {name} manifest")

    result = {
        "measurement": "Direct USB-Blaster/JTAG result from DE10-Lite, MAX 10 10M50DAF484C7G, 50 MHz design clock",
        "board_jtag_id": "0x031050DD",
        "quartus": "25.1std.0 Lite",
        "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
        "selection": baseline["source"],
        "real_photos_read": False,
        "models": {},
    }
    summaries = {}
    for name in ("dense", "pruned"):
        manifest = manifests[name]
        generated = args.run / f"{name}_stream/generated"
        with (generated / "board_results.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_count = manifest["character_count"]
        if len(rows) != expected_count or [int(row["sample_index"]) for row in rows] != list(range(expected_count)):
            raise ValueError(f"Board run {name} is incomplete or sample order differs")
        if "sample_slot" in rows[0] and [int(row["sample_slot"]) for row in rows] != [i % 127 for i in range(expected_count)]:
            raise ValueError(f"JTAG sample slot wrap differs from expected sequence in {name}")
        sequences = [int(row["result_seq"]) for row in rows]
        if any((right - left) % 256 != 1 for left, right in zip(sequences, sequences[1:])):
            raise ValueError(f"JTAG result sequence discontinuity in {name}")
        predictions = [CLASS_NAMES[int(row["predicted_class"])] for row in rows]
        truth = manifest["labels"]
        if len(truth) != len(predictions):
            raise ValueError(f"Truth/prediction length mismatch in {name}")
        confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
        support = [0 for _ in CLASS_NAMES]
        correct = [0 for _ in CLASS_NAMES]
        for expected, predicted in zip(truth, predictions):
            actual_id = CLASS_NAMES.index(expected)
            predicted_id = CLASS_NAMES.index(predicted)
            confusion[actual_id][predicted_id] += 1
            support[actual_id] += 1
            correct[actual_id] += int(expected == predicted)
        by_layout = defaultdict(lambda: {"plates": 0, "exact": 0})
        exact_plates = 0
        for plate in manifest["plates"]:
            observed = "".join(predictions[index] for index in plate["indices"])
            exact = int(observed == plate["truth"])
            exact_plates += exact
            by_layout[plate["layout"]]["plates"] += 1
            by_layout[plate["layout"]]["exact"] += exact
        cycles = sorted({int(row["cycles"]) for row in rows})
        if len(cycles) != 1:
            raise ValueError(f"Unexpected per-sample cycle variation in {name}: {cycles}")
        load_times = [int(row["jtag_load_ms"]) for row in rows]
        total_times = [int(row["jtag_total_ms"]) for row in rows]
        fpga_correct = sum(predicted == expected for predicted, expected in zip(predictions, truth))
        fpga_fixed_agreement = sum(predicted == expected for predicted, expected in zip(predictions, manifest["fixed_predictions"]))
        summaries[name] = {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "bitstream_sha256": sha(generated.parent / "output_files/OcrBench.sof"),
            "fp32_reference_correct": manifest["fp32_character_correct"],
            "int8_fixed_reference_correct": manifest["fixed_character_correct"],
            "fpga_correct_characters": fpga_correct,
            "fpga_accuracy": fpga_correct / len(truth),
            "fpga_accuracy_wilson_95_lower": wilson_lower(fpga_correct, len(truth)),
            "fpga_agrees_with_int8_reference": fpga_fixed_agreement,
            "fpga_exact_plates": exact_plates,
            "plate_count": len(manifest["plates"]),
            "plate_exact_accuracy": exact_plates / len(manifest["plates"]),
            "plate_exact_accuracy_wilson_95_lower": wilson_lower(exact_plates, len(manifest["plates"])),
            "by_layout": dict(by_layout),
            "cycles_per_character": cycles[0],
            "compute_ms_per_character_at_50mhz": cycles[0] * 1000 / 50_000_000,
            "jtag_load_ms": {"median": statistics.median(load_times),
                             "p95_nearest_rank_lower": sorted(load_times)[int(.95 * (len(load_times) - 1))]},
            "jtag_total_ms": {"median": statistics.median(total_times),
                              "p95_nearest_rank_lower": sorted(total_times)[int(.95 * (len(total_times) - 1))]},
            "per_character_confusion_matrix": confusion,
            "per_class": {character: {"support": support[index], "correct": correct[index]}
                          for index, character in enumerate(CLASS_NAMES) if support[index]},
            "quartus_postfit": fit_metrics(args.run, name),
        }
    dense_cycles = summaries["dense"]["cycles_per_character"]
    pruned_cycles = summaries["pruned"]["cycles_per_character"]
    dense_total = summaries["dense"]["jtag_total_ms"]["median"]
    pruned_total = summaries["pruned"]["jtag_total_ms"]["median"]
    result["models"] = summaries
    result["comparison"] = {
        "compute_cycle_speedup_dense_over_pruned": dense_cycles / pruned_cycles,
        "jtag_full_turn_speedup_dense_over_pruned": dense_total / pruned_total,
        "mac_per_character": {"dense": 418200, "pruned": 274200},
        "weights": {"dense": 63150, "pruned": 37950},
        "mac_reduction_fraction": 1 - 274200 / 418200,
        "weight_reduction_fraction": 1 - 37950 / 63150,
        "memory_bit_reduction_fraction": 1 - summaries["pruned"]["quartus_postfit"]["memory_bits"]["used"] /
                                             summaries["dense"]["quartus_postfit"]["memory_bits"]["used"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparison": result["comparison"],
                      "models": {name: {key: value[key] for key in (
                          "fpga_correct_characters", "fpga_exact_plates", "cycles_per_character",
                          "compute_ms_per_character_at_50mhz", "jtag_load_ms", "jtag_total_ms", "quartus_postfit")}
                                for name, value in summaries.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
