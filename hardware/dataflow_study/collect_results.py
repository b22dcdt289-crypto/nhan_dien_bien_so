"""Publish aggregate-only results for the frontal-car LeNet-5 comparison.

The generated files contain counts, timings and confusion matrices, not plate
images, per-plate labels, checkpoints or ROM contents. The original measured
artifacts remain under the local (git-ignored) artifacts/ directory.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MIXED = ROOT / "artifacts/mixed_car_dense_pruned_20261001"
SYNTH = ROOT / "artifacts/synthetic_car_dense_pruned_20261001"
CLASSES = list("0123456789ABCDEFGHKLMNPSTUVXYZ")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quartus_variant(name: str) -> dict:
    base = HERE / "rtl_eval" / f"{name}_stream" / "output_files"
    fit = (base / "OcrBench.fit.summary").read_text(encoding="utf-8")
    sta = (base / "OcrBench.sta.rpt").read_text(encoding="utf-8")
    def field(label: str) -> int:
        found = re.search(re.escape(label) + r"\s*:\s*([\d,]+)", fit)
        if not found:
            raise ValueError(f"Missing {label} from {base}")
        return int(found.group(1).replace(",", ""))
    frequency = re.search(r";\s*([\d.]+) MHz\s*;\s*[\d.]+ MHz\s*;\s*MAX10_CLK1_50", sta)
    slack = re.search(r"Worst-case setup slack is\s*([\d.-]+)", sta)
    if not frequency or not slack or "Fitter Status : Successful" not in fit:
        raise ValueError(f"Incomplete Quartus report for {name}")
    return {
        "source": str(base.relative_to(ROOT)).replace("\\", "/"),
        "device": "10M50DAF484C7G", "top": f"Bench_{name}_stream",
        "logic_elements": field("Total logic elements"),
        "dedicated_logic_registers": field("Dedicated logic registers"),
        "memory_bits": field("Total memory bits"),
        "multiplier_9bit_elements": field("Embedded Multiplier 9-bit elements"),
        "fmax_mhz_slow_85c": float(frequency.group(1)),
        "setup_slack_ns_first_85c": float(slack.group(1)),
        "setup_and_hold_fully_constrained": "Design is not fully constrained" not in sta,
        "physical_board_measured": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate_files = {
        "synthetic_only_metrics.json": SYNTH / "metrics.json",
        "mixed_metrics.json": MIXED / "metrics.json",
        "dataflow_bounds.json": MIXED / "dataflow_bounds.json",
        "cpu_interleaved.json": MIXED / "cpu_interleaved.json",
        "int8_real_test.json": MIXED / "int8_real_test.json",
        "dataflow_functional.json": MIXED / "dataflow_functional.json",
        "tensor_profile.json": MIXED / "tensor_profile.json",
        "history.csv": MIXED / "history.csv",
    }
    for name in ("dense", "pruned"):
        for domain in ("real_presegmented", "real_external"):
            filename = f"{name}_{domain}_confusion.csv"
            aggregate_files[filename] = MIXED / filename
    for destination, source in aggregate_files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / destination)

    mixed = read_json(MIXED / "metrics.json")
    synthetic = read_json(SYNTH / "metrics.json")
    bounds = read_json(MIXED / "dataflow_bounds.json")
    cpu = read_json(MIXED / "cpu_interleaved.json")
    int8 = read_json(MIXED / "int8_real_test.json")
    functional = read_json(MIXED / "dataflow_functional.json")
    tensor_profile = read_json(MIXED / "tensor_profile.json")
    postfit = {name: quartus_variant(name) for name in ("dense", "pruned")}
    (OUT / "quartus_postfit.json").write_text(json.dumps(postfit, indent=2), encoding="utf-8")

    with (OUT / "per_character_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["character", "test_support", "dense_correct", "dense_accuracy",
                         "pruned_correct", "pruned_accuracy"])
        matrices = {name: mixed["models"][name]["real_test_presegmented_plate"]["confusion_matrix"]
                    for name in ("dense", "pruned")}
        for index, character in enumerate(CLASSES):
            support = sum(matrices["dense"][index])
            if sum(matrices["pruned"][index]) != support:
                raise ValueError(f"Test support differs for {character}")
            correct = [matrices[name][index][index] for name in ("dense", "pruned")]
            writer.writerow([character, support, correct[0], correct[0] / support if support else "",
                             correct[1], correct[1] / support if support else ""])

    with (OUT / "confusion_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "true", "predicted", "count"])
        for name in ("dense", "pruned"):
            matrix = mixed["models"][name]["real_test_presegmented_plate"]["confusion_matrix"]
            for i, row in enumerate(matrix):
                for j, count in enumerate(row):
                    if i != j and count:
                        writer.writerow([name, CLASSES[i], CLASSES[j], count])

    summary = {
        "scope": "2,000 frontal rendered car plates (1,000 one-row + 1,000 two-row); mixed fine-tune also uses disjoint real train(1) plates",
        "data": mixed["run"]["data"],
        "checkpoints_sha256": {name: mixed["models"][name]["checkpoint_sha256"]
                               for name in ("dense", "pruned")},
        "models": {},
        "analytical_comparison": bounds["comparison"],
        "quartus_postfit": postfit,
        "functional_schedule_verification": functional,
        "tensor_profile": tensor_profile,
        "interpretation": {
            "real_prepared_test": "OCR only; 32x32 glyph crops prepared using label count",
            "real_external_test": "photo -> annotated plate bbox -> perspective correction -> unconstrained segmentation -> OCR; not full-frame detection",
            "streaming": "frame input over JTAG; not line-buffer pixel-stream convolution",
            "stationary": "WS/OS/RS CPU schedules verified on four glyphs; existing serial RTL keeps one output partial sum; separate RS/WS/PE8 RTL not synthesized",
            "fpga_timing": "post-fit estimate only; timing report warns not fully constrained; no new JTAG board measurement",
        },
    }
    for name in ("dense", "pruned"):
        model = mixed["models"][name]
        summary["models"][name] = {
            "best_real_validation_character_accuracy": model["best_real_val_char_accuracy"],
            "best_epoch": model["best_epoch"],
            "synthetic_only_real_external_plate_exact": synthetic["models"][name]["real_annotated_box_to_text"]["total"]["plate_exact_accuracy"],
            "synthetic_test_character_accuracy": model["synthetic_test_crop"]["accuracy"],
            "prepared_real_test": model["real_test_presegmented_plate"]["total"],
            "prepared_real_by_layout": model["real_test_presegmented_plate"]["by_layout"],
            "real_external_test": model["real_external_annotated_box_to_text"]["total"],
            "real_external_by_layout": model["real_external_annotated_box_to_text"]["by_layout"],
            "real_external_failures": model["real_external_annotated_box_to_text"]["failures"],
            "size_and_macs": model["model_size"],
            "cpu_forward_latency": cpu["latency"][name],
            "int8_reference_prepared_test": int8["models"][name],
        }
    (OUT / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(aggregate_files) + 4} aggregate files to {OUT}")


if __name__ == "__main__":
    main()
