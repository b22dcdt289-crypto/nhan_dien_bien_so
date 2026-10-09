"""Audit actual JTAG output and post-fit Quartus reports for the 3-layout run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware.dataflow_study.analyze_synthetic_board import fit_metrics, sha
from train_lenet5 import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("hardware/dataflow_study/rtl_eval/three_layout_20261009"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/three_layout_dense_pruned_20261009/board_results.json"))
    parser.add_argument("--core-enable-divisor", type=int, choices=(1, 2), default=1,
                        help="Core state advances once per N physical 50 MHz edges")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"measurement": "Direct USB-Blaster/JTAG on DE10-Lite, 50 MHz board clock",
              "physical_input_clock_mhz": 50,
              "core_enable_divisor": args.core_enable_divisor,
              "effective_core_step_mhz": 50 / args.core_enable_divisor,
              "cycle_counter_unit": "physical 50 MHz rising edges",
              "hardware_dataflow": "serial output-accumulating MAC with streamed 32x32 glyph input",
              "ws_rs_rtl_measured": False, "models": {}}
    for name in ("dense", "pruned"):
        root = args.run / f"{name}_stream"
        manifest = json.loads((root / "generated/manifest.json").read_text(encoding="utf-8"))
        with (root / "generated/board_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != manifest["character_count"] or [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"Incomplete board output for {name}")
        seq = [int(row["result_seq"]) for row in rows]
        if any((b - a) % 256 != 1 for a, b in zip(seq, seq[1:])):
            raise RuntimeError(f"JTAG sequence discontinuity for {name}")
        observed = [CLASS_NAMES[int(row["predicted_class"])] for row in rows]
        fixed = manifest["fixed_predictions"]
        truth = manifest["labels"]
        groups = defaultdict(lambda: {"plates": 0, "exact": 0, "characters": 0, "correct": 0})
        for plate in manifest["plates"]:
            text = "".join(observed[i] for i in plate["indices"])
            group = groups[f"{plate['domain']}|{plate['layout']}"]
            group["plates"] += 1
            group["exact"] += int(text == plate["truth"])
            group["characters"] += len(plate["truth"])
            group["correct"] += sum(a == b for a, b in zip(text, plate["truth"]))
        cycles = sorted({int(row["cycles"]) for row in rows})
        postfit = fit_metrics(args.run, name)
        fit_summary = (root / "output_files/OcrBench.fit.summary").read_text(encoding="utf-8", errors="replace")
        fit_report = (root / "output_files/OcrBench.fit.rpt").read_text(encoding="utf-8", errors="replace")
        comb = re.search(r"Total combinational functions\s*:\s*([\d,]+)\s*/\s*([\d,]+)", fit_summary)
        m9k = re.search(r";\s*M9Ks\s*;\s*([\d,]+)\s*/\s*([\d,]+)", fit_report)
        if not comb or not m9k:
            raise RuntimeError("Missing Quartus combinational-function or M9K counts")
        postfit["combinational_functions"] = {"used": int(comb.group(1).replace(",", "")),
                                               "available": int(comb.group(2).replace(",", ""))}
        postfit["m9k_blocks"] = {"used": int(m9k.group(1).replace(",", "")),
                                 "available": int(m9k.group(2).replace(",", ""))}
        result["models"][name] = {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "sof_sha256": sha(root / "output_files/OcrBench.sof"),
            "character_count": len(truth), "character_correct": sum(a == b for a, b in zip(observed, truth)),
            "fixed_reference_agreement": sum(a == b for a, b in zip(observed, fixed)),
            "plate_count": len(manifest["plates"]),
            "plate_exact": sum(group["exact"] for group in groups.values()),
            "by_domain_layout": dict(groups),
            "cycles_per_character_unique": cycles,
            "compute_ms_per_character_at_50mhz": [c / 50_000 for c in cycles],
            "jtag_total_ms_median": statistics.median(int(row["jtag_total_ms"]) for row in rows),
            "quartus_postfit": postfit,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": result["models"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
