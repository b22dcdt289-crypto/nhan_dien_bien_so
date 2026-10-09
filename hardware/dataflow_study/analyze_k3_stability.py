"""Compare repeated DE10-Lite JTAG OCR runs against a recorded 100-glyph run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: int(row[key]) for key in (
            "sample_index", "predicted_class", "cycles", "result_seq",
            "jtag_load_ms", "jtag_total_ms")}
            for row in csv.DictReader(handle)]
    if not rows or [row["sample_index"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Incomplete or unordered JTAG result: {path}")
    if any((b["result_seq"] - a["result_seq"]) % 256 != 1
           for a, b in zip(rows, rows[1:])):
        raise ValueError(f"JTAG result sequence discontinuity: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--passes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = read_rows(args.reference)
    passes = [read_rows(path) for path in args.passes]
    if any((current[0]["result_seq"] - previous[-1]["result_seq"]) % 256 != 1
           for previous, current in zip(passes, passes[1:])):
        raise ValueError("JTAG result sequence is discontinuous between repeat passes")
    summaries = []
    for path, rows in zip(args.passes, passes):
        if len(rows) != len(reference):
            raise ValueError(f"Sample count differs from reference: {path}")
        prediction_mismatches = [i for i, (a, b) in enumerate(zip(reference, rows))
                                 if a["predicted_class"] != b["predicted_class"]]
        cycle_mismatches = [i for i, (a, b) in enumerate(zip(reference, rows))
                            if a["cycles"] != b["cycles"]]
        summaries.append({
            "file": str(path),
            "samples": len(rows),
            "prediction_mismatch_indices": prediction_mismatches,
            "cycle_mismatch_indices": cycle_mismatches,
            "median_jtag_total_ms": statistics.median(row["jtag_total_ms"] for row in rows),
        })
    result = {
        "measurement": "Repeated direct USB-Blaster/JTAG runs on DE10-Lite",
        "reference": str(args.reference),
        "reference_samples": len(reference),
        "pass_count": len(summaries),
        "total_tested_characters": sum(row["samples"] for row in summaries),
        "hardware_prediction_mismatches": sum(len(row["prediction_mismatch_indices"])
                                              for row in summaries),
        "hardware_cycle_mismatches": sum(len(row["cycle_mismatch_indices"])
                                         for row in summaries),
        "passes": summaries,
        "scope_note": "Checks repeatability on fixed 32x32 glyphs; not long-term thermal, camera, or full-plate input reliability",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "pass_count", "total_tested_characters", "hardware_prediction_mismatches",
        "hardware_cycle_mismatches")}, ensure_ascii=False))
    if result["hardware_prediction_mismatches"] or result["hardware_cycle_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
