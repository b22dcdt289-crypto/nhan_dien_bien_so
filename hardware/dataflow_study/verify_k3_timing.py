"""Gate DE10-Lite OCR bitstreams on actual 50 MHz post-fit timing reports.

The extra Fmax headroom is a project policy, not a guarantee of operation.
This script never changes the board clock, RTL, SDC, or Quartus Fmax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


VARIANTS = ("dense", "pruned")
CORNER = "Slow 1200mV 85C"
CLOCK = "MAX10_CLK1_50"
BOARD_CLOCK_MHZ = 50.0
UNCONSTRAINED_ROWS = (
    "Illegal Clocks",
    "Unconstrained Clocks",
    "Unconstrained Input Ports",
    "Unconstrained Input Port Paths",
    "Unconstrained Output Ports",
    "Unconstrained Output Port Paths",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_slack_summary(text: str) -> dict[str, dict[str, float]]:
    rows = re.findall(
        r"Type\s*:\s*([^\r\n]+)\s*\r?\n"
        r"Slack\s*:\s*(-?\d+(?:\.\d+)?)\s*\r?\n"
        r"TNS\s*:\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    if not rows:
        raise ValueError("Timing slack summary is missing")
    return {name: {"slack_ns": float(slack), "tns_ns": float(tns)}
            for name, slack, tns in rows}


def parse_fmax(text: str) -> tuple[float, float]:
    start = "; Slow 1200mV 85C Model Fmax Summary"
    end = "; Slow 1200mV 85C Model Setup Summary"
    if start not in text or end not in text:
        raise ValueError("Slow 85C Fmax section is missing")
    section = text.split(start, 1)[1].split(end, 1)[0]
    row = re.search(
        r";\s*([0-9.]+) MHz\s*;\s*([0-9.]+) MHz\s*;\s*MAX10_CLK1_50\s*;",
        section,
    )
    if not row:
        raise ValueError("50 MHz board clock Fmax row is missing")
    return float(row.group(1)), float(row.group(2))


def verify_variant(root: Path, name: str, expected_sof_sha: str,
                   minimum_fmax_mhz: float, minimum_setup_slack_ns: float) -> dict:
    project = root / f"{name}_stream"
    qsf = (project / "OcrBench.qsf").read_text(encoding="utf-8")
    if "SDC_FILE ../OcrBench_50MHz_board.sdc" not in qsf or \
       "PIN_P11 -to MAX10_CLK1_50" not in qsf or \
       "DEVICE 10M50DAF484C7G" not in qsf or \
       f"TOP_LEVEL_ENTITY Bench_{name}_stream_25" not in qsf:
        raise ValueError(f"{name}: project does not target the checked DE10-Lite SDC/pin/device")
    report = (project / "output_files/OcrBench.sta.rpt").read_text(
        encoding="utf-8", errors="replace")
    summary = (project / "output_files/OcrBench.sta.summary").read_text(
        encoding="utf-8", errors="replace")
    sof = project / "output_files/OcrBench.sof"
    actual_sof_sha = sha256(sof)
    if actual_sof_sha != expected_sof_sha:
        raise ValueError(f"{name}: .sof SHA-256 does not match JTAG result record")

    fmax, restricted_fmax = parse_fmax(report)
    slacks = parse_slack_summary(summary)
    setup_key = f"{CORNER} Model Setup '{CLOCK}'"
    hold_key = f"{CORNER} Model Hold '{CLOCK}'"
    for key in (setup_key, hold_key):
        if key not in slacks:
            raise ValueError(f"{name}: missing {key}")
    unconstrained = {}
    for label in UNCONSTRAINED_ROWS:
        row = re.search(rf";\s*{re.escape(label)}\s*;\s*(\d+)\s*;\s*(\d+)\s*;", report)
        if not row:
            raise ValueError(f"{name}: missing unconstrained-path row: {label}")
        unconstrained[label] = {"setup": int(row.group(1)), "hold": int(row.group(2))}

    failures = []
    if fmax < minimum_fmax_mhz or restricted_fmax < minimum_fmax_mhz:
        failures.append(f"Fmax {fmax}/{restricted_fmax:.2f} < {minimum_fmax_mhz:.2f} MHz")
    if slacks[setup_key]["slack_ns"] < minimum_setup_slack_ns:
        failures.append(f"setup slack < {minimum_setup_slack_ns:.3f} ns")
    if any(row["slack_ns"] < 0 or row["tns_ns"] != 0 for row in slacks.values()):
        failures.append("a setup/hold/recovery/removal/pulse check has negative slack or TNS")
    if any(value for row in unconstrained.values() for value in row.values()):
        failures.append("unconstrained paths remain")
    if "Design is fully constrained for setup requirements" not in report or \
       "Design is fully constrained for hold requirements" not in report:
        failures.append("Quartus did not declare setup and hold fully constrained")

    return {
        "status": "pass" if not failures else "fail",
        "fmax_mhz_slow_85c": fmax,
        "restricted_fmax_mhz_slow_85c": restricted_fmax,
        "setup_slack_ns_slow_85c": slacks[setup_key]["slack_ns"],
        "hold_slack_ns_slow_85c": slacks[hold_key]["slack_ns"],
        "all_timing_checks": slacks,
        "unconstrained": unconstrained,
        "sof_sha256": actual_sof_sha,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--board-results", type=Path, required=True)
    parser.add_argument("--guardband-mhz", type=float, default=10.0,
                        help="Engineering margin above actual board clock; not a guarantee")
    parser.add_argument("--minimum-setup-slack-ns", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.guardband_mhz < 0 or args.minimum_setup_slack_ns < 0:
        parser.error("guardband and minimum slack must be nonnegative")

    sdc = (args.run_root / "OcrBench_50MHz_board.sdc").read_text(encoding="utf-8")
    if not re.search(r"create_clock\s+-name\s+MAX10_CLK1_50\s+-period\s+20\.000\b", sdc):
        raise ValueError("Production SDC must constrain the physical input to 20 ns")
    if "set_multicycle_path" in sdc:
        raise ValueError("Unexpected multicycle timing exception")
    results = json.loads(args.board_results.read_text(encoding="utf-8"))
    if results.get("physical_input_clock_mhz") != 50 or results.get("effective_core_step_mhz") != 25:
        raise ValueError("Board result metadata does not describe the tested 50/25 MHz design")

    minimum_fmax = BOARD_CLOCK_MHZ + args.guardband_mhz
    checked = {name: verify_variant(
        args.run_root, name, results["models"][name]["sof_sha256"],
        minimum_fmax, args.minimum_setup_slack_ns,
    ) for name in VARIANTS}
    output = {
        "actual_board_clock_mhz": BOARD_CLOCK_MHZ,
        "effective_core_step_mhz": results["effective_core_step_mhz"],
        "engineering_guardband_mhz": args.guardband_mhz,
        "minimum_fmax_mhz": minimum_fmax,
        "minimum_setup_slack_ns": args.minimum_setup_slack_ns,
        "overall": "pass" if all(row["status"] == "pass" for row in checked.values()) else "fail",
        "models": checked,
        "scope_note": "Internal 50 MHz and vendor JTAG clocks only; not camera or plate segmentation timing",
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": output["overall"], "models": {
        name: {key: row[key] for key in ("status", "fmax_mhz_slow_85c",
                                          "setup_slack_ns_slow_85c", "hold_slack_ns_slow_85c",
                                          "failures")}
        for name, row in checked.items()}}, ensure_ascii=False))
    if output["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
