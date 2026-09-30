"""Reproducible analytical bounds for the two LeNet-5 shapes on DE10-Lite.

This does not claim RS/WS/OS were post-fit or measured on the board. The
stationary choice affects local data movement, not the network's MAC count.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def describe(conv2: int, pe_count: int, clock_hz: int) -> dict:
    layer_macs = {
        "conv1": 6 * 25 * 28 * 28,
        "conv2": 6 * conv2 * 25 * 10 * 10,
        "fc1": conv2 * 25 * 120,
        "fc2": 120 * 84,
        "fc3": 84 * 30,
    }
    layer_weights = {"conv1": 150, "conv2": 150 * conv2,
                     "fc1": 3000 * conv2, "fc2": 10080, "fc3": 2520}
    layer_biases = {"conv1": 6, "conv2": conv2, "fc1": 120, "fc2": 84, "fc3": 30}
    weights = sum(layer_weights.values())
    biases = sum(layer_biases.values())
    macs = sum(layer_macs.values())
    ideal_cycles = sum(math.ceil(value / pe_count) for value in layer_macs.values())
    buffers = {
        "input_frame_bytes": 32 * 32,
        "conv1_four_line_bytes": 4 * 32,
        "conv1_window_bytes": 25,
        "conv2_four_line_bytes": 4 * 14 * 6,
        "conv2_window_bytes": 25 * 6,
        "conv1_full_feature_map_bytes": 6 * 28 * 28,
        "pool1_full_feature_map_bytes": 6 * 14 * 14,
        "conv2_full_feature_map_bytes": conv2 * 10 * 10,
        "pool2_full_feature_map_bytes": conv2 * 5 * 5,
    }
    return {
        "conv2_channels": conv2,
        "layer_macs": layer_macs, "macs_per_character": macs,
        "macs_per_eight_character_plate": macs * 8,
        "layer_weights": layer_weights, "weights": weights,
        "layer_biases": layer_biases, "biases": biases,
        "parameters": weights + biases,
        "fp32_model_bytes": 4 * (weights + biases),
        "int8_weight_int32_bias_bytes": weights + 4 * biases,
        "int8_weight_int32_bias_bits": 8 * (weights + 4 * biases),
        "buffers": buffers,
        "ideal_pe8_cycles_per_character_no_stalls_or_non_mac_ops": ideal_cycles,
        "ideal_pe8_compute_ms_per_character": 1000 * ideal_cycles / clock_hz,
        "ideal_pe8_compute_ms_per_eight_character_plate": 8000 * ideal_cycles / clock_hz,
        "minimum_input_frame_stream_cycles_at_one_pixel_per_clock": 1024,
        "minimum_input_frame_stream_ms_at_one_pixel_per_clock": 1024 * 1000 / clock_hz,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mixed_car_dense_pruned_20261001/dataflow_bounds.json"))
    parser.add_argument("--pe-count", type=int, default=8)
    parser.add_argument("--clock-hz", type=int, default=50_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.pe_count <= 0 or args.clock_hz <= 0:
        raise ValueError("pe-count and clock-hz must be positive")
    dense = describe(16, args.pe_count, args.clock_hz)
    pruned = describe(8, args.pe_count, args.clock_hz)
    result = {
        "board": {"name": "DE10-Lite / MAX 10 10M50DAF484C7G",
                  "logic_elements_capacity": 49_760,
                  "internal_memory_bits_capacity": 1_677_312,
                  "nine_bit_multiplier_capacity": 288,
                  "clock_assumption_hz": args.clock_hz},
        "model": {"dense": dense, "pruned_conv2_50_percent": pruned},
        "comparison": {
            "mac_reduction_fraction": 1 - pruned["macs_per_character"] / dense["macs_per_character"],
            "weight_reduction_fraction": 1 - pruned["weights"] / dense["weights"],
            "ideal_compute_speedup_pe8": dense["ideal_pe8_cycles_per_character_no_stalls_or_non_mac_ops"] /
                                        pruned["ideal_pe8_cycles_per_character_no_stalls_or_non_mac_ops"],
            "int8_model_bits_fraction_of_board_dense": dense["int8_weight_int32_bias_bits"] / 1_677_312,
            "int8_model_bits_fraction_of_board_pruned": pruned["int8_weight_int32_bias_bits"] / 1_677_312,
        },
        "shared_dataflow_assumptions": {
            "mac_lanes": args.pe_count,
            "weight_bits": 8, "activation_bits": 8, "partial_sum_bits": 32,
            "weight_stationary_local_weight_register_min_bytes": args.pe_count,
            "output_stationary_local_partial_sum_register_min_bytes": 4 * args.pe_count,
            "row_stationary_local_filter_row_register_min_bytes": 5 * args.pe_count,
            "conv1_input_unique_bytes_if_streamed_once": 32 * 32,
            "conv2_input_unique_bytes_if_streamed_once": 6 * 14 * 14,
            "non_mac_ops_not_in_cycle_bound": ["bias", "tanh LUT", "average pooling", "memory stalls",
                                                "pipeline fill/drain", "control", "JTAG/host transfer"],
            "dataflow_warning": "WS, OS and RS have identical MAC totals. Distinct post-fit area, Fmax, SRAM traffic or latency require separate RTL schedules and synthesis; this file contains no such measurements.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
