"""Export new 3-layout checkpoints, 12 held-out plates, and Quartus projects.

The target RTL is the existing serial, single-MAC, output-accumulating stream
core. This does not create hardware WS or RS architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware.de10_lite_ocr.prepare_demo import fixed_forward, write_hex
from hardware.de10_lite_ocr_compare.export_compare import quantize
from train_cost_sensitive_conv2_distill import LeNet5Conv2Pruned
from train_lenet5 import CLASS_NAMES, LeNet5
from lenet5_k3 import LeNet5K3, LeNet5K3Pruned
from train_three_layout_compare import LAYOUTS, generated_plates, real_plates
from train_synthetic_car_compare import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/three_layout_dense_pruned_20261009"))
    parser.add_argument("--out-root", type=Path,
                        default=Path("hardware/dataflow_study/rtl_eval/three_layout_20261009"))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--plates-per-layout-domain", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    if not 1 <= args.plates_per_layout_domain <= 3:
        raise ValueError("1..3 plates/layout/domain are supported by the 7-bit sample ID")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(args.out_root)
    torch.set_num_threads(4)
    run = json.loads((args.models / "metrics.json").read_text(encoding="utf-8"))
    if run["run"].get("kernel_size", 5) != args.kernel_size:
        raise RuntimeError("Training kernel size and hardware export disagree")
    synth = generated_plates(Path("data/car_frontal_synthetic_2x1000_v1"),
                             Path("data/motorcycle_frontal_synthetic_1000_v2"), args.seed)
    prepared_root = Path("data/independent_chars_train1_structured_channel30_v1")
    real = []
    for layout in LAYOUTS:
        real.extend(real_plates(prepared_root, layout, "test", 100, args.seed, set()))
    selected = []
    for layout in LAYOUTS:
        for domain, rows in (("generated", synth["test"]), ("real_presegmented", real)):
            candidates = [row for row in rows if row["record"]["layout"] == layout]
            candidates.sort(key=lambda row: hashlib.sha256(
                f"board|{args.seed}|{domain}|{row['record']['label']}".encode()).hexdigest())
            selected.extend((domain, row) for row in candidates[:args.plates_per_layout_domain])
    images = np.stack([crop for _, plate in selected for crop in plate["crops"]]).astype(np.uint8)
    labels = [char for _, plate in selected for char in plate["record"]["label"]]
    image_q = np.rint((images.astype(np.float32) / 127.5 - 1.0) * 127).astype(np.int8)
    fp_input = torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
    quant_input = torch.from_numpy(image_q.astype(np.float32)).unsqueeze(1)
    lut = [int(np.rint(127 * np.tanh(index / 32))) for index in range(-128, 129)]
    plate_rows = []
    offset = 0
    for domain, plate in selected:
        truth = plate["record"]["label"]
        plate_rows.append({"domain": domain, "layout": plate["record"]["layout"],
                           "truth": truth, "indices": list(range(offset, offset + len(truth)))})
        offset += len(truth)
    template = Path("hardware/dataflow_study/rtl_eval/synthetic_only_fulltest_20261001")
    args.out_root.mkdir(parents=True)
    shutil.copy2(Path("hardware/de10_lite_ocr_compare/run_variant.tcl"), args.out_root / "run_variant.tcl")
    if args.kernel_size == 5:
        shutil.copy2(template / "BenchVariantsFull.sv", args.out_root / "BenchVariantsFull.sv")
    summaries = {}
    model_classes = (("dense", LeNet5K3), ("pruned", LeNet5K3Pruned)) if args.kernel_size == 3 else (
        ("dense", LeNet5), ("pruned", LeNet5Conv2Pruned))
    for name, cls in model_classes:
        checkpoint_path = args.models / f"{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint["classes"] != CLASS_NAMES or
                checkpoint.get("kernel_size", 5) != args.kernel_size or
                sha(checkpoint_path) != run["models"][name]["checkpoint_sha256"]):
            raise RuntimeError("Checkpoint does not match training record")
        model = cls(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        specs, weights, biases, shifts = quantize(checkpoint["model"])
        with torch.inference_mode():
            fp_ids = model(fp_input).argmax(1).tolist()
            fixed_ids = fixed_forward(quant_input, specs, weights, biases, lut).argmax(1).tolist()
        variant = f"{name}_stream"
        target = args.out_root / variant
        target.mkdir()
        for project_file in ("OcrBench.qsf", "OcrBench.qpf"):
            if args.kernel_size == 3 and project_file == "OcrBench.qsf":
                source_file = Path("hardware/dataflow_study/rtl_templates/k3") / variant / project_file
            else:
                source_file = template / variant / project_file
            shutil.copy2(source_file, target / project_file)
        generated = target / "generated"
        generated.mkdir()
        write_hex(generated / "images.hex", image_q.flatten().tolist(), 8)
        write_hex(generated / "weights.hex", weights, 8)
        write_hex(generated / "biases.hex", biases, 32)
        write_hex(generated / "tanh.hex", lut, 8)
        write_hex(generated / "shifts.hex", shifts, 8)
        manifest = {"architecture": name, "kernel_size": args.kernel_size,
                    "streaming_input": True,
                    "rtl_dataflow": "one serial MAC, output-accumulating",
                    "checkpoint_sha256": sha(checkpoint_path),
                    "dataset_manifest_sha256": run["run"]["manifest_sha256"],
                    "seed": args.seed, "source": "held-out generated and prepared real glyphs",
                    "real_photos_read": True, "plate_count": len(plate_rows),
                    "character_count": len(labels), "class_order": CLASS_NAMES,
                    "layers": specs, "weight_count": len(weights), "bias_count": len(biases),
                    "shifts": shifts, "labels": labels,
                    "fp32_predictions": [CLASS_NAMES[i] for i in fp_ids],
                    "fixed_predictions": [CLASS_NAMES[i] for i in fixed_ids],
                    "fp32_character_correct": sum(CLASS_NAMES[i] == t for i, t in zip(fp_ids, labels)),
                    "fixed_character_correct": sum(CLASS_NAMES[i] == t for i, t in zip(fixed_ids, labels)),
                    "fp32_plate_correct": sum("".join(CLASS_NAMES[fp_ids[i]] for i in p["indices"]) == p["truth"] for p in plate_rows),
                    "fixed_plate_correct": sum("".join(CLASS_NAMES[fixed_ids[i]] for i in p["indices"]) == p["truth"] for p in plate_rows),
                    "plates": plate_rows}
        (generated / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[name] = {key: manifest[key] for key in ("checkpoint_sha256", "plate_count",
                             "character_count", "weight_count", "fp32_character_correct",
                             "fixed_character_correct", "fp32_plate_correct", "fixed_plate_correct")}
    print(json.dumps({"output": str(args.out_root), "models": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
