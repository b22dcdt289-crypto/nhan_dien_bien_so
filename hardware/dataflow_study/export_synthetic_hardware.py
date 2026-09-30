"""Export held-out generated glyphs and the two new INT8 LeNet checkpoints.

Only the generated 2,000-plate dataset is opened. The output directory contains
local JTAG test labels and image bytes and must stay out of public source control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from train_synthetic_car_compare import load_synthetic, sha


SEED = 20261003
LAYERS = ("features.0", "features.3", "classifier.0", "classifier.2", "classifier.4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/car_frontal_synthetic_2x1000_v1"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/car2000_synthetic_only_20261001_v2"))
    parser.add_argument("--out-root", type=Path,
                        default=Path("hardware/dataflow_study/rtl_eval/synthetic_only_2000_v2"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plates-per-layout", type=int, default=3,
                        help="Number of held-out test plates to include for each layout.")
    parser.add_argument("--run-tcl", type=Path,
                        default=Path("hardware/de10_lite_ocr_compare/run_variant.tcl"),
                        help="Tcl runner to copy into the isolated hardware run directory.")
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    for name in ("dense_stream", "pruned_stream"):
        generated = args.out_root / name / "generated"
        if generated.exists() and any(generated.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing exported vectors: {generated}")
    torch.set_num_threads(4)
    splits, _source = load_synthetic(args.data, args.seed)
    selected = []
    if not 1 <= args.plates_per_layout <= 100:
        raise ValueError("plates-per-layout must be between 1 and 100")
    for layout in ("one_row_car", "two_row_car"):
        candidates = [plate for plate in splits["test"] if plate["record"]["layout"] == layout]
        candidates.sort(key=lambda plate: hashlib.sha256(
            f"board|{args.seed}|{plate['record']['label']}".encode()).hexdigest())
        if len(candidates) < args.plates_per_layout:
            raise RuntimeError(f"Insufficient held-out {layout} plates: {len(candidates)}")
        selected.extend(candidates[:args.plates_per_layout])
    expected_plates = 2 * args.plates_per_layout
    if len(selected) != expected_plates or len({plate["record"]["label"] for plate in selected}) != expected_plates:
        raise RuntimeError("Unexpected number of distinct held-out synthetic plates")

    images = np.stack([crop for plate in selected for crop in plate["crops"]]).astype(np.uint8)
    labels = [char for plate in selected for char in plate["record"]["label"]]
    if len(images) != expected_plates * 8 or any(len(plate["record"]["label"]) != 8 for plate in selected):
        raise RuntimeError("Expected 8 OCR glyphs per selected plate")
    image_int8 = np.rint((images.astype(np.float32) * (2.0 / 255.0) - 1.0) * 127).astype(np.int8)
    quant_input = torch.from_numpy(image_int8.astype(np.float32)).unsqueeze(1)
    lut = [int(np.rint(127 * np.tanh(index / 32))) for index in range(-128, 129)]
    label_rows = []
    offset = 0
    for plate in selected:
        text = plate["record"]["label"]
        label_rows.append({"layout": plate["record"]["layout"], "truth": text,
                           "indices": list(range(offset, offset + len(text)))})
        offset += len(text)

    summaries = {}
    for name, model_type in (("dense", LeNet5), ("pruned", LeNet5Conv2Pruned)):
        checkpoint_path = args.models / f"{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = model_type(len(CLASS_NAMES)).eval()
        model.load_state_dict(checkpoint["model"])
        fp_input = torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
        with torch.inference_mode():
            fp32_ids = model(fp_input).argmax(1).tolist()
        specs, weights, biases, shifts = quantize(checkpoint["model"])
        fixed_ids = fixed_forward(quant_input, specs, weights, biases, lut).argmax(1).tolist()
        generated = args.out_root / f"{name}_stream" / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        write_hex(generated / "images.hex", image_int8.flatten().tolist(), 8)
        write_hex(generated / "weights.hex", weights, 8)
        write_hex(generated / "biases.hex", biases, 32)
        write_hex(generated / "tanh.hex", lut, 8)
        write_hex(generated / "shifts.hex", shifts, 8)
        manifest = {
            "architecture": name, "streaming_input": True,
            "dataset_manifest_sha256": sha(args.data / "manifest.csv"),
            "checkpoint_sha256": sha(checkpoint_path), "seed": args.seed,
            "source": f"{expected_plates} held-out generated frontal car plates, {args.plates_per_layout} per layout",
            "real_photos_read": False, "plate_count": len(label_rows),
            "character_count": len(labels), "class_order": CLASS_NAMES,
            "layers": specs, "weight_count": len(weights), "bias_count": len(biases),
            "shifts": shifts, "labels": labels,
            "fp32_predictions": [CLASS_NAMES[index] for index in fp32_ids],
            "fixed_predictions": [CLASS_NAMES[index] for index in fixed_ids],
            "fp32_character_correct": sum(a == b for a, b in zip(labels, [CLASS_NAMES[i] for i in fp32_ids])),
            "fixed_character_correct": sum(a == b for a, b in zip(labels, [CLASS_NAMES[i] for i in fixed_ids])),
            "fp32_plate_correct": sum("".join(CLASS_NAMES[fp32_ids[i]] for i in row["indices"]) == row["truth"] for row in label_rows),
            "fixed_plate_correct": sum("".join(CLASS_NAMES[fixed_ids[i]] for i in row["indices"]) == row["truth"] for row in label_rows),
            "plates": label_rows,
        }
        (generated / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[name] = {key: manifest[key] for key in (
            "architecture", "dataset_manifest_sha256", "checkpoint_sha256", "plate_count",
            "character_count", "weight_count", "bias_count", "fp32_character_correct",
            "fixed_character_correct", "fp32_plate_correct", "fixed_plate_correct")}
    # Copy the in-system source/probe Tcl so its output and image paths resolve
    # inside this isolated local hardware run directory.
    (args.out_root).mkdir(parents=True, exist_ok=True)
    source_tcl = args.run_tcl if args.run_tcl.is_absolute() else ROOT / args.run_tcl
    (args.out_root / "run_variant.tcl").write_text(source_tcl.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(args.out_root), "models": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
