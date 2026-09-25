"""Export, statically quantize, and score the saved segmentation model.

The labeled holdout is the same seeded Training split used by
evaluate_patient_dice.py. The official Validation folder has no masks.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch.utils.data import random_split

from brain_tumor_seg.config import SEED, TARGET_SHAPE, VAL_SPLIT
from brain_tumor_seg.data.dataset import BraTSDataset
from brain_tumor_seg.models import MultiTaskUNet3D
from brain_tumor_seg.utils.metrics import dice_coefficient


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DEFAULT_DATA_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1Mhk0XQF7rEZyrBFz0Gf0SWCEJv7QWZ1c"
    r"\3D BRAIN SEGMENTATION\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainingReader(CalibrationDataReader):
    def __init__(self, training_subset, count):
        self.training_subset = training_subset
        self.count = count
        self.position = 0

    def get_next(self):
        if self.position >= self.count:
            return None
        sample = self.training_subset[self.position]
        self.position += 1
        print(f"Calibration {self.position}/{self.count}: {sample['case_id']}", flush=True)
        return {"image": sample["image"].unsqueeze(0).numpy().astype(np.float32)}

    def rewind(self):
        self.position = 0


def make_split(data_root):
    train_dir = data_root / "Training"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training cases not found: {train_dir}")
    case_ids = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    dataset = BraTSDataset(train_dir, case_ids=case_ids, target_shape=TARGET_SHAPE)
    val_count = max(1, int(len(dataset) * VAL_SPLIT))
    return random_split(
        dataset,
        [len(dataset) - val_count, val_count],
        generator=torch.Generator().manual_seed(SEED),
    )


def export_model(checkpoint, fp32_path):
    if fp32_path.exists():
        raise FileExistsError(fp32_path)
    model = MultiTaskUNet3D(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.segmentation.eval()
    example = torch.zeros((1, 4, *TARGET_SHAPE), dtype=torch.float32)
    torch.onnx.export(
        model.segmentation,
        (example,),
        fp32_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(fp32_path))
    print(f"Exported {fp32_path}", flush=True)


def quantize_model(fp32_path, quant_path, training_subset, calibration_count):
    if quant_path.exists():
        raise FileExistsError(quant_path)
    reader = TrainingReader(training_subset, calibration_count)
    quantize_static(
        fp32_path,
        quant_path,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],
    )
    sanitize_onnx(quant_path)
    quantized = onnx.load(str(quant_path))
    onnx.checker.check_model(quantized)
    qdq_count = sum(node.op_type == "QuantizeLinear" for node in quantized.graph.node)
    if qdq_count == 0:
        raise RuntimeError("Quantized model contains no QuantizeLinear nodes")
    print(f"Quantized {quant_path} ({qdq_count} QuantizeLinear nodes)", flush=True)
    return qdq_count


def sanitize_onnx(path):
    """Remove exporter stack traces containing the build machine's local path."""
    model = onnx.load(str(path))
    removed = 0
    for node in model.graph.node:
        remaining = [item for item in node.metadata_props if item.key != "pkg.torch.onnx.stack_trace"]
        removed += len(node.metadata_props) - len(remaining)
        node.ClearField("metadata_props")
        node.metadata_props.extend(remaining)
    if removed:
        clean_path = path.with_name(path.stem + ".clean.onnx")
        if clean_path.exists():
            raise FileExistsError(clean_path)
        onnx.save(model, str(clean_path))
        onnx.checker.check_model(str(clean_path))
        clean_path.replace(path)
    print(f"Removed {removed} local-path stack traces from {path.name}", flush=True)


def load_step1_scores(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    scores = {row["case_id"]: float(row["dice"]) for row in rows}
    if len(scores) != len(rows):
        raise ValueError("Duplicate patient ID in Step 1 scores")
    return scores


def score_models(fp32_path, quant_path, holdout, step1_scores):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    fp32_session = ort.InferenceSession(str(fp32_path), sess_options=options, providers=["CPUExecutionProvider"])
    quant_session = ort.InferenceSession(str(quant_path), sess_options=options, providers=["CPUExecutionProvider"])
    rows = []
    for index in range(len(holdout)):
        sample = holdout[index]
        case_id = sample["case_id"]
        if case_id not in step1_scores:
            raise ValueError(f"Holdout differs from Step 1: {case_id}")
        image = sample["image"].unsqueeze(0).numpy().astype(np.float32)
        mask = sample["mask"].unsqueeze(0)
        fp32_logits = fp32_session.run(["logits"], {"image": image})[0]
        quant_logits = quant_session.run(["logits"], {"image": image})[0]
        fp32_dice = dice_coefficient(torch.sigmoid(torch.from_numpy(fp32_logits)), mask)
        quant_dice = dice_coefficient(torch.sigmoid(torch.from_numpy(quant_logits)), mask)
        rows.append({
            "case_id": case_id,
            "pytorch_dice": step1_scores[case_id],
            "onnx_fp32_dice": fp32_dice,
            "onnx_int8_dice": quant_dice,
            "int8_minus_pytorch": quant_dice - step1_scores[case_id],
        })
        print(f"{index + 1:2d}/{len(holdout)} {case_id}: FP32 {fp32_dice:.6f}, INT8 {quant_dice:.6f}", flush=True)
    if set(step1_scores) != {row["case_id"] for row in rows}:
        raise ValueError("Step 1 and ONNX patient sets do not match")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/best_model.pth")
    parser.add_argument("--step1-scores", type=Path, default=ROOT / "outputs/evaluation/own_patient_dice.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evaluation/onnx_step2")
    parser.add_argument("--calibration-count", type=int, default=16)
    args = parser.parse_args()

    if not args.checkpoint.is_file() or not args.step1_scores.is_file():
        raise FileNotFoundError("The saved checkpoint and Step 1 scores are required")
    fp32_path = args.output_dir / "own_segmentation_fp32.onnx"
    quant_path = args.output_dir / "own_segmentation_int8.onnx"
    csv_path = args.output_dir / "onnx_patient_dice.csv"
    summary_path = args.output_dir / "onnx_dice_summary.json"
    if any(path.exists() for path in (fp32_path, quant_path, csv_path, summary_path)):
        raise FileExistsError("Step 2 output already exists; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    training_subset, holdout = make_split(args.data_root)
    if not 1 <= args.calibration_count <= len(training_subset):
        raise ValueError("Invalid calibration count")
    step1_scores = load_step1_scores(args.step1_scores)
    export_model(args.checkpoint, fp32_path)
    qdq_count = quantize_model(fp32_path, quant_path, training_subset, args.calibration_count)
    rows = score_models(fp32_path, quant_path, holdout, step1_scores)

    with csv_path.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value if key == "case_id" else f"{value:.8f}" for key, value in row.items()})

    summary = {
        "patients": len(rows),
        "split": "Labeled Training holdout; seed 42; same patients as Step 1",
        "data_root": args.data_root.name,
        "calibration_cases": args.calibration_count,
        "calibration_source": "Training subset only",
        "threshold": 0.5,
        "target_shape": list(TARGET_SHAPE),
        "pytorch_mean_dice": statistics.fmean(row["pytorch_dice"] for row in rows),
        "onnx_fp32_mean_dice": statistics.fmean(row["onnx_fp32_dice"] for row in rows),
        "onnx_int8_mean_dice": statistics.fmean(row["onnx_int8_dice"] for row in rows),
        "int8_minus_pytorch_mean_dice": statistics.fmean(row["int8_minus_pytorch"] for row in rows),
        "max_abs_fp32_minus_pytorch_dice": max(abs(row["onnx_fp32_dice"] - row["pytorch_dice"]) for row in rows),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_fp32_bytes": fp32_path.stat().st_size,
        "onnx_int8_bytes": quant_path.stat().st_size,
        "quantize_linear_nodes": qdq_count,
        "quantization": "Static QDQ, signed INT8 activations and weights, per-channel Conv weights",
        "inference_provider": "CPUExecutionProvider",
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {csv_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
