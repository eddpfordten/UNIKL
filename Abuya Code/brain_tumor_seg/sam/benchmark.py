"""Offline accuracy report for Step 1 and single-click MedSAM2 masks."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as ndi_label


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]):
    surface_a = np.logical_xor(a, binary_erosion(a))
    surface_b = np.logical_xor(b, binary_erosion(b))
    distance_to_a = distance_transform_edt(~surface_a, sampling=spacing)
    distance_to_b = distance_transform_edt(~surface_b, sampling=spacing)
    return np.concatenate((distance_to_b[surface_a], distance_to_a[surface_b]))


def binary_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Calculate lesion-mask metrics, including unrelated-component leakage."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} versus {target.shape}")
    pred_count, target_count = int(predicted.sum()), int(target.sum())
    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    dice = 1.0 if pred_count + target_count == 0 else 2.0 * intersection / (pred_count + target_count)
    iou = 1.0 if union == 0 else intersection / union
    volume_error = abs(pred_count - target_count) / max(target_count, 1)

    if pred_count == 0 and target_count == 0:
        hd95 = 0.0
    elif pred_count == 0 or target_count == 0:
        hd95 = float("inf")
    else:
        hd95 = float(np.percentile(_surface_distances(predicted, target, spacing), 95))

    components, count = ndi_label(predicted)
    unrelated = 0
    for component_id in range(1, count + 1):
        component = components == component_id
        if not np.any(np.logical_and(component, target)):
            unrelated += int(component.sum())
    leakage = unrelated / max(pred_count, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "hd95": hd95,
        "volume_error": float(volume_error),
        "unrelated_leakage": float(leakage),
    }


def _nifti_map(directory: Path) -> dict[str, Path]:
    paths = list(directory.rglob("*.nii")) + list(directory.rglob("*.nii.gz"))
    return {path.name.removesuffix(".gz").removesuffix(".nii"): path for path in paths}


def _load_binary(path: Path):
    image = nib.load(str(path))
    return image.get_fdata() > 0, tuple(float(value) for value in image.header.get_zooms()[:3])


def run_benchmark(ground_truth_dir: Path, step1_dir: Path, refined_dir: Path) -> dict:
    """Compare masks with matching filenames in three directories."""
    ground_truth = _nifti_map(ground_truth_dir)
    step1 = _nifti_map(step1_dir)
    refined = _nifti_map(refined_dir)
    case_ids = sorted(set(ground_truth) & set(step1) & set(refined))
    if not case_ids:
        raise ValueError("No matching NIfTI filenames were found in all three directories")

    cases = []
    for case_id in case_ids:
        target, spacing = _load_binary(ground_truth[case_id])
        step1_mask, _ = _load_binary(step1[case_id])
        refined_mask, _ = _load_binary(refined[case_id])
        cases.append({
            "case_id": case_id,
            "step1": binary_metrics(step1_mask, target, spacing),
            "refined": binary_metrics(refined_mask, target, spacing),
        })

    step1_dice = np.array([case["step1"]["dice"] for case in cases])
    refined_dice = np.array([case["refined"]["dice"] for case in cases])
    refined_leakage = np.array([case["refined"]["unrelated_leakage"] for case in cases])
    degraded = np.mean((step1_dice - refined_dice) > 0.05)
    improvement = float(np.median(refined_dice) - np.median(step1_dice))
    summary = {
        "cases": len(cases),
        "step1_median_dice": float(np.median(step1_dice)),
        "refined_median_dice": float(np.median(refined_dice)),
        "median_dice_improvement": improvement,
        "median_unrelated_leakage": float(np.median(refined_leakage)),
        "fraction_degraded_over_0_05_dice": float(degraded),
        "passes_release_gate": bool(
            improvement >= 0.03
            and np.median(refined_leakage) < 0.01
            and degraded < 0.05
        ),
    }
    return {"summary": summary, "cases": cases}


def main():
    parser = argparse.ArgumentParser(
        description="Compare selected-target Step 1 and MedSAM2 masks with held-out ground truth."
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--step1-dir", type=Path, required=True)
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(args.ground_truth_dir, args.step1_dir, args.refined_dir)
    rendered = json.dumps(report, indent=2, allow_nan=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
