"""Score Hayyi's radiomics model on the same labeled patient holdout."""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split


PROJECT_ROOT = Path(__file__).resolve().parent
BRANCH_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "hayyi_branch" / "Abuya Code"
DEFAULT_DATA_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1Mhk0XQF7rEZyrBFz0Gf0SWCEJv7QWZ1c"
    r"\3D BRAIN SEGMENTATION\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1"
)


def load_radiomics(csv_path: Path, stats_path: Path):
    """Load the original feature rows and the saved train-split normalization."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        feature_names = [name for name in reader.fieldnames if name not in {"case_id", "roi", "split"}]
        if feature_names != stats["feature_names"]:
            raise ValueError("Radiomics CSV columns do not match the saved normalization")
        rows = {row["case_id"]: row for row in reader}

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.size != len(feature_names) or std.size != len(feature_names) or np.any(std <= 0):
        raise ValueError("Invalid radiomics normalization dimensions or standard deviations")
    return rows, feature_names, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hayyi-root", type=Path, default=BRANCH_ROOT)
    parser.add_argument(
        "--radiomics-csv", type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "hayyi_radiomics_features.csv",
    )
    parser.add_argument("--radiomics-stats", type=Path, default=None)
    parser.add_argument(
        "--own-scores", type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "own_patient_dice.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "hayyi_best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "hayyi_patient_dice.csv",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Hayyi checkpoint not found: {args.checkpoint}")
    if not args.hayyi_root.is_dir():
        raise FileNotFoundError(f"Hayyi source folder not found: {args.hayyi_root}")
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")
    train_dir = args.data_root / "Training"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Labeled Training folder not found: {train_dir}")

    # Keep Hayyi's model implementation isolated from this checkout's model.
    sys.path.insert(0, str(args.hayyi_root))
    from brain_tumor_seg.config import SEED, TARGET_SHAPE, VAL_SPLIT
    from brain_tumor_seg.data.dataset import BraTSDataset
    from brain_tumor_seg.models import MultiTaskUNet3D
    from brain_tumor_seg.utils.metrics import dice_coefficient

    feature_path = args.radiomics_csv
    stats_path = args.radiomics_stats or args.hayyi_root / "outputs" / "checkpoints" / "radiomics_stats.json"
    rows, feature_names, mean, std = load_radiomics(feature_path, stats_path)

    case_ids = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    dataset = BraTSDataset(train_dir, case_ids=case_ids, target_shape=TARGET_SHAPE)
    val_count = max(1, int(len(dataset) * VAL_SPLIT))
    _, holdout = random_split(
        dataset,
        [len(dataset) - val_count, val_count],
        generator=torch.Generator().manual_seed(SEED),
    )
    holdout_ids = [case_ids[index] for index in holdout.indices]
    missing = [case_id for case_id in holdout_ids if case_id not in rows]
    if missing:
        raise ValueError(f"Radiomics features missing for held-out cases: {missing}")
    if any(rows[case_id]["roi"] != "whole" for case_id in holdout_ids):
        raise ValueError("Holdout radiomics rows differ from Hayyi's whole-tumor training ROI")

    with args.own_scores.open(newline="", encoding="utf-8") as file:
        own_ids = {row["case_id"] for row in csv.DictReader(file)}
    if set(holdout_ids) != own_ids:
        raise ValueError("Hayyi and own-model holdout patient IDs do not match")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskUNet3D(in_channels=4, out_channels=1, radiomics_dim=len(feature_names))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["case_id", "dice"])
        writer.writeheader()
        with torch.inference_mode():
            for index in range(len(holdout)):
                sample = holdout[index]
                row = rows[sample["case_id"]]
                raw = np.asarray(
                    [float(row[name]) if row[name] else 0.0 for name in feature_names],
                    dtype=np.float32,
                )
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                radiomics = torch.from_numpy((raw - mean) / std).unsqueeze(0).to(device)
                image = sample["image"].unsqueeze(0).to(device)
                mask = sample["mask"].unsqueeze(0)
                logits = model(image, radiomics=radiomics).seg_logits
                dice = dice_coefficient(torch.sigmoid(logits).cpu(), mask)
                scores.append(dice)
                writer.writerow({"case_id": sample["case_id"], "dice": f"{dice:.8f}"})
                file.flush()
                print(f"{index + 1:2d}/{len(holdout)} {sample['case_id']}: {dice:.6f}", flush=True)

    summary = {
        "model": "hayyi_radiomics",
        "checkpoint": args.checkpoint.name,
        "data_root": args.data_root.name,
        "split": "Training holdout",
        "seed": SEED,
        "patients": len(scores),
        "mean_dice": statistics.fmean(scores),
        "threshold": 0.5,
        "target_shape": list(TARGET_SHAPE),
        "radiomics_features": len(feature_names),
        "radiomics_csv": feature_path.name,
        "radiomics_stats": stats_path.name,
        "radiomics_roi": "whole tumor from ground-truth mask",
        "radiomics_note": "Ground-truth ROI leaks validation mask information into segmentation.",
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Mean Dice: {summary['mean_dice']:.6f} across {len(scores)} patients")
    print(f"Results: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
