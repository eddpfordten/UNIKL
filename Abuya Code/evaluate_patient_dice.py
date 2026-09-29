"""Score the saved project model on the labeled BraTS-PEDs holdout."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import random_split

from brain_tumor_seg.config import SEED, TARGET_SHAPE, VAL_SPLIT
from brain_tumor_seg.data.dataset import BraTSDataset
from brain_tumor_seg.models import MultiTaskUNet3D
from brain_tumor_seg.utils.metrics import dice_coefficient


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1Mhk0XQF7rEZyrBFz0Gf0SWCEJv7QWZ1c"
    r"\3D BRAIN SEGMENTATION\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "own_patient_dice.csv",
    )
    args = parser.parse_args()

    train_dir = args.data_root / "Training"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training cases not found: {train_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")

    case_ids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    dataset = BraTSDataset(train_dir, case_ids=case_ids, target_shape=TARGET_SHAPE)
    val_count = max(1, int(len(dataset) * VAL_SPLIT))
    _, holdout = random_split(
        dataset,
        [len(dataset) - val_count, val_count],
        generator=torch.Generator().manual_seed(SEED),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskUNet3D(in_channels=4, out_channels=1).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["case_id", "dice"])
        writer.writeheader()
        with torch.inference_mode():
            for index in range(len(holdout)):
                sample = holdout[index]
                image = sample["image"].unsqueeze(0).to(device)
                mask = sample["mask"].unsqueeze(0)
                logits = model(image).seg_logits
                dice = dice_coefficient(torch.sigmoid(logits).cpu(), mask)
                scores.append(dice)
                writer.writerow({"case_id": sample["case_id"], "dice": f"{dice:.8f}"})
                file.flush()
                print(f"{index + 1:2d}/{len(holdout)} {sample['case_id']}: {dice:.6f}", flush=True)

    summary = {
        "model": "own",
        "checkpoint": args.checkpoint.name,
        "data_root": args.data_root.name,
        "split": "Training holdout",
        "seed": SEED,
        "patients": len(scores),
        "mean_dice": statistics.fmean(scores),
        "threshold": 0.5,
        "target_shape": list(TARGET_SHAPE),
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
