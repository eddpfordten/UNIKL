"""Join the two saved per-patient Dice runs by BraTS case ID."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs" / "evaluation"


def read_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    scores = {row["case_id"]: float(row["dice"]) for row in rows}
    if len(scores) != len(rows):
        raise ValueError(f"Duplicate patient IDs in {path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-scores", type=Path, default=ROOT / "own_patient_dice.csv")
    parser.add_argument("--hayyi-scores", type=Path, default=ROOT / "hayyi_patient_dice.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    own_path = args.own_scores
    hayyi_path = args.hayyi_scores
    output_path = args.output_dir / "paired_patient_dice.csv"
    summary_path = args.output_dir / "paired_dice_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Paired evaluation results already exist")

    own = read_scores(own_path)
    hayyi = read_scores(hayyi_path)
    if own.keys() != hayyi.keys():
        raise ValueError("The two evaluations used different patient IDs")

    own_summary = json.loads(own_path.with_name(own_path.stem + "_summary.json").read_text(encoding="utf-8"))
    hayyi_summary = json.loads(hayyi_path.with_name(hayyi_path.stem + "_summary.json").read_text(encoding="utf-8"))
    if own_summary["patients"] != len(own) or hayyi_summary["patients"] != len(hayyi):
        raise ValueError("Patient counts in the run summaries do not match their CSVs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "own_dice", "hayyi_dice", "own_minus_hayyi"],
        )
        writer.writeheader()
        for case_id in sorted(own):
            writer.writerow(
                {
                    "case_id": case_id,
                    "own_dice": f"{own[case_id]:.8f}",
                    "hayyi_dice": f"{hayyi[case_id]:.8f}",
                    "own_minus_hayyi": f"{own[case_id] - hayyi[case_id]:.8f}",
                }
            )

    report = {
        "patients": len(own),
        "split": "38 labeled patients held out from Training; seed 42",
        "own_mean_dice": own_summary["mean_dice"],
        "hayyi_mean_dice": hayyi_summary["mean_dice"],
        "own_minus_hayyi_mean_dice": own_summary["mean_dice"] - hayyi_summary["mean_dice"],
        "own_higher_cases": sum(own[case_id] > hayyi[case_id] for case_id in own),
        "hayyi_higher_cases": sum(hayyi[case_id] > own[case_id] for case_id in own),
        "radiomics_caveat": (
            "Hayyi's supplied radiomics CSV was extracted with ground-truth whole-tumor "
            "masks on the held-out patients, which leaks validation label information."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Patient scores: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
