"""
Batch 3D PyRadiomics extraction over BraTS-PEDs Training / Validation.

Each case folder is processed independently. Training uses the ground-truth
seg as the ROI; Validation has no seg, so a brain mask is built from the
MRI unless you pass predicted masks.
"""
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from brain_tumor_seg.config import RADIOMICS_DIR, RADIOMICS_MODALITIES, TRAIN_DIR, VAL_DIR
from brain_tumor_seg.data.dataloader import get_train_val_case_ids
from brain_tumor_seg.data.dataset import _list_case_ids
from brain_tumor_seg.radiomics.extractor import build_3d_extractor, extract_case_features
from brain_tumor_seg.utils.helpers import ensure_dir

# Elapsed / remaining / rate so long BraTS runs show a usable ETA.
_TQDM_BAR = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def extract_folder_features(
    data_dir: Path,
    split: str,
    modalities: Optional[Sequence[str]] = None,
    roi: str = "whole",
    max_cases: Optional[int] = None,
    case_ids: Optional[Iterable[str]] = None,
    extractor=None,
) -> List[Dict]:
    """
    Run 3D PyRadiomics on every case folder inside data_dir.

    Args:
        data_dir:   Training or Validation root (one subfolder per case).
        split:      Label stored only for logging ('training' / 'validation').
        modalities: MRI channels to extract.
        roi:        Tumor ROI name, or 'brain' for Validation (no seg).
        max_cases:  Optional cap, useful for a notebook smoke test.
        case_ids:   Optional explicit list; default is every folder.
        extractor:  Reuse one extractor across the folder.

    Returns:
        List of per-case feature dicts (failed cases are skipped with a print).
    """
    data_dir = Path(data_dir)
    ids = list(case_ids) if case_ids is not None else _list_case_ids(data_dir)
    if max_cases is not None:
        ids = ids[: max(0, max_cases)]

    if extractor is None:
        extractor = build_3d_extractor()

    channels = list(modalities) if modalities is not None else list(RADIOMICS_MODALITIES)
    rows: List[Dict] = []
    n_fail = 0

    # One bar per case (timer + ETA). Inner modality bars are off so the ETA
    # stays on the case rate, which is what you care about for a full fold.
    progress = tqdm(
        ids,
        desc=f"PyRadiomics {split}",
        unit="case",
        bar_format=_TQDM_BAR,
        dynamic_ncols=True,
    )
    for case_id in progress:
        progress.set_postfix_str(case_id, refresh=False)
        try:
            row = extract_case_features(
                case_id,
                case_dir=data_dir / case_id,
                modalities=channels,
                roi=roi,
                extractor=extractor,
                show_progress=False,
            )
            row["split"] = split
            rows.append(row)
        except Exception as error:
            n_fail += 1
            progress.write(f"Skipping {case_id}: {error}")

    if n_fail:
        print(f"Finished {split}: {len(rows)} ok, {n_fail} skipped.")
    return rows


def extract_training_features(
    train_dir: Path = TRAIN_DIR,
    max_cases: Optional[int] = None,
    roi: str = "whole",
    modalities: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """3D PyRadiomics over the Training folder (ground-truth tumor mask)."""
    return extract_folder_features(
        train_dir,
        split="training",
        modalities=modalities,
        roi=roi,
        max_cases=max_cases,
    )


def extract_train_val_split_features(
    train_dir: Path = TRAIN_DIR,
    modalities: Optional[Sequence[str]] = None,
    roi: str = "whole",
    extractor=None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    PyRadiomics on the same 219 / 38 hold-out used by create_train_dataloader.

    Both splits are folders under Training, so every case has a ground-truth
    -seg.nii.gz. roi defaults to 'whole' (tumor mask), not brain.

    Returns:
        (train_rows, val_rows)
    """
    train_ids, val_ids = get_train_val_case_ids(train_dir)
    print(
        f"PyRadiomics split (tumor ROI): {len(train_ids)} train / "
        f"{len(val_ids)} val — all 4 MRI volumes per case"
    )

    if extractor is None:
        extractor = build_3d_extractor()

    train_rows = extract_folder_features(
        train_dir,
        split="training",
        modalities=modalities,
        roi=roi,
        case_ids=train_ids,
        extractor=extractor,
    )
    val_rows = extract_folder_features(
        train_dir,
        split="validation",
        modalities=modalities,
        roi=roi,
        case_ids=val_ids,
        extractor=extractor,
    )
    return train_rows, val_rows


def extract_validation_features(
    val_dir: Path = VAL_DIR,
    max_cases: Optional[int] = None,
    roi: str = "brain",
    modalities: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """
    3D PyRadiomics over the official Validation folder.

    Those cases have no -seg.nii.gz, so the default ROI is the brain mask
    built from non-zero MRI voxels. Pass predicted tumor masks into the
    case folders as `{case_id}-seg.nii.gz` and set roi='whole' to measure
    the tumor instead.
    """
    return extract_folder_features(
        val_dir,
        split="validation",
        modalities=modalities,
        roi=roi,
        max_cases=max_cases,
    )


def save_features_csv(rows: List[Dict], path: Optional[Path] = None) -> Path:
    """Write a list of feature dicts to CSV (union of keys across rows)."""
    if not rows:
        raise ValueError("No radiomic feature rows to save.")

    path = Path(path) if path is not None else RADIOMICS_DIR / "pyradiomics_3d_features.csv"
    ensure_dir(path.parent)

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} case(s) -> {path}")
    return path
