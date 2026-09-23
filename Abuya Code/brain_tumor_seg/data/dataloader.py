"""
DataLoader factory functions.

Creates PyTorch DataLoaders for training, validation, and inference.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from brain_tumor_seg.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FUSE_RADIOMICS,
    NUM_WORKERS,
    PREDICT_SURVIVAL,
    RADIOMICS_FEATURES_CSV,
    RADIOMICS_STATS_FILENAME,
    SEED,
    SURVIVAL_METADATA_PATH,
    SURVIVAL_STATS_FILENAME,
    TARGET_SHAPE,
    TRAIN_DIR,
    VAL_DIR,
    VAL_SPLIT,
)
from brain_tumor_seg.data.dataset import BraTSDataset, BraTSInferenceDataset, _list_case_ids
from brain_tumor_seg.data.survival import SurvivalStats, load_survival_days, select_survival
from brain_tumor_seg.utils.helpers import set_seed


def infer_radiomics_dim(dataset) -> int:
    """
    Radiomics vector width attached to a Dataset / Subset, else 0.

    Lets the notebook size MultiTaskUNet3D(radiomics_dim=...) from the same
    loader the Trainer will see, without changing Trainer(...).train(...).
    """
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset

    table = getattr(current, "radiomics", None)
    if table is None:
        return 0
    return int(table.n_features)


def _subset_case_ids(subset: Subset, all_case_ids: List[str]) -> List[str]:
    """Recover the case IDs behind a random_split Subset."""
    return [all_case_ids[i] for i in subset.indices]


def get_train_val_case_ids(
    train_dir: Path = TRAIN_DIR,
    val_split: float = VAL_SPLIT,
) -> Tuple[List[str], List[str]]:
    """
    Same 219 / 38 hold-out as create_train_dataloader (SEED + VAL_SPLIT).

    Both lists are folder names under the Training directory, so every case
    still has a ground-truth tumor mask. This is not the official Validation
    folder (that one has no -seg.nii.gz).
    """
    set_seed(SEED)
    all_case_ids = _list_case_ids(train_dir)
    val_size = max(1, int(len(all_case_ids) * val_split))
    train_size = len(all_case_ids) - val_size

    # Indices only — avoid constructing BraTSDataset just to recover IDs.
    indices = list(range(len(all_case_ids)))
    train_subset, val_subset = random_split(
        indices,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_ids = [all_case_ids[i] for i in train_subset.indices]
    val_ids = [all_case_ids[i] for i in val_subset.indices]
    return train_ids, val_ids


def _attach_survival(
    full_dataset: BraTSDataset,
    train_ids: List[str],
    val_ids: List[str],
    survival_path: Path,
    stats_path: Optional[Path],
) -> Optional[SurvivalStats]:
    """
    Load survival labels, fit the normalization and attach both to the dataset.

    The stats are fitted on the training cases only; using the held-out cases
    would leak their label distribution into the target scale.

    Returns:
        The fitted SurvivalStats, or None if no usable labels were found.
    """
    try:
        survival = load_survival_days(survival_path)
    except (FileNotFoundError, KeyError) as error:
        print(f"Survival labels unavailable ({error}); training segmentation only.")
        return None

    train_labels = select_survival(train_ids, survival)
    val_labels = select_survival(val_ids, survival)

    if not train_labels:
        print("No survival labels overlap the training split; training segmentation only.")
        return None

    stats = SurvivalStats.from_days(train_labels.values())

    # The Subset objects produced by random_split hold a reference to this same
    # dataset, so attaching here reaches both splits.
    full_dataset.attach_survival(survival, stats)

    if stats_path is not None:
        stats.save(stats_path)

    print(
        f"Survival labels:  {len(train_labels)} train / {len(val_labels)} val "
        f"(of {len(survival)} in the metadata)"
    )
    return stats


def _attach_radiomics(
    full_dataset: BraTSDataset,
    train_ids: List[str],
    val_ids: List[str],
    radiomics_path: Path,
    stats_path: Optional[Path],
) -> Optional[int]:
    """
    Load the offline PyRadiomics CSV, z-score on train IDs, attach to dataset.

    Returns:
        Feature dimensionality, or None if the CSV is missing / unusable.
    """
    path = Path(radiomics_path)
    if not path.is_file():
        print(
            f"Radiomics CSV not found ({path}); survival head trains without fusion. "
            "Run notebook section 3E first."
        )
        return None

    from brain_tumor_seg.radiomics.features import RadiomicsFeatureTable

    table = RadiomicsFeatureTable.from_csv(path)
    table.fit_normalize(train_ids)

    if stats_path is not None:
        table.save_stats(stats_path)

    full_dataset.attach_radiomics(table)

    n_train = sum(1 for case_id in train_ids if case_id in table)
    n_val = sum(1 for case_id in val_ids if case_id in table)
    print(
        f"Radiomics fusion: {table.n_features} features "
        f"({n_train}/{len(train_ids)} train, {n_val}/{len(val_ids)} val covered)"
    )
    return table.n_features


def create_train_dataloader(
    train_dir: Path = TRAIN_DIR,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    predict_survival: bool = PREDICT_SURVIVAL,
    survival_path: Path = SURVIVAL_METADATA_PATH,
    stats_path: Optional[Path] = None,
    fuse_radiomics: bool = FUSE_RADIOMICS,
    radiomics_path: Path = RADIOMICS_FEATURES_CSV,
    radiomics_stats_path: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from the Training folder.

    Because the Validation folder has no ground-truth masks, we hold out
    a fraction of training cases for validation metrics (Dice, loss).

    Args:
        train_dir:            Folder of training case subfolders.
        batch_size:           Samples per batch.
        val_split:            Fraction of cases held out for validation.
        target_shape:         Volume shape after resizing.
        predict_survival:     Attach survival targets for the second head.
        survival_path:        Metadata TSV holding the survival column.
        stats_path:           Where to persist the fitted survival normalization
                              (defaults to CHECKPOINT_DIR/SURVIVAL_STATS_FILENAME).
        fuse_radiomics:       Attach offline PyRadiomics vectors for survival fusion.
        radiomics_path:       CSV from extract_train_val_split_features.
        radiomics_stats_path: Where to persist radiomics mean/std
                              (defaults to CHECKPOINT_DIR/RADIOMICS_STATS_FILENAME).

    Returns:
        (train_loader, val_loader)

    Note:
        Trainer(model, train_loader, val_loader, ...).train(...) is unchanged;
        radiomics ride along on each batch when fuse_radiomics is True.
    """
    set_seed(SEED)

    all_case_ids = _list_case_ids(train_dir)
    full_dataset = BraTSDataset(train_dir, case_ids=all_case_ids, target_shape=target_shape)

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_ids = _subset_case_ids(train_dataset, all_case_ids)
    val_ids = _subset_case_ids(val_dataset, all_case_ids)

    print(f"Training cases:   {len(train_dataset)}")
    print(f"Validation cases: {len(val_dataset)}")

    if predict_survival:
        _attach_survival(
            full_dataset,
            train_ids=train_ids,
            val_ids=val_ids,
            survival_path=survival_path,
            stats_path=stats_path or (CHECKPOINT_DIR / SURVIVAL_STATS_FILENAME),
        )

    if fuse_radiomics:
        _attach_radiomics(
            full_dataset,
            train_ids=train_ids,
            val_ids=val_ids,
            radiomics_path=radiomics_path,
            stats_path=radiomics_stats_path
            or (CHECKPOINT_DIR / RADIOMICS_STATS_FILENAME),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_val_dataloader(
    val_dir: Path = VAL_DIR,
    batch_size: int = BATCH_SIZE,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> DataLoader:
    """
    Create a DataLoader for the official Validation folder (no masks).

    Use this for running inference on unseen cases.
    """
    dataset = BraTSInferenceDataset(val_dir, target_shape=target_shape)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Inference cases (Validation folder): {len(dataset)}")
    return loader


def create_inference_dataloader(
    data_dir: Path = VAL_DIR,
    batch_size: int = BATCH_SIZE,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> DataLoader:
    """Alias for create_val_dataloader — loads cases without masks."""
    return create_val_dataloader(data_dir, batch_size, target_shape)
