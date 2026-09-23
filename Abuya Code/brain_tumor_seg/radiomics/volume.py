"""
Load native 3D MRI volumes and ROI masks for PyRadiomics.

This is the module that actually extracts the MRI volume. The training
DataLoader resizes every scan to 96^3 and z-scores it for the U-Net; that
tensor is the wrong input for radiomics. Shape and texture features need
the original NIfTI: native voxel spacing, origin, and raw intensity.

Training cases live under PKG - BraTS-PEDs-v1/BraTS-PEDs-v1/Training.
Validation cases live under .../Validation (no seg mask — a brain ROI is
built from non-zero MRI voxels so PyRadiomics still has a 3D region).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as error:
    raise ImportError(
        "SimpleITK is required to load MRI volumes for PyRadiomics.\n"
        "    py -3.12 -m pip install SimpleITK PyWavelets\n"
        "    py -3.12 -m pip install \"https://github.com/steubk/pyradiomics/releases/download/prebuilt-fix-py312/pyradiomics-3.1.0%2Bpy312fix.363964e-cp312-cp312-win_amd64.whl\""
    ) from error

from brain_tumor_seg.config import MODALITIES, RADIOMICS_LABEL, TRAIN_DIR, VAL_DIR


# BraTS-PEDs tumor subregion labels in the -seg.nii.gz file.
BRATS_LABELS = {
    "ncr": 1,   # necrotic / non-enhancing core
    "ed": 2,    # peritumoral edema
    "et": 4,    # enhancing tumor
}


@dataclass
class RadiomicsVolume:
    """One 3D MRI channel plus the ROI mask, both as SimpleITK images."""

    case_id: str
    modality: str
    # 3D MRI volume extracted for PyRadiomics (native NIfTI, not 96^3).
    image: sitk.Image
    mask: sitk.Image
    image_path: Path
    mask_path: Optional[Path]
    roi: str


def case_dir_for(case_id: str, split: Optional[str] = None) -> Path:
    """
    Resolve a BraTS case folder under Training or Validation.

    Args:
        case_id: Folder name, e.g. 'BraTS-PED-00016-000'.
        split:   'training', 'validation', or None to search both.
    """
    if split == "training":
        path = TRAIN_DIR / case_id
        if not path.is_dir():
            raise FileNotFoundError(f"Training case not found: {path}")
        return path
    if split == "validation":
        path = VAL_DIR / case_id
        if not path.is_dir():
            raise FileNotFoundError(f"Validation case not found: {path}")
        return path

    for folder in (TRAIN_DIR, VAL_DIR):
        path = folder / case_id
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Case {case_id} not found under {TRAIN_DIR} or {VAL_DIR}"
    )


def mri_volume_path(case_dir: Path, case_id: str, modality: str) -> Path:
    """Path to one 3D MRI NIfTI, e.g. BraTS-PED-00016-000-t1c.nii.gz."""
    path = Path(case_dir) / f"{case_id}-{modality}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"MRI volume not found: {path}")
    return path


def seg_mask_path(case_dir: Path, case_id: str) -> Optional[Path]:
    """Path to the tumor seg mask, or None when the folder has no mask."""
    path = Path(case_dir) / f"{case_id}-seg.nii.gz"
    return path if path.is_file() else None


# =============================================================================
# MRI VOLUME EXTRACTION FOR PYRADIOMICS
# This is the code block that loads the 3D MRI volume PyRadiomics consumes.
# sitk.ReadImage keeps spacing / origin / direction; passing a numpy array
# would drop that geometry and make shape features meaningless.
# =============================================================================
def load_mri_volume_for_pyradiomics(
    case_dir: Path,
    case_id: str,
    modality: str,
) -> sitk.Image:
    """
    Extract one 3D MRI volume from a BraTS case folder for PyRadiomics.

    Reads the native `{case_id}-{modality}.nii.gz` (t1c / t1n / t2f / t2w)
    as a SimpleITK Image. Do not use the resized training tensor here.

    Args:
        case_dir: Case folder under Training or Validation.
        case_id:  BraTS subject ID (matches the folder name).
        modality: One of MODALITIES, e.g. 't1c'.

    Returns:
        3D SimpleITK image. Pass this as the first argument of
        RadiomicsFeatureExtractor.execute(image, mask).
    """
    path = mri_volume_path(case_dir, case_id, modality)
    # --- MRI volume extracted here (3D NIfTI -> SimpleITK Image) ---
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, got {image.GetDimension()}D at {path}"
        )
    return image


def load_tumor_mask_for_pyradiomics(
    case_dir: Path,
    case_id: str,
    roi: str = "whole",
    label_value: int = RADIOMICS_LABEL,
) -> sitk.Image:
    """
    Load the 3D ROI mask that tells PyRadiomics which voxels to measure.

    BraTS training segs use labels 1 / 2 / 4. PyRadiomics extracts one label
    at a time, so by default every tumor voxel is remapped to `label_value`
    (whole tumor). Pass roi='et' / 'ed' / 'ncr' for a single subregion.

    Args:
        case_dir:    Case folder that contains `{case_id}-seg.nii.gz`.
        case_id:     BraTS subject ID.
        roi:         'whole', 'et', 'ed' or 'ncr'.
        label_value: Value written into the binary ROI (default 1).

    Returns:
        3D SimpleITK label image aligned with the MRI volume.
    """
    path = seg_mask_path(case_dir, case_id)
    if path is None:
        raise FileNotFoundError(
            f"No segmentation mask in {case_dir}. Training cases have "
            f"{case_id}-seg.nii.gz; Validation cases do not. Use "
            "load_brain_mask_for_pyradiomics() or a predicted mask."
        )

    # Native 3D seg, same grid as the MRI — not the resized 96^3 training mask.
    mask = sitk.ReadImage(str(path), sitk.sitkUInt32)
    array = sitk.GetArrayFromImage(mask)

    if roi == "whole":
        selected = array > 0
    elif roi in BRATS_LABELS:
        selected = array == BRATS_LABELS[roi]
    else:
        raise ValueError(
            f"roi must be 'whole', 'et', 'ed' or 'ncr', got {roi!r}"
        )

    binary = np.where(selected, label_value, 0).astype(np.uint32)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(mask)
    return out


def load_brain_mask_for_pyradiomics(
    image: sitk.Image,
    label_value: int = RADIOMICS_LABEL,
) -> sitk.Image:
    """
    Build a 3D brain ROI from non-zero MRI voxels.

    Used for official Validation cases, which have no tumor mask. This is
    whole-brain radiomics, not tumor radiomics — say so in any table you
    publish from those rows.
    """
    array = sitk.GetArrayFromImage(image)
    binary = np.where(array != 0, label_value, 0).astype(np.uint32)
    mask = sitk.GetImageFromArray(binary)
    mask.CopyInformation(image)
    return mask


def load_case_volumes_for_pyradiomics(
    case_id: str,
    modality: str = "t1c",
    split: Optional[str] = None,
    roi: str = "whole",
    case_dir: Optional[Path] = None,
) -> RadiomicsVolume:
    """
    Extract the MRI volume + ROI for one case / one modality.

    This is the convenience wrapper the notebook calls. Under the hood it
    is load_mri_volume_for_pyradiomics + a mask loader.

    Args:
        case_id:  BraTS subject ID.
        modality: MRI channel to load.
        split:    'training', 'validation', or None to search both.
        roi:      'whole' / 'et' / 'ed' / 'ncr', or 'brain' to skip the seg.
        case_dir: Optional explicit folder (overrides split lookup).

    Returns:
        RadiomicsVolume ready for extractor.execute(image, mask).
    """
    case_dir = Path(case_dir) if case_dir is not None else case_dir_for(case_id, split)

    # MRI VOLUME EXTRACTED FOR PYRADIOMICS (see load_mri_volume_for_pyradiomics).
    image = load_mri_volume_for_pyradiomics(case_dir, case_id, modality)
    image_path = mri_volume_path(case_dir, case_id, modality)
    mask_path = seg_mask_path(case_dir, case_id)

    if roi == "brain" or mask_path is None:
        mask = load_brain_mask_for_pyradiomics(image)
        used_roi = "brain"
    else:
        mask = load_tumor_mask_for_pyradiomics(case_dir, case_id, roi=roi)
        used_roi = roi

    return RadiomicsVolume(
        case_id=case_id,
        modality=modality,
        image=image,
        mask=mask,
        image_path=image_path,
        mask_path=mask_path,
        roi=used_roi,
    )


def describe_sitk_volume(image: sitk.Image) -> str:
    """One-line summary so you can confirm a 3D volume was loaded."""
    size = image.GetSize()
    spacing = tuple(round(s, 3) for s in image.GetSpacing())
    origin = tuple(round(o, 2) for o in image.GetOrigin())
    return (
        f"{image.GetDimension()}D volume  "
        f"size={size}  spacing={spacing} mm  origin={origin}"
    )


def list_modalities(case_dir: Path, case_id: str) -> List[str]:
    """Which of the 4 MRI channels are actually on disk for this case."""
    return [
        mod for mod in MODALITIES
        if (Path(case_dir) / f"{case_id}-{mod}.nii.gz").is_file()
    ]
