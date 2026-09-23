"""
Run PyRadiomics on a 3D MRI volume + ROI mask.

The extractor is built once and reused. Each call to extract_case_features
loads every requested MRI channel through volume.py and runs
RadiomicsFeatureExtractor.execute(image, mask) on the 3D pair.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import SimpleITK as sitk
from tqdm.auto import tqdm

from brain_tumor_seg.config import RADIOMICS_LABEL, RADIOMICS_MODALITIES
from brain_tumor_seg.radiomics.settings import (
    RADIOMICS_FEATURE_CLASSES,
    RADIOMICS_IMAGE_TYPES,
    pyradiomics_3d_settings,
)
from brain_tumor_seg.radiomics.volume import (
    case_dir_for,
    load_brain_mask_for_pyradiomics,
    load_mri_volume_for_pyradiomics,
    load_tumor_mask_for_pyradiomics,
    seg_mask_path,
)

# Elapsed / remaining / rate — same layout as the batch folder bar.
_TQDM_BAR = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


NumericFeatureDict = Dict[str, Union[float, str]]


def _require_pyradiomics():
    """Import PyRadiomics only when extraction is actually requested."""
    try:
        import logging

        from radiomics import featureextractor, setVerbosity
    except ImportError as error:
        raise ImportError(
            "PyRadiomics is not installed. From the project root run:\n"
            "    py -3.12 -m pip install SimpleITK PyWavelets\n"
            "    py -3.12 -m pip install \"https://github.com/steubk/pyradiomics/releases/download/prebuilt-fix-py312/pyradiomics-3.1.0%2Bpy312fix.363964e-cp312-cp312-win_amd64.whl\""
        ) from error

    # PyRadiomics logs every feature class at INFO; keep the notebook readable.
    setVerbosity(logging.ERROR)
    return featureextractor


def build_3d_extractor(
    bin_width: Optional[int] = None,
    feature_classes: Sequence[str] = RADIOMICS_FEATURE_CLASSES,
    image_types: Sequence[str] = RADIOMICS_IMAGE_TYPES,
):
    """
    Build a RadiomicsFeatureExtractor configured for 3D MRI (force2D=False).

    Args:
        bin_width:       Histogram bin width; None uses the config default.
        feature_classes: PyRadiomics feature families to enable.
        image_types:     Filters to apply ('Original', optionally 'Wavelet', 'LoG').

    Returns:
        A configured radiomics.featureextractor.RadiomicsFeatureExtractor.
    """
    featureextractor = _require_pyradiomics()
    settings = pyradiomics_3d_settings()
    if bin_width is not None:
        settings["binWidth"] = bin_width

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Start from a clean slate, then enable only what this project wants.
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()

    for image_type in image_types:
        extractor.enableImageTypeByName(image_type)

    for name in feature_classes:
        extractor.enableFeatureClassByName(name)

    return extractor


def numeric_features(result: Dict) -> Dict[str, float]:
    """
    Drop PyRadiomics diagnostics_* keys, keep the actual radiomic values.

    Diagnostic entries describe the run (version, bounding box, voxel count);
    they are useful for logs but should not go into a feature matrix.
    """
    features: Dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[key] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def extract_features_from_volume(
    image: sitk.Image,
    mask: sitk.Image,
    extractor=None,
    label: int = RADIOMICS_LABEL,
) -> Dict[str, float]:
    """
    Run PyRadiomics on an already-loaded 3D MRI volume and mask.

    Args:
        image:     3D MRI from load_mri_volume_for_pyradiomics().
        mask:      3D ROI from load_tumor_mask_for_pyradiomics().
        extractor: Reuse a built extractor; None builds the default 3D one.
        label:     ROI label inside the mask (1 after whole-tumor remap).

    Returns:
        {feature_name: value} without diagnostics keys.
    """
    if extractor is None:
        extractor = build_3d_extractor()

    # Empty ROI: PyRadiomics would raise a less obvious geometry error.
    mask_voxels = sitk.GetArrayFromImage(mask)
    if not bool((mask_voxels == label).any()):
        raise ValueError(
            f"ROI is empty for label={label}; PyRadiomics needs at least one voxel."
        )

    # image and mask are SimpleITK 3D volumes, not file paths and not the
    # 96^3 training tensors.
    result = extractor.execute(image, mask, label=label)
    return numeric_features(result)


def extract_case_features(
    case_id: str,
    split: Optional[str] = None,
    modalities: Optional[Iterable[str]] = None,
    roi: str = "whole",
    case_dir: Optional[Path] = None,
    extractor=None,
    label: int = RADIOMICS_LABEL,
    show_progress: bool = True,
) -> NumericFeatureDict:
    """
    Extract 3D PyRadiomics features for every MRI channel of one case.

    Shape features (geometry of the ROI) are stored once, without a modality
    prefix. Texture / first-order features are prefixed with the channel
    (t1c_, t1n_, t2f_, t2w_) so the four volumes stay distinguishable.

    Args:
        case_id:       BraTS subject ID.
        split:         'training', 'validation', or None to search both.
        modalities:    MRI channels to extract. Defaults to all four.
        roi:           'whole' / 'et' / 'ed' / 'ncr' / 'brain'.
        case_dir:      Optional explicit case folder.
        extractor:     Reuse a built extractor across cases.
        label:         ROI label PyRadiomics should read.
        show_progress: tqdm bar over modalities (timer + ETA). Turn off when
                       a folder-level bar already wraps this call.

    Returns:
        Flat dict including 'case_id' and 'roi', plus radiomic values.
    """
    if extractor is None:
        extractor = build_3d_extractor()

    case_dir = Path(case_dir) if case_dir is not None else case_dir_for(case_id, split)
    channels: List[str] = list(modalities) if modalities is not None else list(RADIOMICS_MODALITIES)

    # Load the ROI once; every MRI channel shares the same 3D mask.
    if roi == "brain" or seg_mask_path(case_dir, case_id) is None:
        # Validation (or an explicit brain ROI): mask from non-zero MRI voxels.
        image0 = load_mri_volume_for_pyradiomics(case_dir, case_id, channels[0])
        mask = load_brain_mask_for_pyradiomics(image0)
        used_roi = "brain"
    else:
        mask = load_tumor_mask_for_pyradiomics(case_dir, case_id, roi=roi, label_value=label)
        used_roi = roi

    row: NumericFeatureDict = {"case_id": case_id, "roi": used_roi}
    shape_stored = False

    channel_iter = channels
    if show_progress and len(channels) > 1:
        channel_iter = tqdm(
            channels,
            desc=f"PyRadiomics {case_id}",
            unit="vol",
            bar_format=_TQDM_BAR,
            dynamic_ncols=True,
            leave=True,
        )

    for modality in channel_iter:
        if show_progress and hasattr(channel_iter, "set_postfix_str"):
            channel_iter.set_postfix_str(modality, refresh=False)

        # ----- MRI VOLUME EXTRACTED FOR PYRADIOMICS (this channel) -----
        image = load_mri_volume_for_pyradiomics(case_dir, case_id, modality)
        features = extract_features_from_volume(image, mask, extractor=extractor, label=label)

        for name, value in features.items():
            if "_shape_" in name:
                # Same mask for every channel — keep shape once.
                if not shape_stored:
                    row[name] = value
                continue
            row[f"{modality}_{name}"] = value
        shape_stored = True

    return row


def preview_feature_names(row: Dict, limit: int = 12) -> List[str]:
    """First few radiomic feature names, skipping the metadata keys."""
    names = [key for key in row if key not in {"case_id", "roi", "split"}]
    return names[:limit]


def format_feature_preview(row: Dict, limit: int = 15) -> str:
    """Human-readable snippet of one case's radiomic vector."""
    skip = {"case_id", "roi", "split"}
    names = [key for key in row if key not in skip]
    lines = [
        f"case_id     : {row.get('case_id')}",
        f"roi         : {row.get('roi')}",
        f"n_features  : {len(names)}",
        "",
    ]
    for name in names[:limit]:
        value = row[name]
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4g}")
        else:
            lines.append(f"  {name}: {value}")
    if len(names) > limit:
        lines.append(f"  ... {len(names) - limit} more")
    return "\n".join(lines)
