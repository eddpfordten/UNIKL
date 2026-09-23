"""
3D PyRadiomics extractor settings for BraTS-PEDs MRI.

Kept in one place so the notebook and the batch runner share the same
feature classes, bin width and (importantly) force2D=False — that flag is
what makes this a 3D extraction rather than slice-wise 2D texture.
"""
from typing import Dict, List

from brain_tumor_seg.config import RADIOMICS_BIN_WIDTH


# Feature families computed inside the 3D ROI.
# shape = geometry of the mask; the rest describe intensity / texture of the MRI.
RADIOMICS_FEATURE_CLASSES: List[str] = [
    "shape",
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
]

# Only the original (unfiltered) volume by default. Wavelets/LoG explode the
# feature count and runtime; turn them on via extra_image_types if needed.
RADIOMICS_IMAGE_TYPES: List[str] = ["Original"]


def pyradiomics_3d_settings(bin_width: int = RADIOMICS_BIN_WIDTH) -> Dict:
    """
    Settings dict passed into radiomics.featureextractor.RadiomicsFeatureExtractor.

    Args:
        bin_width: Grey-level bin width after MRI normalisation.

    Returns:
        Keyword arguments understood by PyRadiomics.
    """
    return {
        # 3D, not per-slice: force2D=False is the whole point of this module.
        "force2D": False,
        # MRI intensities are not on a shared scale (unlike CT HU), so
        # normalise each volume before the histogram is built.
        "normalize": True,
        "normalizeScale": 100,
        "binWidth": bin_width,
        # Keep native BraTS spacing (usually 1 mm isotropic). Do not resample
        # down to the 96^3 training grid — that would distort shape features.
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        # BraTS image/mask affines can differ by a tiny float; this tolerance
        # stops PyRadiomics rejecting a valid pair.
        "geometryTolerance": 1e-3,
        "correctMask": True,
        # Shift intensities after normalisation so first-order energy stays
        # defined for MRI (PyRadiomics MRI example uses 300).
        "voxelArrayShift": 300,
    }
