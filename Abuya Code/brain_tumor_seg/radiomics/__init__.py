"""
3D PyRadiomics feature extraction from BraTS-PEDs MRI volumes.

The MRI volume is loaded in volume.py (load_mri_volume_for_pyradiomics).
extractor.py runs PyRadiomics on that 3D SimpleITK image + ROI mask.
batch.py walks the Training / Validation folders.
"""
from .batch import (
    extract_folder_features,
    extract_train_val_split_features,
    extract_training_features,
    extract_validation_features,
    save_features_csv,
)
from .extractor import (
    build_3d_extractor,
    extract_case_features,
    extract_features_from_volume,
    format_feature_preview,
    numeric_features,
    preview_feature_names,
)
from .features import RadiomicsFeatureTable, infer_radiomics_dim
from .settings import (
    RADIOMICS_FEATURE_CLASSES,
    RADIOMICS_IMAGE_TYPES,
    pyradiomics_3d_settings,
)
from .volume import (
    RadiomicsVolume,
    describe_sitk_volume,
    load_brain_mask_for_pyradiomics,
    load_case_volumes_for_pyradiomics,
    load_mri_volume_for_pyradiomics,
    load_tumor_mask_for_pyradiomics,
)

__all__ = [
    "RadiomicsVolume",
    "RadiomicsFeatureTable",
    "RADIOMICS_FEATURE_CLASSES",
    "RADIOMICS_IMAGE_TYPES",
    "build_3d_extractor",
    "describe_sitk_volume",
    "extract_case_features",
    "extract_features_from_volume",
    "extract_folder_features",
    "format_feature_preview",
    "extract_train_val_split_features",
    "extract_training_features",
    "extract_validation_features",
    "infer_radiomics_dim",
    "load_brain_mask_for_pyradiomics",
    "load_case_volumes_for_pyradiomics",
    "load_mri_volume_for_pyradiomics",
    "load_tumor_mask_for_pyradiomics",
    "numeric_features",
    "preview_feature_names",
    "pyradiomics_3d_settings",
    "save_features_csv",
]
