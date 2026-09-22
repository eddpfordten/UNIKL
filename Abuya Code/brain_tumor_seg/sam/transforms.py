"""Reversible mapping between native NIfTI and the model/display grid."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

PLANE_TO_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def volume_as_frames(volume: np.ndarray, plane: str) -> np.ndarray:
    if plane not in PLANE_TO_AXIS:
        raise ValueError(f"Unknown plane {plane!r}")
    return np.moveaxis(volume, PLANE_TO_AXIS[plane], 0)


def frames_as_volume(frames: np.ndarray, plane: str) -> np.ndarray:
    if plane not in PLANE_TO_AXIS:
        raise ValueError(f"Unknown plane {plane!r}")
    return np.moveaxis(frames, 0, PLANE_TO_AXIS[plane])


def rotated_to_prompt_xy(shown_x: float, shown_y: float, rows: int, cols: int):
    """Undo ``np.rot90`` and return SAM coordinates (x=column, y=row)."""
    row = float(np.clip(rows - 1 - shown_x, 0, rows - 1))
    col = float(np.clip(shown_y, 0, cols - 1))
    return col, row


def prompt_to_rotated_xy(x: float, y: float, rows: int):
    """Map SAM slice coordinates onto the rotated Matplotlib display."""
    return float(rows - 1 - y), float(x)


def _resize_exact(array: np.ndarray, shape: Tuple[int, int, int], order: int) -> np.ndarray:
    shape = tuple(int(value) for value in shape)
    if tuple(array.shape) == shape:
        return array.copy()
    resized = zoom(array, [target / source for target, source in zip(shape, array.shape)], order=order)
    slices = tuple(slice(0, min(source, target)) for source, target in zip(resized.shape, shape))
    cropped = resized[slices]
    pads = [(0, target - source) for source, target in zip(cropped.shape, shape)]
    return np.pad(cropped, pads, mode="constant")


@dataclass(frozen=True)
class VolumeTransform:
    """Metadata required to restore a display-grid mask to native space."""

    native_shape: Tuple[int, int, int]
    display_shape: Tuple[int, int, int]
    affine: np.ndarray
    header: nib.Nifti1Header

    def to_display(self, array: np.ndarray, order: int = 1) -> np.ndarray:
        if tuple(array.shape) != tuple(self.native_shape):
            raise ValueError(f"Expected native shape {self.native_shape}, got {array.shape}")
        return _resize_exact(array, self.display_shape, order)

    def to_native_mask(self, mask: np.ndarray) -> np.ndarray:
        if tuple(mask.shape) != tuple(self.display_shape):
            raise ValueError(f"Expected display shape {self.display_shape}, got {mask.shape}")
        native = _resize_exact(mask.astype(np.uint8), self.native_shape, order=0)
        return (native > 0).astype(np.uint8)

    def save_mask(self, mask: np.ndarray, destination: Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        native = self.to_native_mask(mask)
        header = self.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(native, self.affine, header=header)
        temporary = destination.with_name(destination.name + ".tmp.nii.gz")
        nib.save(image, str(temporary))
        temporary.replace(destination)
        return destination

    @classmethod
    def from_image(cls, image: nib.Nifti1Image, display_shape=(128, 128, 128)) -> "VolumeTransform":
        native_shape = tuple(int(value) for value in image.shape[:3])
        return cls(
            native_shape=native_shape,
            display_shape=tuple(int(value) for value in display_shape),
            affine=np.asarray(image.affine).copy(),
            header=image.header.copy(),
        )
