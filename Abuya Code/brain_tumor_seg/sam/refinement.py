"""Pure helpers for selecting and constraining one MedSAM2 tumor target."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.ndimage import label as ndi_label

from .transforms import PLANE_TO_AXIS, volume_as_frames


SNAP_RADIUS = 6
BOX_MARGIN_FRACTION = 0.15
BOX_MARGIN_MIN = 4
BOX_MARGIN_MAX = 12
ROI_MARGIN_FRACTION = 0.25
ROI_MARGIN_MIN = 4
SLICE_MARGIN_FRACTION = 0.20
SLICE_MARGIN_MIN = 2
SLICE_MARGIN_MAX = 8


class RefinementRejectedError(ValueError):
    """Raised when a MedSAM2 result fails the safety/quality gates."""


@dataclass(frozen=True)
class AutoPrompt:
    anchor_slice: int
    box: tuple[float, float, float, float]
    first_slice: int
    last_slice: int
    roi: tuple[slice, slice, slice]


@dataclass(frozen=True)
class RefinementQuality:
    dice_with_seed: float
    volume_ratio: float
    overlap_voxels: int


def select_component_at_click(
    labels: np.ndarray,
    plane: str,
    slice_index: int,
    x: float,
    y: float,
    snap_radius: int = SNAP_RADIUS,
) -> int | None:
    """Return the clicked label, or the nearest label in a small 2D radius."""
    if labels.ndim != 3 or plane not in PLANE_TO_AXIS:
        raise ValueError("labels must be 3D and plane must be anatomical")
    frames = volume_as_frames(labels, plane)
    if not 0 <= slice_index < len(frames):
        return None
    frame = frames[slice_index]
    px, py = int(round(x)), int(round(y))
    if not (0 <= py < frame.shape[0] and 0 <= px < frame.shape[1]):
        return None
    direct = int(frame[py, px])
    if direct > 0:
        return direct

    radius = max(0, int(snap_radius))
    y0, y1 = max(0, py - radius), min(frame.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(frame.shape[1], px + radius + 1)
    nearby = frame[y0:y1, x0:x1]
    candidates = []
    for label_id in np.unique(nearby):
        label_id = int(label_id)
        if label_id <= 0:
            continue
        yy, xx = np.where(frame == label_id)
        distances = (xx - px) ** 2 + (yy - py) ** 2
        minimum = int(distances.min())
        if minimum <= radius * radius:
            candidates.append((minimum, -len(xx), label_id))
    return min(candidates)[2] if candidates else None


def _expanded_slices(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if not len(coords):
        raise ValueError("selected tumor mask is empty")
    result = []
    for axis, size in enumerate(mask.shape):
        low, high = int(coords[:, axis].min()), int(coords[:, axis].max())
        span = high - low + 1
        margin = max(ROI_MARGIN_MIN, ceil(span * ROI_MARGIN_FRACTION))
        result.append(slice(max(0, low - margin), min(size, high + margin + 1)))
    return tuple(result)


def build_auto_prompt(selected_mask: np.ndarray, plane: str) -> AutoPrompt:
    """Build a deterministic box on the target's largest cross-section."""
    selected = np.asarray(selected_mask, dtype=bool)
    if selected.ndim != 3 or plane not in PLANE_TO_AXIS:
        raise ValueError("selected_mask must be 3D and plane must be anatomical")
    frames = volume_as_frames(selected, plane)
    areas = frames.reshape(frames.shape[0], -1).sum(axis=1)
    if not np.any(areas):
        raise ValueError("selected tumor mask is empty")
    anchor = int(np.argmax(areas))
    yy, xx = np.where(frames[anchor])
    x_min, x_max = int(xx.min()), int(xx.max())
    y_min, y_max = int(yy.min()), int(yy.max())
    longest = max(x_max - x_min + 1, y_max - y_min + 1)
    margin = int(np.clip(ceil(longest * BOX_MARGIN_FRACTION), BOX_MARGIN_MIN, BOX_MARGIN_MAX))
    height, width = frames.shape[1:]
    box = (
        float(max(0, x_min - margin)),
        float(max(0, y_min - margin)),
        float(min(width - 1, x_max + margin)),
        float(min(height - 1, y_max + margin)),
    )

    occupied = np.flatnonzero(areas)
    first, last = int(occupied[0]), int(occupied[-1])
    span = last - first + 1
    slice_margin = int(np.clip(ceil(span * SLICE_MARGIN_FRACTION), SLICE_MARGIN_MIN, SLICE_MARGIN_MAX))
    return AutoPrompt(
        anchor_slice=anchor,
        box=box,
        first_slice=max(0, first - slice_margin),
        last_slice=min(len(frames) - 1, last + slice_margin),
        roi=_expanded_slices(selected),
    )


def filter_refinement(
    predicted_mask: np.ndarray,
    selected_mask: np.ndarray,
    roi: tuple[slice, slice, slice],
    min_volume_ratio: float = 0.25,
    max_volume_ratio: float = 4.0,
    min_seed_dice: float = 0.20,
) -> tuple[np.ndarray, RefinementQuality]:
    """Keep the prediction component that overlaps the selected Step 1 tumor."""
    predicted = np.asarray(predicted_mask, dtype=bool)
    seed = np.asarray(selected_mask, dtype=bool)
    if predicted.shape != seed.shape or predicted.ndim != 3:
        raise ValueError("predicted_mask and selected_mask must be matching 3D arrays")
    constrained = np.zeros_like(predicted)
    constrained[roi] = predicted[roi]
    components, count = ndi_label(constrained)
    if count == 0:
        raise RefinementRejectedError("MedSAM2 returned an empty mask for the selected tumor.")

    best_id = 0
    best_overlap = 0
    for component_id in range(1, count + 1):
        overlap = int(np.logical_and(components == component_id, seed).sum())
        if overlap > best_overlap:
            best_id, best_overlap = component_id, overlap
    if best_id == 0 or best_overlap == 0:
        raise RefinementRejectedError("MedSAM2 did not overlap the selected Step 1 tumor.")

    result = components == best_id
    seed_volume = int(seed.sum())
    result_volume = int(result.sum())
    ratio = result_volume / max(seed_volume, 1)
    dice = 2.0 * best_overlap / max(seed_volume + result_volume, 1)
    if not min_volume_ratio <= ratio <= max_volume_ratio:
        raise RefinementRejectedError(
            f"MedSAM2 volume was implausible ({ratio:.2f}× the selected Step 1 tumor)."
        )
    if dice < min_seed_dice:
        raise RefinementRejectedError(
            f"MedSAM2 overlap was too low ({dice:.2f} Dice with the selected Step 1 tumor)."
        )
    quality = RefinementQuality(dice, ratio, best_overlap)
    return result.astype(np.float32), quality
