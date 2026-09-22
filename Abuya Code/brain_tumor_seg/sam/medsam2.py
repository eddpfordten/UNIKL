"""A narrow adapter around the official MedSAM2 NPZ video predictor."""

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .transforms import PLANE_TO_AXIS, frames_as_volume, volume_as_frames


class MedSAM2Refiner:
    """Cache MedSAM2 and refine one binary 3D object from slice prompts."""

    def __init__(self, checkpoint: Path, config: str = "configs/sam2.1_hiera_t512.yaml"):
        self.checkpoint = Path(checkpoint)
        self.config = config
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MedSAM2 requires CUDA, but this Python installation has CPU-only PyTorch. "
                "Run setup_medsam2.ps1 with Python 3.12 first."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found: {self.checkpoint}. Run setup_medsam2.ps1."
            )
        try:
            from sam2.build_sam import build_sam2_video_predictor_npz
        except ImportError as exc:
            raise RuntimeError(
                "The official MedSAM2 package is not installed. Run setup_medsam2.ps1."
            ) from exc
        self._predictor = build_sam2_video_predictor_npz(
            self.config, str(self.checkpoint), device="cuda", apply_postprocessing=False
        )
        return self._predictor

    @staticmethod
    def _frames(volume: np.ndarray, plane: str) -> torch.Tensor:
        volume = np.asarray(volume, dtype=np.float32)
        nonzero = volume[volume != 0]
        if nonzero.size:
            low, high = np.percentile(nonzero, (0.5, 99.5))
            if high > low:
                volume = np.clip(volume, low, high)
                volume = (volume - low) / (high - low)
                volume[volume < 0] = 0
            else:
                volume = np.zeros_like(volume)
        else:
            volume = np.zeros_like(volume)
        frames = volume_as_frames(np.clip(volume, 0.0, 1.0), plane)
        converted = []
        for frame in frames:
            image = Image.fromarray(np.uint8(frame * 255.0), mode="L").convert("RGB")
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
            converted.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        tensor = torch.from_numpy(np.stack(converted)).to("cuda")
        mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[:, None, None]
        return (tensor - mean) / std

    def refine(
        self,
        volume: np.ndarray,
        seed_mask: Optional[np.ndarray],
        plane: str,
        slice_index: int,
        box: Optional[Sequence[float]],
        points: Sequence[Sequence[float]],
        point_labels: Sequence[int],
    ) -> np.ndarray:
        if plane not in PLANE_TO_AXIS:
            raise ValueError(f"Unknown plane {plane!r}")
        if volume.ndim != 3:
            raise ValueError("volume must be a 3D array")
        if seed_mask is None:
            seed_mask = np.zeros_like(volume, dtype=np.float32)
        if volume.shape != seed_mask.shape:
            raise ValueError("volume and seed_mask must be matching 3D arrays")
        if not box and not points:
            raise ValueError("Add a box or at least one foreground/background point")
        frames = volume_as_frames(volume, plane)
        if not 0 <= slice_index < len(frames):
            raise ValueError(f"slice index {slice_index} is outside {plane} volume")

        predictor = self._load()
        images = self._frames(volume, plane)
        height, width = frames.shape[1:]
        result = np.zeros(frames.shape, dtype=np.uint8)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else nullcontext()
        try:
            with torch.inference_mode(), autocast:
                # Point/box input replaces a mask prompt on the same frame in
                # SAM2, so do not submit seed_mask immediately before it. The
                # manual prompt is the single source of truth for this run.
                for reverse in (False, True):
                    state = predictor.init_state(
                        images, height, width, offload_video_to_cpu=True, offload_state_to_cpu=True
                    )
                    try:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=slice_index,
                            obj_id=1,
                            points=np.asarray(points, dtype=np.float32) if points else None,
                            labels=np.asarray(point_labels, dtype=np.int32) if points else None,
                            box=np.asarray(box, dtype=np.float32) if box else None,
                            clear_old_points=True,
                        )
                        for frame_index, _object_ids, logits in predictor.propagate_in_video(
                            state, reverse=reverse
                        ):
                            predicted = (logits[0, 0] > 0).detach().cpu().numpy()
                            result[frame_index] |= predicted.astype(np.uint8)
                    finally:
                        predictor.reset_state(state)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "MedSAM2 ran out of GPU memory. Close other GPU applications and try again."
            ) from exc
        finally:
            del images
            torch.cuda.empty_cache()
        return frames_as_volume(result, plane).astype(np.float32)
