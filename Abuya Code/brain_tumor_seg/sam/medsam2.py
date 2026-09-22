"""A narrow, single-target adapter around the official MedSAM2 predictor."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .refinement import RefinementQuality, build_auto_prompt, filter_refinement
from .transforms import PLANE_TO_AXIS, frames_as_volume, volume_as_frames


MEDSAM_LOGIT_THRESHOLD = 0.0


class MedSAM2Refiner:
    """Cache MedSAM2 and refine one Step 1 tumor selected by the user."""

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
        """Apply the official non-CT percentile scaling and RGB conversion."""
        normalized = np.asarray(volume, dtype=np.float32).copy()
        foreground = normalized[normalized > 0]
        if foreground.size:
            low, high = np.percentile(foreground, (0.5, 99.5))
            if high > low:
                normalized = np.clip(normalized, low, high)
                normalized = (normalized - low) / (high - low)
                normalized[volume <= 0] = 0
        normalized = np.clip(normalized, 0.0, 1.0)

        frames = volume_as_frames(normalized, plane)
        converted = []
        for frame in frames:
            image = Image.fromarray(np.uint8(frame * 255.0), mode="L").convert("RGB")
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
            converted.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        tensor = torch.from_numpy(np.stack(converted)).to("cuda")
        mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[:, None, None]
        return (tensor - mean) / std

    @staticmethod
    def _propagate_once(
        predictor,
        images: torch.Tensor,
        height: int,
        width: int,
        anchor_slice: int,
        box: tuple[float, float, float, float],
        first_slice: int,
        last_slice: int,
        reverse: bool,
    ) -> dict[int, np.ndarray]:
        """Run one direction with an isolated state, as in official inference."""
        state = predictor.init_state(
            images, height, width, offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        scores: dict[int, np.ndarray] = {}
        try:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=anchor_slice,
                obj_id=1,
                box=np.asarray(box, dtype=np.float32),
                clear_old_points=True,
            )
            distance = anchor_slice - first_slice if reverse else last_slice - anchor_slice
            for frame_index, _object_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=anchor_slice,
                max_frame_num_to_track=distance,
                reverse=reverse,
            ):
                if first_slice <= frame_index <= last_slice:
                    scores[int(frame_index)] = logits[0, 0].detach().float().cpu().numpy()
        finally:
            predictor.reset_state(state)
        return scores

    def refine(
        self,
        volume: np.ndarray,
        selected_mask: np.ndarray,
        plane: str,
    ) -> tuple[np.ndarray, RefinementQuality]:
        """Refine only ``selected_mask`` using an automatically generated box."""
        if plane not in PLANE_TO_AXIS:
            raise ValueError(f"Unknown plane {plane!r}")
        if volume.shape != selected_mask.shape or volume.ndim != 3:
            raise ValueError("volume and selected_mask must be matching 3D arrays")

        prompt = build_auto_prompt(selected_mask, plane)
        predictor = self._load()
        frames = volume_as_frames(volume, plane)
        height, width = frames.shape[1:]
        images = None
        try:
            images = self._frames(volume, plane)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                forward = self._propagate_once(
                    predictor, images, height, width, prompt.anchor_slice, prompt.box,
                    prompt.first_slice, prompt.last_slice, reverse=False,
                )
                reverse = self._propagate_once(
                    predictor, images, height, width, prompt.anchor_slice, prompt.box,
                    prompt.first_slice, prompt.last_slice, reverse=True,
                )

            scores = np.full(frames.shape, -np.inf, dtype=np.float32)
            for outputs in (forward, reverse):
                for frame_index, frame_scores in outputs.items():
                    scores[frame_index] = np.maximum(scores[frame_index], frame_scores)
            predicted_frames = scores > MEDSAM_LOGIT_THRESHOLD
            predicted = frames_as_volume(predicted_frames, plane)
            return filter_refinement(predicted, selected_mask, prompt.roi)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "MedSAM2 ran out of GPU memory. Close other GPU applications and try again."
            ) from exc
        finally:
            if images is not None:
                del images
            torch.cuda.empty_cache()
