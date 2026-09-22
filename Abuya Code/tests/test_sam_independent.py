import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from brain_tumor_seg.sam.medsam2 import MedSAM2Refiner


class _PromptOnlyPredictor:
    def __init__(self):
        self.mask_prompt_calls = 0

    def init_state(self, _images, height, width, **_kwargs):
        return {"height": height, "width": width}

    def add_new_mask(self, *_args, **_kwargs):
        self.mask_prompt_calls += 1

    def add_new_points_or_box(self, **_kwargs):
        return None

    def propagate_in_video(self, state, reverse=False):
        logits = torch.full((1, 1, state["height"], state["width"]), -1.0)
        logits[0, 0, 2:6, 2:6] = 1.0
        yield 1, [1], logits

    def reset_state(self, _state):
        return None


class IndependentSamTests(unittest.TestCase):
    def test_manual_sam_runs_without_step1_seed(self):
        predictor = _PromptOnlyPredictor()
        refiner = MedSAM2Refiner(Path("unused.pt"))
        refiner._load = lambda: predictor
        refiner._frames = lambda _volume, _plane: torch.zeros((3, 3, 8, 8))
        volume = np.zeros((8, 8, 3), dtype=np.float32)

        with patch("torch.cuda.is_bf16_supported", return_value=False):
            result = refiner.refine(
                volume=volume,
                seed_mask=None,
                plane="axial",
                slice_index=1,
                box=(1, 1, 6, 6),
                points=[(4, 4)],
                point_labels=[1],
            )

        self.assertEqual(result.shape, volume.shape)
        self.assertGreater(result.sum(), 0)
        self.assertEqual(predictor.mask_prompt_calls, 0)


if __name__ == "__main__":
    unittest.main()
