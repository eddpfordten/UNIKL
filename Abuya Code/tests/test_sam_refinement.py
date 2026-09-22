import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from brain_tumor_seg.sam.medsam2 import MedSAM2Refiner
from brain_tumor_seg.sam.refinement import (
    RefinementRejectedError,
    build_auto_prompt,
    filter_refinement,
    select_component_at_click,
)


class ComponentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((24, 24, 12), dtype=np.int32)
        self.labels[3:8, 4:9, 5:8] = 1
        self.labels[14:21, 13:20, 5:8] = 2

    def test_exact_click_selects_component(self):
        selected = select_component_at_click(self.labels, "axial", 6, 5, 5)
        self.assertEqual(selected, 1)

    def test_nearby_click_snaps_to_nearest_component(self):
        selected = select_component_at_click(self.labels, "axial", 6, 11, 15, snap_radius=6)
        self.assertEqual(selected, 2)

    def test_distant_click_selects_nothing(self):
        selected = select_component_at_click(self.labels, "axial", 6, 0, 23, snap_radius=6)
        self.assertIsNone(selected)

    def test_tie_prefers_larger_slice_component(self):
        labels = np.zeros((15, 15, 3), dtype=np.int32)
        labels[5, 4, 1] = 1
        labels[4:7, 8:10, 1] = 2
        selected = select_component_at_click(labels, "axial", 1, 6, 5, snap_radius=3)
        self.assertEqual(selected, 2)


class AutoPromptTests(unittest.TestCase):
    def test_largest_slice_becomes_anchor_and_box_is_expanded(self):
        seed = np.zeros((20, 22, 12), dtype=np.uint8)
        seed[8:11, 9:13, 4] = 1
        seed[6:14, 7:16, 5] = 1
        seed[8:11, 9:13, 6] = 1
        prompt = build_auto_prompt(seed, "axial")
        self.assertEqual(prompt.anchor_slice, 5)
        self.assertEqual(prompt.box, (3.0, 2.0, 19.0, 17.0))
        self.assertLessEqual(prompt.first_slice, 2)
        self.assertGreaterEqual(prompt.last_slice, 8)

    def test_box_is_clamped_at_volume_boundary(self):
        seed = np.zeros((12, 12, 6), dtype=np.uint8)
        seed[0:3, 0:2, 2:4] = 1
        prompt = build_auto_prompt(seed, "axial")
        self.assertEqual(prompt.box[0:2], (0.0, 0.0))


class RefinementFilterTests(unittest.TestCase):
    def setUp(self):
        self.seed = np.zeros((24, 24, 12), dtype=np.uint8)
        self.seed[8:14, 9:15, 4:8] = 1
        self.prompt = build_auto_prompt(self.seed, "axial")

    def test_keeps_only_component_overlapping_seed(self):
        predicted = self.seed.copy()
        predicted[0:3, 0:3, 0:3] = 1
        result, quality = filter_refinement(predicted, self.seed, self.prompt.roi)
        np.testing.assert_array_equal(result, self.seed.astype(np.float32))
        self.assertEqual(quality.dice_with_seed, 1.0)

    def test_rejects_result_without_seed_overlap(self):
        predicted = np.zeros_like(self.seed)
        predicted[18:22, 18:22, 8:11] = 1
        with self.assertRaises(RefinementRejectedError):
            filter_refinement(predicted, self.seed, tuple(slice(0, n) for n in self.seed.shape))

    def test_rejects_implausible_volume(self):
        predicted = np.ones_like(self.seed)
        with self.assertRaises(RefinementRejectedError):
            filter_refinement(predicted, self.seed, tuple(slice(0, n) for n in self.seed.shape))


class _FakePredictor:
    def __init__(self):
        self.states = []
        self.reset_states = []
        self.boxes = []

    def init_state(self, _images, height, width, **_kwargs):
        state = {"id": len(self.states), "height": height, "width": width, "box": None}
        self.states.append(state)
        return state

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box, **_kwargs):
        inference_state["box"] = np.asarray(box)
        inference_state["anchor"] = frame_idx
        self.boxes.append((inference_state["id"], obj_id, np.asarray(box).copy()))

    def propagate_in_video(self, state, start_frame_idx, reverse, **_kwargs):
        step = -1 if reverse else 1
        for frame_idx in (start_frame_idx, start_frame_idx + step):
            logits = torch.full((1, 1, state["height"], state["width"]), -1.0)
            x0, y0, x1, y1 = state["box"].astype(int)
            logits[0, 0, y0 + 4:y1 - 3, x0 + 4:x1 - 3] = 1.0
            yield frame_idx, [1], logits

    def reset_state(self, state):
        self.reset_states.append(state["id"])


class RefinerWorkflowTests(unittest.TestCase):
    def test_forward_and_reverse_use_fresh_box_prompt_states(self):
        volume = np.zeros((20, 22, 12), dtype=np.float32)
        seed = np.zeros_like(volume)
        seed[6:14, 7:16, 4:7] = 1
        fake = _FakePredictor()
        refiner = MedSAM2Refiner(Path("unused.pt"))
        refiner._load = lambda: fake
        refiner._frames = lambda _volume, plane: torch.zeros(
            (12, 3, 8, 8), dtype=torch.float32
        )

        with patch("torch.cuda.is_bf16_supported", return_value=False):
            result, quality = refiner.refine(volume, seed, "axial")

        self.assertEqual(len(fake.states), 2)
        self.assertEqual(fake.reset_states, [0, 1])
        self.assertEqual([item[0] for item in fake.boxes], [0, 1])
        self.assertGreater(int(result.sum()), 0)
        self.assertGreaterEqual(quality.dice_with_seed, 0.20)


if __name__ == "__main__":
    unittest.main()
