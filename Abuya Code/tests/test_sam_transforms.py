import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from brain_tumor_seg.sam.transforms import (
    VolumeTransform,
    frames_as_volume,
    prompt_to_rotated_xy,
    rotated_to_prompt_xy,
    volume_as_frames,
)


class VolumeTransformTests(unittest.TestCase):
    def setUp(self):
        data = np.zeros((20, 24, 12), dtype=np.float32)
        self.image = nib.Nifti1Image(data, np.diag([1.0, 1.5, 2.0, 1.0]))
        self.transform = VolumeTransform.from_image(self.image, (32, 32, 32))

    def test_native_display_native_preserves_shape_and_binary_values(self):
        native = np.zeros(self.image.shape, dtype=np.uint8)
        native[5:14, 7:18, 3:9] = 1
        display = self.transform.to_display(native, order=0)
        restored = self.transform.to_native_mask(display)
        self.assertEqual(display.shape, (32, 32, 32))
        self.assertEqual(restored.shape, native.shape)
        self.assertEqual(set(np.unique(restored)), {0, 1})
        intersection = np.logical_and(native, restored).sum()
        dice = 2 * intersection / (native.sum() + restored.sum())
        self.assertGreater(dice, 0.85)

    def test_display_transform_matches_direct_training_resize(self):
        native = np.zeros(self.image.shape, dtype=np.uint8)
        native[:, :, -1] = 1
        display = self.transform.to_display(native, order=0)

        # A direct resize keeps the native boundary on the display boundary.
        # Cube-padding would shift this plane inward and disagree with Step 1.
        self.assertTrue(np.all(display[:, :, -1] == 1))

    def test_saved_mask_uses_native_geometry(self):
        display = np.zeros((32, 32, 32), dtype=np.uint8)
        display[8:20, 9:21, 10:18] = 1
        path = Path(__file__).with_name("_test_mask.nii.gz")
        try:
            path = self.transform.save_mask(display, path)
            saved = nib.load(str(path))
            self.assertEqual(saved.shape, self.image.shape)
            np.testing.assert_allclose(saved.affine, self.image.affine)
            self.assertEqual(saved.get_data_dtype(), np.dtype(np.uint8))
        finally:
            path.unlink(missing_ok=True)

    def test_rotated_prompt_coordinates_round_trip_at_corners(self):
        rows, cols = 20, 30
        for x, y in ((0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)):
            shown_x, shown_y = prompt_to_rotated_xy(x, y, cols)
            actual_x, actual_y = rotated_to_prompt_xy(shown_x, shown_y, rows, cols)
            self.assertEqual((actual_x, actual_y), (float(x), float(y)))

    def test_prompt_transform_matches_np_rot90_pixel_location(self):
        rows, cols = 4, 7
        source = np.zeros((rows, cols), dtype=np.uint8)
        for x, y in ((0, 0), (5, 1), (2, 3), (cols - 1, rows - 1)):
            source.fill(0)
            source[y, x] = 1
            shown = np.rot90(source)
            shown_y, shown_x = np.argwhere(shown == 1)[0]
            mapped_x, mapped_y = prompt_to_rotated_xy(x, y, cols)
            self.assertEqual((mapped_x, mapped_y), (float(shown_x), float(shown_y)))
            self.assertEqual(
                rotated_to_prompt_xy(mapped_x, mapped_y, rows, cols),
                (float(x), float(y)),
            )

    def test_all_anatomical_planes_round_trip(self):
        volume = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        for plane in ("sagittal", "coronal", "axial"):
            frames = volume_as_frames(volume, plane)
            np.testing.assert_array_equal(frames_as_volume(frames, plane), volume)


if __name__ == "__main__":
    unittest.main()
