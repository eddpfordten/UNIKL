import unittest

import numpy as np

from brain_tumor_seg.sam.benchmark import binary_metrics


class BenchmarkMetricTests(unittest.TestCase):
    def test_perfect_mask(self):
        target = np.zeros((12, 12, 12), dtype=np.uint8)
        target[3:8, 4:9, 2:7] = 1
        metrics = binary_metrics(target, target)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["hd95"], 0.0)
        self.assertEqual(metrics["unrelated_leakage"], 0.0)

    def test_unrelated_component_is_measured_as_leakage(self):
        target = np.zeros((16, 16, 16), dtype=np.uint8)
        target[3:7, 3:7, 3:7] = 1
        predicted = target.copy()
        predicted[12:14, 12:14, 12:14] = 1
        metrics = binary_metrics(predicted, target)
        self.assertAlmostEqual(metrics["unrelated_leakage"], 8 / 72)


if __name__ == "__main__":
    unittest.main()
