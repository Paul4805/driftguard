"""Fast unit/smoke tests for the DriftGuard core.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Make the project root importable when tests are run from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftguard import (  # noqa: E402
    KCFTracker,
    bbox_iou,
    compute_psr,
    evaluate_gt_full,
    subpixel_peak_offset,
)


class TestPSR(unittest.TestCase):
    def test_peaked_response_has_high_psr(self) -> None:
        response = np.random.default_rng(0).normal(0, 0.01, (80, 80))
        response[40, 40] = 1.0
        self.assertGreater(compute_psr(response), 20.0)

    def test_flat_response_has_low_psr(self) -> None:
        response = np.random.default_rng(1).normal(1.0, 0.5, (80, 80))
        # A flat/noisy map has no trustworthy peak.
        self.assertLess(compute_psr(response), 5.0)

    def test_subpixel_peak_direction(self) -> None:
        response = np.zeros((9, 9), dtype=np.float64)
        response[4, 3] = 0.8
        response[4, 4] = 1.0
        response[4, 5] = 0.9
        dx, dy = subpixel_peak_offset(response, 4, 4)
        self.assertGreater(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0, places=6)


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self) -> None:
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)


class TestGTEvaluation(unittest.TestCase):
    def test_success_and_precision_metrics(self) -> None:
        gt = {
            0: (0.0, 0.0, 10.0, 10.0),
            1: (0.0, 0.0, 10.0, 10.0),
            2: (0.0, 0.0, 10.0, 10.0),
        }
        perfect = dict(gt)
        shifted = {
            0: (0.0, 0.0, 10.0, 10.0),
            1: (10.0, 0.0, 10.0, 10.0),
            2: (100.0, 100.0, 10.0, 10.0),
        }
        result = evaluate_gt_full(gt, shifted)
        summary = result["summary"]
        self.assertEqual(summary["gt_frames"], 3.0)
        self.assertAlmostEqual(summary["success_rate_iou_0.5"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(summary["precision_center_20px"], 2.0 / 3.0, places=4)
        self.assertGreater(summary["success_auc"], 0.0)
        self.assertEqual(len(result["success_curve"]), 101)
        self.assertEqual(len(result["precision_curve"]), 51)

        perfect_result = evaluate_gt_full(gt, perfect)
        self.assertAlmostEqual(perfect_result["summary"]["success_auc"], 1.0, places=4)
        self.assertAlmostEqual(
            perfect_result["summary"]["precision_center_20px"], 1.0, places=4
        )


class TestKCFCore(unittest.TestCase):
    def test_tracks_diagonal_translation(self) -> None:
        """The custom KCF must return the response map and follow motion."""
        rng = np.random.default_rng(7)
        height, width = 240, 400
        background = rng.integers(60, 180, (height, width, 3), dtype=np.uint8)
        background = cv2.GaussianBlur(background, (9, 9), 0)

        target = np.zeros((60, 30, 3), np.uint8)
        target[:30] = (20, 30, 160)
        target[30:] = (150, 80, 30)
        cv2.circle(target, (15, 15), 8, (220, 220, 220), -1)

        frames = []
        cx, cy = 120.0, 200.0
        for _ in range(30):
            frame = background.copy()
            x = int(round(cx - 15))
            y = int(round(cy - 30))
            frame[y : y + 60, x : x + 30] = target
            frames.append(frame)
            cx += 4.2
            cy -= 2.1

        tracker = KCFTracker()
        tracker.initialize(frames[0], (120 - 15, 200 - 30, 30, 60))
        last_psr = 0.0
        last_box = (0.0, 0.0, 0.0, 0.0)
        for idx in range(1, len(frames)):
            box, response = tracker.detect(frames[idx])
            last_psr = compute_psr(response)
            tracker.update_model(frames[idx], box)
            last_box = box

        expected_x = 120 - 15 + 4.2 * (len(frames) - 1)
        expected_y = 200 - 30 - 2.1 * (len(frames) - 1)
        self.assertLess(abs(last_box[0] - expected_x), 4.0)
        self.assertLess(abs(last_box[1] - expected_y), 4.0)
        self.assertGreater(last_psr, 7.0)


if __name__ == "__main__":
    unittest.main()
