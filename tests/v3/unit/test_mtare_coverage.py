from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.evaluation.mtare_coverage import MTAReCoverageAccumulator


class MTAReCoverageTests(unittest.TestCase):
    def test_unique_half_meter_voxels_match_native_volume(self):
        metric = MTAReCoverageAccumulator()
        first = metric.update(stamp_sec=1.0, registered_scan_xyz_m=np.asarray([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.6, 0.1, 0.1]]), vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)
        self.assertEqual(first.explored_voxels, 2)
        self.assertAlmostEqual(first.explored_volume_m3, 0.25)
        second = metric.update(stamp_sec=2.0, registered_scan_xyz_m=np.asarray([[0.2, 0.2, 0.2], [1.1, 0.1, 0.1]]), vehicle_xyz_m=(0.1, 0, 0), vehicle_yaw_rad=0)
        self.assertEqual(second.new_voxels, 1)
        self.assertEqual(second.explored_voxels, 3)

    def test_native_translation_threshold_and_auc(self):
        metric = MTAReCoverageAccumulator()
        point = np.asarray([[0.0, 0.0, 0.0]])
        metric.update(stamp_sec=10.0, registered_scan_xyz_m=point, vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)
        metric.update(stamp_sec=11.0, registered_scan_xyz_m=point + (0.5, 0, 0), vehicle_xyz_m=(0.1, 0, 0), vehicle_yaw_rad=0)
        last = metric.update(stamp_sec=12.0, registered_scan_xyz_m=point + (1.0, 0, 0), vehicle_xyz_m=(0.3, 0, 0), vehicle_yaw_rad=0)
        self.assertAlmostEqual(last.traveling_distance_m, 0.3)
        summary = metric.summary(budget_sec=3.0)
        self.assertAlmostEqual(summary["coverage_time_auc_m3_s"], 0.875)
        self.assertAlmostEqual(summary["mean_explored_volume_m3"], 0.875 / 3.0)

    def test_complete_map_recall_is_labeled_diagnostic(self):
        reference = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
        metric = MTAReCoverageAccumulator(reference_surface_xyz_m=reference)
        cycle = metric.update(stamp_sec=1.0, registered_scan_xyz_m=np.asarray([[0, 0, 0]], dtype=float), vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)
        self.assertEqual(cycle.complete_map_surface_recall_diagnostic, 0.5)
        self.assertIn("Diagnostic only", metric.summary(budget_sec=1.0)["reference_denominator_warning"])

    def test_rejects_nonfinite_and_duplicate_time(self):
        metric = MTAReCoverageAccumulator()
        with self.assertRaises(ValueError):
            metric.update(stamp_sec=1.0, registered_scan_xyz_m=np.asarray([[np.nan, 0, 0]]), vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)
        metric.update(stamp_sec=1.0, registered_scan_xyz_m=np.empty((0, 3)), vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)
        with self.assertRaises(ValueError):
            metric.update(stamp_sec=1.0, registered_scan_xyz_m=np.empty((0, 3)), vehicle_xyz_m=(0, 0, 0), vehicle_yaw_rad=0)


if __name__ == "__main__":
    unittest.main()
