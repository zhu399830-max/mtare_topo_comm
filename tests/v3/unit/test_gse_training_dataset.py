from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from mtare_topo.data.gse_training_dataset import GSESequenceDataset


class GSETrainingDatasetTest(unittest.TestCase):
    @staticmethod
    def _dataset_root(root: Path) -> Path:
        artifacts = root / "artifacts"
        path = artifacts / "dataset/train/p.zarr"
        path.parent.mkdir(parents=True)
        group = zarr.open_group(str(path), mode="w")
        group.create_dataset("range_m", data=np.full((5, 16, 720), 5.0, dtype=np.float32))
        group.create_dataset("valid_mask", data=np.ones((5, 16, 720), dtype=np.uint8))
        group.create_dataset("local_frame_references", data=np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32))
        group.create_dataset("event_index", data=np.asarray([1], dtype=np.uint8))
        group.create_dataset("local_axis_robot", data=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        group.create_dataset("geometry", data=np.asarray([[4.0, 3.0, 0.0, 0.1]], dtype=np.float32))
        group.create_dataset("geometry_valid_mask", data=np.asarray([[1, 1, 1, 1]], dtype=np.uint8))
        group.create_dataset("association_identity", data=np.asarray([3], dtype=np.int64))
        group.create_dataset("association_valid_mask", data=np.asarray([1], dtype=np.uint8))
        group.create_dataset("exit_mask", data=np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.uint8))
        group.create_dataset("exit_heading_unit", data=np.asarray([[[0.0, 1.0]] + [[0.0, 0.0]] * 5], dtype=np.float32))
        group.create_dataset("exit_opening_width_m", data=np.asarray([[2.0, 0, 0, 0, 0, 0]], dtype=np.float32))
        group.create_dataset("exit_width_valid_mask", data=np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.uint8))
        group.create_dataset("exit_vertical_profile_m", data=np.zeros((1, 6, 4), dtype=np.float32))
        group.create_dataset("exit_identity", data=np.asarray([[7, -1, -1, -1, -1, -1]], dtype=np.int64))
        (artifacts / "sequence_manifest.jsonl").write_text(
            json.dumps(
                {
                    "observation_id": "p:o0",
                    "parent_id": "p",
                    "split": "train",
                    "world_sequence_row": 0,
                    "global_sequence_index": 7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return root

    def test_reader_reconstructs_five_frames_without_pose_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dataset = GSESequenceDataset(self._dataset_root(Path(raw)), "train")
            item = dataset[0]
            self.assertEqual(item["student"].shape, (5, 2, 16, 720))
            self.assertEqual(item["targets"]["exit_mask"].shape, (6,))
            self.assertNotIn("pose", item)
            self.assertNotIn("yaw", item)
            np.testing.assert_allclose(item["targets"]["local_axis"], [1.0, 0.0, 0.0])
            np.testing.assert_array_equal(dataset.association_labels(), [3])
            self.assertIs(dataset.association_labels(), dataset.association_labels())
            np.testing.assert_array_equal(dataset.event_labels(), [1])
            self.assertIs(dataset.event_labels(), dataset.event_labels())

    def test_training_roll_is_epoch_deterministic_and_validation_roll_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._dataset_root(Path(raw))
            dataset = GSESequenceDataset(root, "train", augment_azimuth=True, augmentation_seed=2)
            first = dataset[0]
            repeated = dataset[0]
            np.testing.assert_array_equal(first["targets"]["local_axis"], repeated["targets"]["local_axis"])
            first_shift = dataset._shift(7)
            self.assertEqual(first_shift % 4, 0)
            dataset.set_epoch(1)
            second_shift = dataset._shift(7)
            self.assertNotEqual(first_shift, second_shift)
            self.assertEqual(second_shift % 4, 0)
            with self.assertRaises(ValueError):
                GSESequenceDataset(root, "validation", augment_azimuth=True)

    def test_reader_rejects_corrupted_target_shape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self._dataset_root(Path(raw))
            path = root / "artifacts/dataset/train/p.zarr"
            group = zarr.open_group(str(path), mode="a")
            del group["exit_vertical_profile_m"]
            group.create_dataset(
                "exit_vertical_profile_m",
                data=np.zeros((1, 6, 3), dtype=np.float32),
            )
            with self.assertRaises(RuntimeError):
                _ = GSESequenceDataset(root, "train")[0]


if __name__ == "__main__":
    unittest.main()
