from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from numcodecs import Blosc
import numpy as np
import zarr

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader
from mtare_topo.data.primitive_relation_training import PrimitiveRelationTrainingShard


def _dataset(group, name, values):
    data = np.asarray(values)
    group.create_dataset(name, data=data, chunks=(max(1, min(2, len(data))), *data.shape[1:]), compressor=Blosc(cname="zstd", clevel=1))


class PrimitiveRelationTrainingShardTest(unittest.TestCase):
    def _write_pair(self, root: Path, *, teacher_parent: str = "S01_C01") -> tuple[Path, Path]:
        sensor_path = root / "sensor.zarr"; teacher_path = root / "teacher.zarr"
        sensor = zarr.open_group(str(sensor_path), mode="w")
        sensor.attrs.update({"parent_id": "S01_C01", "partition": "fit", "geometry_realization": "ellipse", "sensor_shape": [16, 720], "maximum_range_m": 50.0, "student_pose_input_forbidden": True})
        ranges = np.full((7,16,720), 12.5, dtype=np.float32); valid = np.ones((7,16,720), dtype=np.uint8)
        _dataset(sensor,"range_m",ranges);_dataset(sensor,"valid_mask",valid)
        # Absolute pose arrays exist in P1a for Teacher/evaluation, but the reader must never return them as student input.
        _dataset(sensor,"sensor_xyz_m",np.arange(21,dtype=np.float64).reshape(7,3));_dataset(sensor,"axis_xyz_m",np.zeros((7,3),dtype=np.float64));_dataset(sensor,"yaw_deg",np.arange(7,dtype=np.float64))
        teacher = zarr.open_group(str(teacher_path), mode="w")
        teacher.attrs.update({"parent_id": teacher_parent, "partition": "fit", "geometry_realization": "ellipse", "maximum_slots": 32, "window_frames": 5, "student_identity_input_forbidden": True})
        frame_row=np.asarray([[0,1,2,3,4],[2,3,4,5,6]],dtype=np.int64); n=2
        _dataset(teacher,"frame_row",frame_row);_dataset(teacher,"source_global_sequence_index",np.asarray([10,11],dtype=np.int64));_dataset(teacher,"variant_global_sequence_index",np.asarray([30,31],dtype=np.int64))
        primitive_index=np.full((n,32),-1,dtype=np.int32);primitive_index[:,:2]=np.asarray([0,1])
        primitive_mask=(primitive_index>=0).astype(np.uint8)
        _dataset(teacher,"primitive_index",primitive_index);_dataset(teacher,"primitive_mask",primitive_mask)
        _dataset(teacher,"axis_control_current_sensor_m",np.zeros((n,32,3,3),dtype=np.float32));_dataset(teacher,"endpoint_half_axes_m",np.ones((n,32,2,2),dtype=np.float32));_dataset(teacher,"endpoint_shape_exponent",np.full((n,32,2),2,dtype=np.float32));_dataset(teacher,"support_ray_count",np.ones((n,32),dtype=np.int32));_dataset(teacher,"temporal_visibility",np.ones((n,5,32),dtype=np.uint8));_dataset(teacher,"frame_primitive_index",np.tile(primitive_index[:,None,:],(1,5,1)))
        temporal=np.full((n,5,32),-1,dtype=np.int8);temporal[:,:,:2]=np.asarray([0,1],dtype=np.int8)
        neighbors=np.full((n,32,2,3),-1,dtype=np.int8);neighbors[:,0,1,0]=2;neighbors[:,1,0,0]=1
        overlap=np.zeros((n,32,4),dtype=np.uint8);overlap[:,0,0]=2;overlap[:,1,0]=1
        _dataset(teacher,"temporal_destination",temporal);_dataset(teacher,"endpoint_neighbor",neighbors);_dataset(teacher,"disconnected_overlap_packed",overlap)
        translation=np.zeros((n,5,3),dtype=np.float32);translation[:,0,0]=-4
        yaw=np.zeros((n,5),dtype=np.float32);yaw[:,0]=-10
        _dataset(teacher,"relative_translation_current_sensor_m",translation);_dataset(teacher,"relative_yaw_current_sensor_deg",yaw)
        return sensor_path,teacher_path

    def test_student_interface_excludes_absolute_pose_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sensor,teacher=self._write_pair(Path(directory)); shard=PrimitiveRelationTrainingShard(sensor,teacher);example=shard[0]
            self.assertEqual(len(shard),2);self.assertEqual(example.student.range_valid.shape,(5,2,16,720));self.assertTrue(np.all(example.student.range_valid[:,0]==0.25));self.assertTrue(np.all(example.student.range_valid[:,1]==1.0))
            self.assertEqual(set(example.student.__dict__),{"range_valid","relative_translation_current_sensor_m","relative_yaw_current_sensor_deg"})
            self.assertNotIn("sensor_xyz_m",example.student.__dict__);self.assertNotIn("primitive_index",example.student.__dict__)
            self.assertEqual(example.source_global_sequence_index,10);self.assertEqual(example.variant_global_sequence_index,30)
            self.assertEqual(example.targets.endpoint_attachment[0,1,1,0],1);self.assertEqual(example.targets.disconnected_overlap[0,1],1)

    def test_negative_index_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sensor,teacher=self._write_pair(Path(directory)); shard=PrimitiveRelationTrainingShard(sensor,teacher)
            self.assertEqual(shard[-1].source_global_sequence_index,11)
            with self.assertRaises(IndexError): _=shard[2]

    def test_pair_attribute_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sensor,teacher=self._write_pair(Path(directory),teacher_parent="wrong")
            with self.assertRaisesRegex(ValueError,"parent_id"): PrimitiveRelationTrainingShard(sensor,teacher)

    def test_batch_loader_is_exact_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensor, teacher = self._write_pair(root)
            sensor_root = root / "sensors"; teacher_root = root / "teachers"
            sensor_root.mkdir(); teacher_root.mkdir()
            sensor.rename(sensor_root / "task.zarr")
            teacher.rename(teacher_root / "task.zarr")
            loader = PrimitiveRelationBatchLoader(sensor_root, teacher_root)
            self.assertEqual(len(loader), 2)
            batch = loader._read("task.zarr", np.asarray([1, 0], dtype=np.int64))
            shard = PrimitiveRelationTrainingShard(
                sensor_root / "task.zarr", teacher_root / "task.zarr",
            )
            for batch_row, shard_row in enumerate((1, 0)):
                example = shard[shard_row]
                np.testing.assert_array_equal(batch.range_valid[batch_row], example.student.range_valid)
                np.testing.assert_array_equal(batch.primitive_index[batch_row], example.targets.primitive_index)
                np.testing.assert_array_equal(batch.endpoint_attachment[batch_row], example.targets.endpoint_attachment)
                np.testing.assert_array_equal(batch.disconnected_overlap[batch_row], example.targets.disconnected_overlap)
            first = [value.source_global_sequence_index.tolist() for value in loader.iter_epoch(
                batch_size=1, seed=7, epoch=3, shuffle=True, block_size=2,
            )]
            second = [value.source_global_sequence_index.tolist() for value in loader.iter_epoch(
                batch_size=1, seed=7, epoch=3, shuffle=True, block_size=2,
            )]
            self.assertEqual(first, second)

    def test_batch_loader_rejects_invalid_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensor, teacher = self._write_pair(root)
            sensor_root = root / "sensors"; teacher_root = root / "teachers"
            sensor_root.mkdir(); teacher_root.mkdir()
            sensor.rename(sensor_root / "task.zarr")
            teacher.rename(teacher_root / "task.zarr")
            loader = PrimitiveRelationBatchLoader(sensor_root, teacher_root)
            with self.assertRaises(IndexError):
                loader._read("task.zarr", np.asarray([2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
