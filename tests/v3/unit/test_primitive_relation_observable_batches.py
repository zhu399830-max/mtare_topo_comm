from pathlib import Path
import tempfile

from numcodecs import Blosc
import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import pack_endpoint_observed
from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationBatchLoader
from tests.v3.unit.test_primitive_relation_training import PrimitiveRelationTrainingShardTest as _TrainingHelper


def _write_sidecar(path: Path, observed: np.ndarray, source: np.ndarray) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update({
        "schema_version": "primitive_attachment_observability_sidecar_v1",
        "support_band_m": .25, "hidden_pair_semantics": "unknown_never_negative",
        "endpoint_count": 64, "packed_endpoint_bytes": 8,
    })
    compressor = Blosc(cname="zstd", clevel=1)
    group.create_dataset("endpoint_observed_packed", data=pack_endpoint_observed(observed), chunks=(2, 8), compressor=compressor)
    group.create_dataset("source_global_sequence_index", data=source, chunks=(2,), compressor=compressor)


def test_observable_loader_binds_and_reads_sidecar() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); helper = _TrainingHelper()
        sensor, teacher = helper._write_pair(root)
        sensor_root = root / "sensors"; teacher_root = root / "teachers"; sidecar_root = root / "sidecars"
        sensor_root.mkdir(); teacher_root.mkdir(); sidecar_root.mkdir()
        sensor.rename(sensor_root / "task.zarr"); teacher.rename(teacher_root / "task.zarr")
        observed = np.zeros((2, 32, 2), dtype=np.uint8)
        observed[:, 0, 1] = 1; observed[:, 1, 0] = 1
        _write_sidecar(sidecar_root / "task.zarr", observed, np.asarray([10, 11], dtype=np.int64))
        loader = ObservablePrimitiveRelationBatchLoader(sensor_root, teacher_root, sidecar_root)
        batch = loader._read("task.zarr", np.asarray([1, 0]))
        np.testing.assert_array_equal(batch.endpoint_observed, observed[[1, 0]])
        assert batch.base.source_global_sequence_index.tolist() == [11, 10]


def test_observable_loader_rejects_sequence_identity_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); helper = _TrainingHelper()
        sensor, teacher = helper._write_pair(root)
        sensor_root = root / "sensors"; teacher_root = root / "teachers"; sidecar_root = root / "sidecars"
        sensor_root.mkdir(); teacher_root.mkdir(); sidecar_root.mkdir()
        sensor.rename(sensor_root / "task.zarr"); teacher.rename(teacher_root / "task.zarr")
        _write_sidecar(sidecar_root / "task.zarr", np.zeros((2, 32, 2), dtype=np.uint8), np.asarray([10, 99], dtype=np.int64))
        try:
            ObservablePrimitiveRelationBatchLoader(sensor_root, teacher_root, sidecar_root)
        except ValueError as error:
            assert "identity" in str(error)
        else:
            raise AssertionError("misaligned source sequence identity was accepted")
