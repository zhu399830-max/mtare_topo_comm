from pathlib import Path
import tempfile

from numcodecs import Blosc
import numpy as np
import pytest
import torch
import zarr

from mtare_topo.data.primitive_composition_anchor_batches import (
    CompositionAnchorBatchLoader,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    composition_anchor_numpy_batch_to_torch,
)
from tests.v3.unit.test_primitive_relation_observable_batches import (
    _write_sidecar as _write_observability,
)
from tests.v3.unit.test_primitive_relation_training import (
    PrimitiveRelationTrainingShardTest as _TrainingHelper,
)


def _write_anchor(path: Path, value: np.ndarray, source: np.ndarray) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update({
        "schema_version": "primitive_composition_anchor_target_sidecar_v1",
        "coordinate_frame": "current_sensor_xyz_yaw_only",
        "inactive_slot_value": 0.0,
        "student_construction_identity_input_forbidden": True,
    })
    compressor = Blosc(cname="zstd", clevel=1)
    group.create_dataset(
        "anchor_current_sensor_m", data=value, chunks=(2, 32, 2, 3),
        compressor=compressor,
    )
    group.create_dataset(
        "source_global_sequence_index", data=source, chunks=(2,), compressor=compressor,
    )


def _roots(root: Path):
    helper = _TrainingHelper()
    sensor, teacher = helper._write_pair(root)
    roots = [root / name for name in ("sensors", "teachers", "observability", "anchors")]
    for value in roots:
        value.mkdir()
    sensor.rename(roots[0] / "task.zarr")
    teacher.rename(roots[1] / "task.zarr")
    source = np.asarray([10, 11], dtype=np.int64)
    observed = np.zeros((2, 32, 2), dtype=np.uint8)
    observed[:, :2] = 1
    _write_observability(roots[2] / "task.zarr", observed, source)
    anchors = np.zeros((2, 32, 2, 3), dtype=np.float32)
    anchors[:, :2] = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    _write_anchor(roots[3] / "task.zarr", anchors, source)
    return roots, anchors


def test_composition_anchor_loader_reads_aligned_targets():
    with tempfile.TemporaryDirectory() as directory:
        roots, anchors = _roots(Path(directory))
        loader = CompositionAnchorBatchLoader(*roots)
        batch = loader._read("task.zarr", np.asarray([1, 0]))
        np.testing.assert_array_equal(batch.anchor_current_sensor_m, anchors[[1, 0]])
        torch_batch = composition_anchor_numpy_batch_to_torch(batch, device=torch.device("cpu"))
        torch.testing.assert_close(
            torch_batch.anchor_current_sensor_m,
            torch.from_numpy(anchors[[1, 0]]),
        )


def test_composition_anchor_loader_rejects_sequence_drift():
    with tempfile.TemporaryDirectory() as directory:
        roots, _ = _roots(Path(directory))
        group = zarr.open_group(str(roots[3] / "task.zarr"), mode="a")
        group["source_global_sequence_index"][:] = (10, 99)
        with pytest.raises(ValueError, match="identity"):
            CompositionAnchorBatchLoader(*roots)


def test_composition_anchor_loader_rejects_nonzero_inactive_slot():
    with tempfile.TemporaryDirectory() as directory:
        roots, _ = _roots(Path(directory))
        group = zarr.open_group(str(roots[3] / "task.zarr"), mode="a")
        group["anchor_current_sensor_m"][0, 8, 0, 0] = 1.0
        loader = CompositionAnchorBatchLoader(*roots)
        with pytest.raises(ValueError, match="inactive"):
            loader._read("task.zarr", np.asarray([0]))
