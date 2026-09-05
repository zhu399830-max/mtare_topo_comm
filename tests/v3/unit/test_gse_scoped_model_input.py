"""Synthetic input parity and no-teacher-input contract for frozen inference."""
import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_scoped_model_input import ScopedCompositionModelReader


TASK = "S01_fixture_C01__c1_mixed"
SELECTION = [{"task": TASK, "row_index": 1, "source_global_sequence_index": 17}]


def fixture(tmp_path):
    teacher = zarr.open_group(str(tmp_path / "teacher" / (TASK + ".zarr")), mode="w")
    sensor = zarr.open_group(str(tmp_path / "sensor" / (TASK + ".zarr")), mode="w")
    for group in (teacher, sensor):
        group.attrs.update({"parent_id": TASK.split("__")[0], "partition": "fit", "geometry_realization": "c1_mixed"})
    teacher.attrs.update({"student_identity_input_forbidden": True, "window_frames": 5})
    sensor.attrs.update({"student_pose_input_forbidden": True, "sensor_shape": [16, 720], "maximum_range_m": 50.})
    teacher.create_dataset("frame_row", data=np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]))
    teacher.create_dataset("source_global_sequence_index", data=np.array([16, 17]))
    teacher.create_dataset("relative_translation_current_sensor_m", data=np.zeros((2, 5, 3), dtype="f4"))
    teacher.create_dataset("relative_yaw_current_sensor_deg", data=np.zeros((2, 5), dtype="f4"))
    teacher.create_dataset("primitive_index", data=np.zeros((2, 32), dtype="i4"))
    sensor.create_dataset("range_m", data=np.broadcast_to(np.arange(7, dtype="f4")[:, None, None], (7, 16, 720)), chunks=(1, 16, 720))
    sensor.create_dataset("valid_mask", data=np.ones((7, 16, 720), dtype="u1"), chunks=(1, 16, 720))
    sensor.create_dataset("traversal_index", data=np.arange(7))
    return sensor, teacher


def reader(tmp_path, records=SELECTION):
    return ScopedCompositionModelReader(tmp_path / "sensor", tmp_path / "teacher", records)


def test_exact_legacy_input_math_without_teacher_geometry_or_global_pose(tmp_path):
    fixture(tmp_path)
    value = reader(tmp_path)
    result = value.read_task(TASK)
    assert len(result.student) == 1
    student = result.student[0]
    assert set(student.__dataclass_fields__) == {"range_valid", "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"}
    np.testing.assert_array_equal(student.range_valid[:, 0, 0, 0], np.arange(2, 7, dtype="f4") / np.float32(50.))
    assert student.range_valid.shape == (5, 2, 16, 720)
    assert result.source_sequence_indices.tolist() == [17]
    assert not any("primitive_index" in path or "traversal_index" in path for path in value.opened)
    assert not any("range_m/0." in path or "range_m/1." in path for path in value.opened)


def test_teacher_identity_mutation_cannot_change_forward_input(tmp_path):
    sensor, teacher = fixture(tmp_path)
    before = reader(tmp_path).read_task(TASK).student[0].range_valid
    teacher["primitive_index"][:] = 1234
    sensor["traversal_index"][:] = -999
    np.testing.assert_array_equal(reader(tmp_path).read_task(TASK).student[0].range_valid, before)


def test_nonselected_world_refused_before_io(tmp_path):
    with pytest.raises(ValueError, match="selection"):
        reader(tmp_path, [{**SELECTION[0], "task": "S01_fixture_C10__c1_mixed"}])


@pytest.mark.parametrize("corruption", ["future_order", "source_index", "current_pose", "valid_mask", "range_nan"])
def test_sensor_and_causal_contract_corruption_fails(tmp_path, corruption):
    sensor, teacher = fixture(tmp_path)
    if corruption == "future_order":
        teacher["frame_row"][1] = [2, 3, 4, 6, 5]
    elif corruption == "source_index":
        teacher["source_global_sequence_index"][1] = 18
    elif corruption == "current_pose":
        teacher["relative_yaw_current_sensor_deg"][1, 4] = 1
    elif corruption == "valid_mask":
        sensor["valid_mask"][2, 0, 0] = 2
    else:
        sensor["range_m"][2, 0, 0] = np.nan
    with pytest.raises(ValueError):
        reader(tmp_path).read_task(TASK)


def test_unselected_task_not_opened(tmp_path):
    fixture(tmp_path)
    value = reader(tmp_path)
    with pytest.raises(PermissionError):
        value.read_task("S02_fixture_C01__c1_mixed")
    assert not value.opened
