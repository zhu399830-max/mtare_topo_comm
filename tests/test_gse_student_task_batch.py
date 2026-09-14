import hashlib
import numpy as np
import pytest
from mtare_topo.data.gse_membership_fit_reader import load_student_task_batch, load_student_window


@pytest.fixture
def cache(tmp_path):
    arrays = dict(ranges_m=np.ones((2, 5, 16, 720), np.float32),
                  valid_mask=np.ones((2, 5, 16, 720), np.uint8),
                  relative_translation_current_sensor_m=np.zeros((2, 5, 3), np.float32),
                  relative_yaw_current_sensor_deg=np.zeros((2, 5), np.float32),
                  frame_rows=np.arange(10, dtype=np.int32).reshape(2, 5),
                  source_sequence_ids=np.array([3, 8], dtype=np.int64))
    arrays['ranges_m'][1] *= 2
    path = tmp_path / 'task.npz'; np.savez_compressed(path, **arrays)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    bindings = [dict(task='synthetic', student_path='task.npz', student_sha256=sha,
                     layout='saved_task_batch', decoded_observations=2, row_index=i,
                     source_sequence_id=int(arrays['source_sequence_ids'][i]),
                     frame_rows=arrays['frame_rows'][i].tolist()) for i in range(2)]
    return tmp_path, bindings


def test_equal_existing_reader_and_immutable(cache):
    root, bindings = cache
    batch = load_student_task_batch(root, bindings)
    for window, binding in zip(batch, bindings):
        old = load_student_window(root, binding)
        assert set(vars(window)) == set(vars(old))
        for key, value in vars(window).items():
            assert np.array_equal(value, getattr(old, key))
            assert not value.flags.writeable


@pytest.mark.parametrize('defect', ['partial', 'reordered', 'identity', 'hash', 'mixed_task'])
def test_invalid_binding_rejected(cache, defect):
    root, bindings = cache
    if defect == 'partial': bindings = bindings[:1]
    elif defect == 'reordered': bindings = bindings[::-1]
    elif defect == 'identity': bindings[1]['frame_rows'][0] = 999
    elif defect == 'hash':
        for b in bindings: b['student_sha256'] = '0' * 64
    else: bindings[1]['task'] = 'another'
    with pytest.raises(ValueError): load_student_task_batch(root, bindings)
