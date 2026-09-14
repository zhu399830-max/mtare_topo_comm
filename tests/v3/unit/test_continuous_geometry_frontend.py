import hashlib
import io
import numpy as np
import pytest
from mtare_topo.integration.continuous_geometry_frontend import replay_package


def package():
    data=io.BytesIO();frames=np.array([[0,1,2,3,4],[1,2,3,4,5]],np.int32)
    np.savez_compressed(data,ranges_m=np.zeros((2,5,16,720),np.float32),
        valid_mask=np.zeros((2,5,16,720),np.uint8),
        relative_translation_current_sensor_m=np.zeros((2,5,3),np.float32),
        relative_yaw_current_sensor_deg=np.zeros((2,5),np.float32),
        frame_rows=frames,source_sequence_ids=np.array([10,11],np.int64))
    payload=data.getvalue()
    return payload,dict(task='S09_flat_complex_C04__ellipse',observations=2,
        bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
        source_frames=frames.tolist(),source_sequence_ids=[10,11])


POLICY=dict(max_residual_m=.01,min_crossing_sine=.1,
            endpoint_tolerance_m=1e-8,maximum_candidates=32)


def test_empty_real_interface_never_invents_opening_or_motion():
    payload,manifest=package()
    rows=replay_package(payload,manifest,composition_policy=POLICY)
    assert len(rows)==2
    assert all(not r['structures'] and not r['primitives'] and not r['traversed_edges'] for r in rows)
    assert rows[1]['source_frame_keys'][-1].endswith('frame:5')
    np.testing.assert_array_equal(rows[1]['sensor_to_local_odometry'],np.eye(4))


def test_input_drift_rejected():
    payload,manifest=package()
    manifest['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA'):
        replay_package(payload,manifest,composition_policy=POLICY)
