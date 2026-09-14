from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.data.multiview_missing_input_export_v1 import encode_and_validate


def fixture():
    f=np.array([np.arange(i,i+5) for i in (10,11,14)],dtype=np.int32)
    v=SimpleNamespace(task='S01_flat_tree_small_C01__ellipse',ranges_m=np.ones((3,5,16,720),np.float32),
        valid_mask=np.ones((3,5,16,720),np.uint8),relative_translation_current_sensor_m=np.zeros((3,5,3),np.float32),
        relative_yaw_current_sensor_deg=np.zeros((3,5),np.float32),frame_rows=f,source_sequence_ids=np.array([20,21,24],np.int64))
    expected=[dict(frame_rows=f[i].tolist(),source_sequence_id=int(v.source_sequence_ids[i])) for i in range(3)]
    return v,expected


def test_gaps_from_reused_windows_allowed_with_exact_source_identity():
    v,e=fixture();raw,h=encode_and_validate(v,e);assert raw and len(h)==64
    with pytest.raises(ValueError,match='scoped'):encode_and_validate(v,e[::-1])


def test_shared_frame_mismatch_detected_even_across_missing_gap():
    v,e=fixture();v.ranges_m[2,0,0,0]=2
    with pytest.raises(ValueError,match='overlapping'):encode_and_validate(v,e)
