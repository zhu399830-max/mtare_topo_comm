from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.data.continuous_model_input_export_v1 import encode_and_validate


def fixture():
    return SimpleNamespace(task='S09_flat_complex_C04__ellipse',
        ranges_m=np.ones((10,5,16,720),dtype=np.float32),
        valid_mask=np.ones((10,5,16,720),dtype=np.uint8),
        relative_translation_current_sensor_m=np.zeros((10,5,3),dtype=np.float32),
        relative_yaw_current_sensor_deg=np.zeros((10,5),dtype=np.float32),
        frame_rows=np.array([list(range(i,i+5)) for i in range(4012,4022)],dtype=np.int32),
        source_sequence_ids=np.arange(158206,158216,dtype=np.int64))


def test_existing_six_field_encoder_binding_all_ten_windows():
    b,h=encode_and_validate(fixture());assert b and len(h)==64


def test_shared_source_frame_mismatch_rejected():
    f=fixture();f.ranges_m[1,0,0,0]=2
    with pytest.raises(ValueError,match='overlap differs'):encode_and_validate(f)
