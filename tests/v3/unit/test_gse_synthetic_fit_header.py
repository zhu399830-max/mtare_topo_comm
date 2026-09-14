import pytest
from mtare_topo.data.gse_synthetic_fit_example import cache_header
from mtare_topo.representation.gse_surface_training_v1 import _validate_header


def test_cache_header_uses_training_schema_not_loss_frame_spelling():
    row=dict(case_id='fixture',frame_rows=[0,1,2,3,4])
    header=cache_header(row,'a'*64,'b'*64)
    assert header['coordinate_frame']=='current_sensor'
    _validate_header(header,'b'*64)
    header['coordinate_frame']='current_sensor_m'
    with pytest.raises(ValueError):_validate_header(header,'b'*64)


def test_header_rejects_invalid_source_frames_before_training():
    with pytest.raises(ValueError):cache_header(dict(case_id='fixture',frame_rows=[0,1,2,3,3]),'a'*64,'b'*64)
