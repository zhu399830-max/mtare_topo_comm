import hashlib
import io
import numpy as np
import pytest
from mtare_topo.data.multiview_teacher_reader_v1 import student_from_reference


def fixture():
    arrays=dict(ranges_m=np.full((2,5,16,720),3.,np.float32),valid_mask=np.ones((2,5,16,720),np.uint8),
        relative_translation_current_sensor_m=np.zeros((2,5,3),np.float32),relative_yaw_current_sensor_deg=np.zeros((2,5),np.float32),
        frame_rows=np.array([list(range(10,15)),list(range(11,16))],np.int32),source_sequence_ids=np.array([20,21],np.int64))
    b=io.BytesIO();np.savez_compressed(b,**arrays);raw=b.getvalue()
    ref=dict(source=dict(task='S01_flat_tree_small_C01__ellipse',source_sequence_id=21,frame_rows=list(range(11,16))),
        input_row=1,observation_count=2,input_sha256=hashlib.sha256(raw).hexdigest())
    return raw,ref,arrays


def test_original_raw_six_fields_preserved():
    raw,ref,arrays=fixture();student=student_from_reference(raw,ref)
    assert set(student)==set(arrays)
    for k in student:np.testing.assert_array_equal(student[k],arrays[k][1])
    assert student['ranges_m'][0,0,0]==3. # not normalized by50


def test_wrong_package_row_and_teacher_extra_rejected():
    raw,ref,arrays=fixture()
    with pytest.raises(ValueError):student_from_reference(raw,dict(ref,input_row=0))
    with pytest.raises(ValueError):student_from_reference(raw,dict(ref,input_sha256='0'*64))
    b=io.BytesIO();np.savez_compressed(b,**arrays,teacher_node_id=np.array([1]))
    wrong=b.getvalue()
    with pytest.raises(ValueError,match='six-field'):student_from_reference(wrong,dict(ref,input_sha256=hashlib.sha256(wrong).hexdigest()))
