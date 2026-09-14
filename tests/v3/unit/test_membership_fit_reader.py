import gzip
import hashlib
import json
from dataclasses import fields
import numpy as np
import pytest
from mtare_topo.data.gse_membership_fit_reader import (
    STUDENT_FIELDS, load_student_window, load_partial_reference)


def payload():
    return dict(ranges_m=np.ones((5,16,720),np.float32),
        valid_mask=np.ones((5,16,720),np.uint8),
        relative_translation_current_sensor_m=np.zeros((5,3),np.float32),
        relative_yaw_current_sensor_deg=np.zeros(5,np.float32))


def saved(tmp_path, values, **extra):
    path=tmp_path/'input.npz';np.savez(path,**values)
    return dict(student_path=path.name,student_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layout='saved_single_window',decoded_observations=1,frame_rows=[0,1,2,3,4],**extra)


def test_only_sensor_fields_and_readonly(tmp_path):
    b=saved(tmp_path,payload());w=load_student_window(tmp_path,b)
    assert tuple(f.name for f in fields(w))==STUDENT_FIELDS
    assert not w.ranges_m.flags.writeable


def test_no_teacher_fields(tmp_path):
    b=saved(tmp_path,dict(payload(),teacher_center=np.zeros(3)))
    with pytest.raises(ValueError,match='extra fields'):
        load_student_window(tmp_path,b)


def test_batch_identity_and_collateral_count(tmp_path):
    p={k:np.stack([v,v*0]) for k,v in payload().items()}
    p.update(frame_rows=np.array([[0,1,2,3,4],[5,6,7,8,9]]),source_sequence_ids=np.array([7,9]))
    b=saved(tmp_path,p);b.update(layout='saved_task_batch',decoded_observations=2,
        row_index=1,source_sequence_id=9,frame_rows=[5,6,7,8,9])
    assert not load_student_window(tmp_path,b).ranges_m.any()
    b['source_sequence_id']=7
    with pytest.raises(ValueError,match='identity'):
        load_student_window(tmp_path,b)


def test_hash_drift(tmp_path):
    b=saved(tmp_path,payload());b['student_sha256']='0'*64
    with pytest.raises(ValueError,match='hash drift'):
        load_student_window(tmp_path,b)


def test_reference_retains_unknown_and_all_relations(tmp_path):
    target={'record':{'source_frame_indices':[0,1,2,3,4],
        'coordinate_frame':'current_sensor_m','membership':[[True,None],[False,None]]}}
    path=tmp_path/'target.gz';path.write_bytes(gzip.compress(json.dumps({'produced_targets':target}).encode()))
    b=dict(target_path=path.name,target_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frame_rows=[0,1,2,3,4])
    assert load_partial_reference(tmp_path,b)==target
