import numpy as np
import pytest
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.evaluation.gse_saved_negative_witness_audit import audit_saved_negative_witnesses


def fixture():
    source=dict(task='synthetic',source_sequence_id=0,frame_rows=list(range(5)))
    valid=np.zeros(57600,dtype=bool);valid[:2]=True
    codes=np.zeros(57600,dtype=np.uint16);codes[:2]=[1,2]
    bundle=dict(source=source,student=dict(valid_mask=valid,ranges_m=np.ones(57600,dtype=np.float32)),
        sensor_teacher_only={'primitive_membership_code':codes},
        codebook_teacher_only=dict(primitive_ids=['terminal','opening'],source_sets=[[],[0],[1]]))
    proof=dict(opening_index=0,anchor_index=1,junction_node_teacher_only='junction',
        cap_entering_ray_indices=[0],cap_source_interior_ray_indices=[],opening_junction_ray_indices=[1],
        terminal_outgoing_opening_ray_indices=[],terminal_continuation_sources=['terminal'],opening_continuation_sources=['opening'])
    record=dict(membership=[[True,False]])
    target=dict(record=record,target_record_sha256=canonical_sha(record),source_binding={'source':source},
        teacher_provenance=dict(terminal_anchor_start=1,terminal_nonmembership=[proof],
        terminals=[dict(endpoint_key_teacher_only=['terminal',0],witnesses=[dict(ray_index=0,stored_t=1.,source_frame_index=0)])],
        anchors=[{'node_id_teacher_only':'junction'}],relations=[dict(anchor_index=0,ray_indices=[1])],
        openings=[{'crossing_ray_indices':[1]}]))
    return target,bundle


def test_consistency_is_not_qualification():
    report=audit_saved_negative_witnesses(*fixture())
    assert report['checked_negative_candidates']==1
    assert not report['full_label_qualification']
    assert not report['geometric_surface_intersection_rechecked']


@pytest.mark.parametrize('defect',['invalid','owner','range','frame','crossing','missing','overlap'])
def test_independent_checks_reject_corrupted_witnesses(defect):
    target,bundle=fixture();p=target['teacher_provenance']
    if defect=='invalid':bundle['student']['valid_mask'][1]=False
    elif defect=='owner':bundle['sensor_teacher_only']['primitive_membership_code'][1]=1
    elif defect=='range':p['terminals'][0]['witnesses'][0]['stored_t']=2.
    elif defect=='frame':p['terminals'][0]['witnesses'][0]['source_frame_index']=1
    elif defect=='crossing':p['openings'][0]['crossing_ray_indices']=[]
    elif defect=='missing':p['terminal_nonmembership']=[]
    else:p['terminal_nonmembership'][0]['opening_continuation_sources'].append('terminal')
    with pytest.raises(ValueError):audit_saved_negative_witnesses(target,bundle)
