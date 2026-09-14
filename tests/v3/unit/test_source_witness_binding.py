import copy
import numpy as np
import pytest
from mtare_topo.teacher.source_witness_binding import permitted_sources, named_sources
from mtare_topo.teacher.gse_directed_interface_evidence_v1 import directed_evidence


def fixture():
    return dict(student={'valid_mask': np.ones(2, dtype=np.uint8)},
        sensor_teacher_only={'primitive_membership_code': np.array([1,1], dtype=np.uint16)},
        codebook_teacher_only={'primitive_ids':['wall'], 'source_sets':[[],[0]]},
        source={'frame_rows':[0], 'source_sequence_id':'case'},
        source_witness_required=True,
        source_witness_teacher_only=dict(primitive_ids=['wall'], frame_rows=[0],
            source_sequence_id='case', mode='diagnostic', reported=np.ones((2,1),dtype=bool),
            possible_surface=np.array([[1],[0]],dtype=bool),
            uncertain_operands=np.zeros((2,1),dtype=bool),
            surface_uniqueness_certified=np.zeros(2,dtype=bool)))


def test_policy_binding_and_legacy():
    b=fixture(); before=copy.deepcopy(b)
    assert permitted_sources(b)==[[0],[]]
    assert named_sources(b)==[['wall'],[]]
    np.testing.assert_array_equal(b['student']['valid_mask'],before['student']['valid_mask'])
    b['source_witness_teacher_only']['mode']='certified'
    assert permitted_sources(b)==[[],[]]
    del b['source_witness_teacher_only']
    with pytest.raises(ValueError): permitted_sources(b)
    b['source_witness_required']=False
    assert permitted_sources(b)==[[0],[0]]


@pytest.mark.parametrize('key,value', [('primitive_ids',['other']),('frame_rows',[1]),
    ('source_sequence_id','other'),('reported',np.zeros((2,1),dtype=bool))])
def test_binding_drift_rejected(key,value):
    b=fixture();b['source_witness_teacher_only'][key]=value
    with pytest.raises(ValueError): permitted_sources(b)


def test_unknown_source_blocks_entry_not_geometric_departure():
    b=fixture()
    interfaces=[dict(interface_id_teacher_only=i,node_id_teacher_only='node',
        source_key_teacher_only='wall',inward_direction=[sign,0.,0.])
        for i,sign in enumerate((1.,-1.))]
    records=[dict(ray_index=1,interface_id_teacher_only=i,t=1.,inside_roi=True) for i in range(2)]
    out=directed_evidence(records,interfaces,directions=np.array([[1.,0,0],[1.,0,0]]),
        first_return=np.full(2,3,dtype=np.float32),valid=np.ones(2,dtype=bool),
        return_sources=named_sources(b))['interfaces']
    assert out[0]['entering_ray_indices']==[] and out[0]['unknown_ray_indices']==[1]
    assert out[1]['leaving_ray_indices']==[1]
