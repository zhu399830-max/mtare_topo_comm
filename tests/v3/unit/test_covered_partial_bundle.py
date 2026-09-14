from copy import deepcopy
import numpy as np
import pytest
from test_covered_five_frame_bundle import fixture
from mtare_topo.data.covered_five_frame_bundle import assemble_five_frame_bundle
from mtare_topo.data.covered_partial_bundle import partial_bundle
from mtare_topo.teacher.source_witness_binding import permitted_sources


def args():
    case,book,frames=fixture()
    value=assemble_five_frame_bundle(case=case,indexed_frames=frames,codebook=book)
    student=value.pop('student');sensor=value.pop('diagnostic_only')
    value['reported_source_codebook']=sensor.pop('codebook')
    audits=[]
    for frame in range(5):
        reported=np.zeros((11520,5),dtype=bool);reported[:,:2]=True
        audits.append((frame,dict(reported=reported,possible_surface=reported.copy(),
            uncertain_operands=np.zeros_like(reported),valid=np.ones(11520,dtype=bool),
            surface_uniqueness_certified=np.zeros(11520,dtype=bool),
            numerical_geometry_certified=np.array(False))))
    return dict(case=case,metadata=value,student=student,sensor=sensor,indexed_audits=audits)


def test_existing_contract_no_scan_change_or_certification():
    a=args();out=partial_bundle(**a)
    assert set(out['student'])==set(a['student'])
    for k,v in a['student'].items():np.testing.assert_array_equal(out['student'][k],v)
    assert out['source_witness_required'] and not out['qualification']['training_eligible']
    assert not any(permitted_sources(out))
    out['student']['ranges_m'][:]=0
    assert a['student']['ranges_m'][0,0,0]==2


@pytest.mark.parametrize('defect',['order','pose','reported','certificate','student_id'])
def test_misalignment_fails(defect):
    a=args()
    if defect=='order':a['indexed_audits'].reverse()
    if defect=='pose':a['sensor']['sensor_xyz_m'][0,0]+=1
    if defect=='reported':a['indexed_audits'][0][1]['reported'][0,0]=False
    if defect=='certificate':a['indexed_audits'][0][1]['surface_uniqueness_certified'][0]=True
    if defect=='student_id':a['student']['node_id']=np.array([1])
    with pytest.raises(ValueError):partial_bundle(**a)
