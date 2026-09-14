import numpy as np
import pytest
from mtare_topo.teacher.gse_interior_section_candidates import nominate_interior_sections,section_frame_from_original_rings


def records():
    return np.array([[0,1,0,0,1,0,.05],[1,1,0,0,1,.05,.1],[2,1,0,0,1,1,1.05],
                     [3,1,0,0,1,2,2.05],[3,2,.01,.01,1,2,2.05]],float)


def test_gaps_and_multi_source_preserved():
    result=nominate_interior_sections(records())
    assert len(result['candidates'])==2 and result['ambiguous_return_indices']==[3]
    assert result['candidates'][0]['reference_arc_m']==pytest.approx(.075)
    assert all(not c['training_qualified'] for c in result['candidates'])


def test_order_density_and_residuals_do_not_choose_reference():
    rows=records();expected=nominate_interior_sections(rows)['candidates']
    assert nominate_interior_sections(rows[::-1])['candidates']==expected
    rows[:,2:4]=100
    assert nominate_interior_sections(rows)['candidates']==expected
    more=np.concatenate([rows,np.array([[10,1,0,0,1,0,.05]])])
    assert nominate_interior_sections(more)['candidates'][0]['reference_arc_m']==expected[0]['reference_arc_m']


def test_no_shift_from_zero_span_or_duplicate_record():
    row=np.array([[0,1,0,0,1,2,2]],float)
    assert nominate_interior_sections(row)['candidates'][0]['reference_arc_m'] is None
    with pytest.raises(ValueError):nominate_interior_sections(np.concatenate([row,row]))


def test_original_ring_center_and_tangent():
    t=np.arange(64)*2*np.pi/64
    rings=[np.column_stack([np.full(64,x),np.cos(t),np.sin(t)]) for x in [0,.05,.1]]
    v=np.concatenate([*rings,np.array([[0,0,0],[.1,0,0]])]);arc=np.r_[np.repeat([0,.05,.1],64),0,.1]
    s=section_frame_from_original_rings(v,arc,[.05,.1])
    assert np.allclose(s['center_m'],[.075,0,0]) and np.allclose(s['normal'],[1,0,0])
    with pytest.raises(ValueError):section_frame_from_original_rings(v,arc,[0,.1])
