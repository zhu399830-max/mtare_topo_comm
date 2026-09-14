from dataclasses import replace
import numpy as np
import pytest
from test_gse_local_section_evidence_v1 import fixture
from mtare_topo.teacher.gse_window_opening_proposals_v1 import propose_window_openings


BOUNDARY=np.array([np.sqrt(100.-.1**2-.2**2),.1,.2])


def run(owners=None, mask=None, reverse=False, center=None):
    v,f,r=fixture()
    shift=BOUNDARY-np.array([.1,.35,.35])
    origins=r.origins_m+shift
    directions=r.directions.copy()
    if reverse:
        origins[8]=BOUNDARY+[1.,0.,0.];directions[8]=[-1.,0.,0.]
    r=replace(r,origins_m=origins,directions=directions,
              valid=r.valid if mask is None else np.array(mask,dtype=bool))
    out=propose_window_openings(v+shift,f,section_center_m=BOUNDARY if center is None else center,
        outward_direction=[1.,0.,0.],roi_center_m=[0.,0.,0.],rays=r,
        unique_return_source_index=np.zeros(9,dtype=int) if owners is None else np.array(owners),source_index=0)
    return out


def test_partial_boundary_can_propose_but_never_fills_unknown_fields():
    out=run(mask=[True]+[False]*7+[True]);p=out['proposals'][0]
    assert p['proposal_supported']
    assert p['observed_contour_cells'] < p['reference_contour_cells']
    assert p['width_m'] is p['height_m'] is p['anchor'] is p['membership'] is None
    assert not p['training_eligible'] and out['qualified_labels']==0


def test_no_boundary_is_not_a_supported_proposal():
    assert not run(mask=[False]*8+[True])['proposals'][0]['proposal_supported']


def test_other_source_and_unknown_returns_cannot_supply_contour():
    for other in (1,-1):
        assert not run(owners=[other]*8+[0])['proposals'][0]['proposal_supported']


def test_inward_crossing_does_not_support_outward_opening():
    assert not run(reverse=True)['proposals'][0]['proposal_supported']


def test_internal_constructed_cap_is_not_window_boundary():
    with pytest.raises(ValueError,match='10m ROI'):
        run(center=(0.,0.,0.))
