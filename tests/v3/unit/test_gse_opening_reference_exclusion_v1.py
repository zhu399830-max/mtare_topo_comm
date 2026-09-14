from types import SimpleNamespace
import numpy as np
import pytest
from test_gse_reference_exclusion_binding_v1 import fixture
from mtare_topo.teacher.gse_opening_reference_exclusion_v1 import (
    all_window_reference_positions,bound_opening_reference_exclusion)
from mtare_topo.teacher.gse_reference_exclusion_v1 import reference_exclusion


def primitive(points):
    return SimpleNamespace(centerline_xyz_m=np.asarray(points,dtype=np.float64))


def test_all_roots_keep_reverse_and_stacked_without_visibility_filter():
    p=primitive([[-20,0,0],[20,0,0]])
    reverse=primitive([[20,0,0],[-20,0,0]])
    upper=primitive([[-20,0,6],[20,0,6]])
    roots=all_window_reference_positions([p,reverse,upper],center_world_m=np.zeros(3),yaw_deg=0.)
    assert roots.shape==(6,3)
    assert sum(np.all(roots==[10,0,0],axis=1))==2
    assert sum(roots[:,2]==6)==2
    result=reference_exclusion(query_xyz_m=np.array([[8,0,6.]]),observed_state=np.array([1]),
        all_anchor_xyz_m=roots,inventory_complete=True,matching_radius_m=1.)
    assert result['reference_negative_mask']==[False]


def test_tangent_or_vertex_ambiguity_never_becomes_empty_complete():
    for p in [primitive([[-20,10,0],[20,10,0]]),primitive([[-10,0,0],[20,0,0]])]:
        with pytest.raises(ValueError,match='ambiguous'):
            all_window_reference_positions([p],center_world_m=np.zeros(3),yaw_deg=0.)


def test_actual_projection_binding_and_unknown_mask_protocol():
    # Synthetic ranges validate provenance mechanics, not mesh realism.
    b,g,q,binding=fixture()
    result=bound_opening_reference_exclusion(b,g,np.concatenate((q,[[9,9,9]])),
        expected_binding=binding,matching_radius_m=1.)
    assert result['reference_count']==0
    assert result['reference_negative_mask']==[True,False]
    assert result['reference_kind']=='window_opening'
    assert not result['training_eligible'] and not result['physical_traversability']


def test_wrong_source_stops_before_opening_exclusion():
    b,g,q,binding=fixture();binding['source']['task']='different'
    with pytest.raises(ValueError,match='binding'):
        bound_opening_reference_exclusion(b,g,q,expected_binding=binding,matching_radius_m=1.)
