import numpy as np
import pytest
from mtare_topo.teacher.gse_reference_exclusion_v1 import reference_exclusion


def run(points,states,anchors,complete=True):
    return reference_exclusion(query_xyz_m=np.array(points,dtype=float).reshape(-1,3),
        observed_state=np.array(states,dtype=np.uint8),all_anchor_xyz_m=np.array(anchors,dtype=float).reshape(-1,3),
        inventory_complete=complete,matching_radius_m=4.)


def test_observed_reference_exclusion_supplies_a_nonempty_negative():
    r=run([[8,0,0],[8,1,0]],[1,0],[[0,0,0]])
    assert r['reference_negative_mask']==[True,False]
    assert not r['whole_region_complete'] and not r['training_eligible']


def test_hidden_reference_must_not_be_dropped_from_inventory():
    r=run([[8,0,0]],[1],[[0,0,0],[8,0,0]])
    assert r['reference_negative_mask']==[False]
    assert r['reason']==['POSSIBLE_REFERENCE_MATCH_KEEP_UNKNOWN']


def test_empty_positive_list_is_not_an_exhaustive_inventory():
    assert run([[8,0,0]],[1],[],complete=False)['reference_negative_mask']==[False]


def test_match_boundary_and_three_dimensional_distance():
    assert run([[4,0,0],[0,0,5]],[1,2],[[0,0,0]])['reference_negative_mask']==[False,True]


def test_completely_unobserved_space_stays_unknown_even_with_no_references():
    assert run([[0,0,0]],[0],[])['reference_negative_mask']==[False]


def test_invalid_states_rejected():
    with pytest.raises(ValueError):run([[0,0,0]],[3],[])


def test_actual_ray_grid_distinguishes_before_return_surface_and_occlusion():
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
    from mtare_topo.teacher.gse_reference_exclusion_v1 import exclusion_from_ray_grid
    grid=build_surface_ray_grid(np.array([[.1,.1,.1]]),np.array([[8.1,.1,.1]]),
        np.array([True]),np.array([0]))
    result=exclusion_from_ray_grid(query_xyz_m=[[6.1,.1,.1],[8.1,.1,.1],[9.1,.1,.1],[6.,.1,.1]],
        grid=grid,all_anchor_xyz_m=[[0.,0.,0.]],inventory_complete=True,matching_radius_m=4.)
    assert result['query_observed_states']==[1,2,0,0]
    assert result['reference_negative_mask']==[True,True,False,False]
    assert not result['training_eligible']


def test_grid_tampering_is_rejected():
    from dataclasses import replace
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
    from mtare_topo.teacher.gse_reference_exclusion_v1 import exclusion_from_ray_grid
    grid=build_surface_ray_grid(np.empty((0,3)),np.empty((0,3)),np.array([],dtype=bool),np.array([],dtype=int))
    with pytest.raises(ValueError,match='seal'):
        exclusion_from_ray_grid(query_xyz_m=[[1.,1.,1.]],grid=replace(grid,content_sha256='bad'),
            all_anchor_xyz_m=[],inventory_complete=True,matching_radius_m=4.)
