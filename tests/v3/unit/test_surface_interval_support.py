import numpy as np
from mtare_topo.teacher.gse_surface_interval_support import join_surface_interval_support
from mtare_topo.teacher.gse_surface_interval_support import describe_endpoint_support


def interval(*spans):return dict(status='REFERENCE_INTERVALS_ONLY',intervals_m=list(spans))


def row(ray,source,low,high,residual=0):return [ray,source,residual,residual,1,low,high]


def test_reentry_components_not_collapsed_by_source():
    r=join_surface_interval_support([row(0,0,2,3),row(1,0,12,13)],{0:interval([0,5],[10,15])})
    assert r['conditional_interval_index'].tolist()==[0,1]
    assert r['membership'] is None and not r['point_mask_qualified']


def test_multiple_sources_not_chosen_by_lower_residual():
    a=np.array([row(0,0,2,3,.000001),row(0,1,2,3,.03)])
    r=join_surface_interval_support(a,{0:interval([0,5]),1:interval([0,5])})
    assert r['row_state'].tolist()==[3,3]
    assert not len(r['single_source_interior_ray_indices'])
    a[:,2:4]=a[::-1,2:4]
    assert np.array_equal(r['row_state'],join_surface_interval_support(a,{0:interval([0,5]),1:interval([0,5])})['row_state'])


def test_boundary_no_axis_and_ambiguous_crossing_separate():
    r=join_surface_interval_support([row(0,0,0,1),row(1,0,6,7),row(2,1,2,3)],
        {0:interval([0,5]),1:dict(status='UNKNOWN',intervals_m=[])})
    assert r['row_state'].tolist()==[2,1,0]


def test_order_independence():
    a=np.array([row(10,0,2,3),row(4,0,3,4)])
    b={0:interval([0,5])}
    assert np.array_equal(join_surface_interval_support(a,b)['ray_indices'],join_surface_interval_support(a[::-1],b)['ray_indices'])


def test_source_cap_inside_roi_separate_without_relabeling():
    a=[row(0,0,0,0),row(1,0,19,20)]
    r=describe_endpoint_support(a,{0:interval([0,20])},{0:20})
    assert r['row_state'].tolist()==[2,2]
    assert r['source_endpoint_interval_hypothesis'].tolist()==[0,0]
    assert r['zero_width_source_cap_hypothesis'].tolist()==[True,False]
    assert not r['roi_crop_boundary_contact'].any()
    assert r['physical_terminal'] is None and r['structural_membership'] is None


def test_crop_contact_not_excused_by_source_endpoint():
    r=describe_endpoint_support([row(0,0,0,10)],{0:interval([0,10])},{0:20})
    assert r['source_endpoint_contact_bits'].tolist()==[1]
    assert r['roi_crop_boundary_contact'].tolist()==[True]
    assert r['source_endpoint_interval_hypothesis'].tolist()==[-1]


def test_source_end_outside_window_not_promoted():
    r=describe_endpoint_support([row(0,0,0,0)],{0:interval([5,15])},{0:20})
    assert r['row_state'].tolist()==[1]
    assert r['source_endpoint_interval_hypothesis'].tolist()==[-1]


def test_boundary_vertex_ambiguity_and_multisource_remain():
    r=describe_endpoint_support([row(0,0,0,0),row(0,1,0,0)],
        {0:dict(status='UNKNOWN',intervals_m=[]),1:interval([0,20])},{0:20,1:20})
    assert r['source_endpoint_interval_hypothesis'].tolist()==[-1,0]
    assert r['source_hypotheses_per_ray'].tolist()==[2]
    assert r['point_mask_qualified'] is False
