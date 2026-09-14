from mtare_topo.teacher.gse_local_axis_intervals import local_axis_intervals,return_interval_candidates


def test_straight_inside_interval():
    r=local_axis_intervals([[-20.,0,0],[20.,0,0]],center_m=[0,0,0])
    assert r['intervals_m']==[[10.,30.]]
    assert r['membership'] is None and not r['physical_traversability']


def test_one_source_has_two_separate_window_intervals():
    r=local_axis_intervals([[-20.,0,0],[20.,0,0],[20.,4,0],[-20.,4,0]],center_m=[0,0,0])
    assert len(r['intervals_m'])==2
    assert r['intervals_m'][0][1]<r['intervals_m'][1][0]
    assert return_interval_candidates(r)['status']=='UNKNOWN'


def test_ray_source_identity_cannot_supply_return_arc():
    r=local_axis_intervals([[-20.,0,0],[20.,0,0]],center_m=[0,0,0])
    a=return_interval_candidates(r)
    assert a['reason']=='MISSING_RETURN_ARC_EVIDENCE'
    b=return_interval_candidates(r,independently_bound_arc_range_m=[15,16])
    assert b['interval_indices']==[0] and b['status']=='UNIQUE_REFERENCE_INTERVAL_ONLY'
    assert b['membership'] is None and not b['point_mask_qualified']


def test_boundary_or_broad_range_does_not_force_membership():
    r=local_axis_intervals([[-20.,0,0],[20.,0,0]],center_m=[0,0,0])
    assert return_interval_candidates(r,independently_bound_arc_range_m=[10,11])['status']=='UNKNOWN'
    assert return_interval_candidates(r,independently_bound_arc_range_m=[0,40])['status']=='UNKNOWN'
