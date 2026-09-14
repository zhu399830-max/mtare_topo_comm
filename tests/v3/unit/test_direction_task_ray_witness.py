import numpy as np
from mtare_topo.teacher.direction_task_ray_witness import outward_section,crossed_sections,candidate_witnesses
from mtare_topo.topology.task_correspondence import mutual_unique_matches,cosine_matrix

def section():return dict(port_reference='node/edge/endpoint0',center_world_m=[10,0,0],normal_world=[1,0,0],inscribed_radius_m=2)

def test_axis_boundary_and_reverse_endpoint_geometry():
    s=outward_section([[0,0,0],[20,0,0]],[[2,3],[4,5]],[0,0,0])
    assert s['center_world_m']==[10,0,0] and s['inscribed_radius_m']==3
    assert outward_section([[20,0,0],[0,0,0]],[[4,5],[2,3]],[0,0,0]) is None

def test_multiple_crossings_and_short_source_remain_unknown():
    assert outward_section([[0,0,0],[20,0,0],[0,1,0]],[[2,2],[2,2]],[0,0,0]) is None
    assert outward_section([[0,0,0],[4,0,0]],[[2,2],[2,2]],[0,0,0]) is None

def test_blocked_hit_before_and_at_section_is_not_free_witness():
    hits=crossed_sections(np.zeros((4,3)),[[8,0,0],[10,0,0],[12,0,0],[12,0,5]],[section()])
    assert hits[:,0].tolist()==[False,False,True,False]

def test_reverse_ray_and_stacked_section_do_not_alias():
    upper={**section(),'port_reference':'upper','center_world_m':[10,0,5]}
    hits=crossed_sections([[0,0,0],[12,0,0]],[[12,0,0],[0,0,0]],[section(),upper])
    assert hits.tolist()==[[True,False],[False,False]]

def test_competition_unknown_not_nearest_or_most_rays():
    sections=[section(),{**section(),'port_reference':'other'}]
    r=candidate_witnesses([0,1],np.array([[1,1],[1,0]],bool),sections)
    assert r['conditional_port_reference'] is None and r['reason']=='COMPETING_SECTIONS'
    assert not r['training_qualified']

def test_same_observation_no_witness_is_unknown_not_background():
    r=candidate_witnesses([0],np.zeros((1,1),bool),[section()])
    assert r['conditional_port_reference'] is None and not r['reference_complete']

def test_mutual_selection_prevents_many_to_one_and_retains_unknown():
    assert mutual_unique_matches([[.9,.1],[.8,.2],[np.nan,np.nan]])=={0:0,1:None,2:None}
    assert mutual_unique_matches([[.5,.5]])=={0:None}
    assert mutual_unique_matches(np.empty((2,0)))=={0:None,1:None}

def test_permutations_change_indices_not_selection():
    a=np.array([[.9,.1],[.2,.8]])
    assert mutual_unique_matches(a)=={0:0,1:1}
    assert mutual_unique_matches(a[:,::-1])=={0:1,1:0}
    assert np.isnan(cosine_matrix([None],[[1,0]])).all()
