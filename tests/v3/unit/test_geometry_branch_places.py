from copy import deepcopy
import numpy as np
from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces

def runtime():return GeometryBranchPlaces(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3)
def record(i):return dict(timestamp=i,coordinate_frame='map',sensor_to_local_odometry=np.eye(4).tolist())
def proposals():return [dict(axis_start_xyz_m=[0,0,0],axis_target_xyz_m=p,source_refs=['observation/ray:1']) for p in ([4,0,0],[-4,0,0],[0,4,0])]

def test_geometry_causes_place_not_travel_spacing():
    r=runtime()
    for i in range(3):s=r.observe(record(i),proposals(),navigation_node_id=0)
    assert s['status']=='supported_observation'
    assert len(r.snapshot()['places'])==1
    assert not r.snapshot()['places'][0]['physical_junction_center']

def test_two_directions_or_duplicate_axes_not_junction():
    r=runtime();p=proposals()[:2]
    for i in range(3):r.observe(record(i),p+[deepcopy(p[0])],navigation_node_id=0)
    assert r.snapshot()['places']==[]

def test_disappearing_branch_remains_unattempted_and_prefix_immutable():
    r=runtime()
    for i in range(3):r.observe(record(i),proposals(),navigation_node_id=0)
    snapshot=r.snapshot();r.observe(record(3),proposals()[:2],navigation_node_id=1)
    assert snapshot==r.snapshot()
    assert len(snapshot['places'][0]['frontiers'])==3

def test_attempt_never_manufactures_edge():
    r=runtime()
    for i in range(3):r.observe(record(i),proposals(),navigation_node_id=0)
    r.record_attempt(0,0,outcome='requested',stamp_sec=3)
    r.record_attempt(0,0,outcome='xy_arrived',stamp_sec=4)
    assert r.snapshot()['verified_edges']==[]
    assert not r.snapshot()['places'][0]['frontiers'][0]['connection_verified']


def test_shared_return_continuity_does_not_split_at_observer_radius():
    r=runtime();p=proposals()
    for j,item in enumerate(p):item['source_refs']=['persistent-ray:'+str(j)]
    for i in range(3):r.observe(record(i),p,navigation_node_id=0)
    moved=record(3);moved['sensor_to_local_odometry'][0][3]=4.1
    q=deepcopy(p)
    for item in q:
        item['axis_target_xyz_m'][0]+=4.1;item['axis_start_xyz_m'][0]+=4.1
    r.observe(moved,q,navigation_node_id=1)
    assert len(r.snapshot()['places'])==1
    assert len(r.snapshot()['places'][0]['frontiers'])==3


def test_unrelated_returns_do_not_bridge_distant_place():
    r=runtime()
    for i in range(3):r.observe(record(i),proposals(),navigation_node_id=0)
    moved=record(3);moved['sensor_to_local_odometry'][0][3]=10
    q=proposals()
    for item in q:item['source_refs']=['new-only']
    r.observe(moved,q,navigation_node_id=1)
    assert len(r.snapshot()['places'])==2

def retaining_runtime():
    return GeometryBranchPlaces(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,
        support_observations=3,retain_matching_partial_observations=True)

def test_partial_observation_keeps_identity_but_never_counts_as_support():
    r=retaining_runtime()
    for i in range(5):
        s=r.observe(record(i),proposals() if i%2==0 else proposals()[:2],navigation_node_id=0)
        if i in (1,3):
            assert s['added_support'] is False
            assert r.places[0]['state']=='provisional'
    assert s['status']=='supported_observation' and len(r.places)==1
    assert r.places[0]['orders']==[0,2,4]

def test_partial_mismatch_or_leaving_radius_cannot_bridge_places():
    for distant in (True,False):
        r=retaining_runtime();r.observe(record(0),proposals(),navigation_node_id=0)
        moved=record(1);p=proposals()[:2]
        if distant:moved['sensor_to_local_odometry'][0][3]=4.1
        else:p[0]['axis_target_xyz_m']=[0,-4,0]
        r.observe(moved,p,navigation_node_id=0)
        r.observe(record(2),proposals(),navigation_node_id=0)
        assert len(r.places)==2 and all(x['state']=='provisional' for x in r.places)
