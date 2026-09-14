from copy import deepcopy
import numpy as np
import pytest
from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay


def record(i,z=0.):
    pose=np.eye(4);pose[0,3]=i
    return dict(schema_version='geometry_structure_trace_v1',coordinate_frame='odom',
        timestamp=i,sensor_to_local_odometry=pose.tolist(),source_frame_keys=[str(f) for f in range(i,i+5)],
        structures=[],primitives=[dict(index=0,axis_controls_world_m=[[-8,0,0],[0,0,0],[8,0,z]],
        source_rays=[dict(frame_key=str(i),ray_index=0)])])


def runtime():return GeometryNavigationReplay(anchor_spacing_m=2.,lookahead_m=4.)


def test_current_sector_support_is_directional_not_axis_wide():
    row=record(0);row['primitives'][0]['current_sector_columns']=[0]
    proposals=runtime().update(row)['proposals']
    assert len(proposals)==2
    for p in proposals:
        assert p['current_direction_supported']==(p['axis_target_xyz_m'][0]>0)
    # Old rays still fit the axis, but cannot claim a current observation.
    row['primitives'][0]['current_sector_columns']=[]
    assert all(p['current_direction_supported'] is False for p in runtime().update(row)['proposals'])
    del row['primitives'][0]['current_sector_columns']
    assert all(p['current_direction_supported'] is None for p in runtime().update(row)['proposals'])


def test_geometry_to_target_to_graph_keeps_middle_nodes():
    r=runtime()
    for i in range(5):r.update(record(i))
    s=r.snapshot()
    assert len(s['nodes'])==3 and len(s['edges'])==2
    assert [(e['source'],e['target']) for e in s['edges']]==[(0,1),(1,2)]
    assert all(d['geometry_used'] and not d['executed'] for d in s['decisions'])


def test_actual_geometry_changes_selected_target():
    a=runtime().update(record(0));b=runtime().update(record(0,4.))
    assert a['selected']['axis_target_xyz_m']!=b['selected']['axis_target_xyz_m']


def test_prefix_and_snapshots_immutable():
    r=runtime();r.update(record(0));prefix=r.snapshot()
    r.update(record(1));assert prefix==runtime_after_one()


def runtime_after_one():
    r=runtime();r.update(record(0));return r.snapshot()


def test_no_cross_segment_edge():
    r=runtime();r.update(record(0));r.update(record(10),continuous=False)
    assert len(r.snapshot()['nodes'])==2 and r.snapshot()['edges']==[]


def test_missing_frames_and_variant_merge_rejected():
    r=runtime();r.update(record(0))
    with pytest.raises(ValueError):r.update(record(3))
    other=record(1);other['coordinate_frame']='other_variant'
    with pytest.raises(ValueError):r.update(other)
