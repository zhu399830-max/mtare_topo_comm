import numpy as np
from mtare_topo.semantics.observed_direction_components import ray_components,decompose

def test_blocked_column_splits_without_forcing_two():
    mask=np.zeros((3,10),bool);mask[1,1:4]=True;mask[1,5:8]=True
    assert ray_components(mask)==[[11,12,13],[15,16,17]]
    mask[1,4]=True
    assert ray_components(mask)==[list(range(11,18))]

def test_azimuth_wraps_but_vertical_layers_and_diagonals_do_not():
    mask=np.zeros((4,8),bool);mask[0,[0,7]]=True;mask[3,0]=True;mask[1,1]=True
    assert ray_components(mask)==[[0,7],[9],[24]]

def test_no_component_pruning_and_empty_grid():
    mask=np.zeros((3,9),bool);mask[0,1]=True;mask[2,5]=True
    assert ray_components(mask)==[[1],[23]]
    assert ray_components(np.zeros((3,9),bool))==[]

def test_original_ray_partition_and_unknown_parent_retained():
    ranges=np.zeros((16,720));valid=np.ones_like(ranges,bool);ids=[3601,3602,3603,3604,3605]
    ranges.reshape(-1)[ids]=[12,12,5,12,12]
    parent=dict(source_refs=['f/ray:'+str(i) for i in ids],axis_start_xyz_m=[0,0,0],axis_target_xyz_m=[4,0,0])
    children,audit=decompose(ranges,valid,np.eye(4),'f',[parent])
    assert len(children)==2 and audit[0]['residual_ray_indices']==[3603]
    assert audit[0]['component_ray_indices']==[[3601,3602],[3604,3605]]
    assert all(not c['physical_opening'] for c in children)
    ranges[:]=5
    children,audit=decompose(ranges,valid,np.eye(4),'f',[parent])
    assert len(children)==1 and children[0]['component_kind'].startswith('UNRESOLVED')
    assert audit[0]['residual_ray_indices']==ids

def test_whole_3d_ray_direction_preserved():
    ranges=np.ones((16,720))*20;valid=np.ones_like(ranges,bool)
    p=dict(source_refs=['f/ray:719'],axis_start_xyz_m=[0,0,0],axis_target_xyz_m=[4,0,0])
    children,_=decompose(ranges,valid,np.eye(4),'f',[p])
    assert children[0]['axis_target_xyz_m'][2]!=0
