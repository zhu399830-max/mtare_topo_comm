from copy import deepcopy
import pytest
from mtare_topo.planning.recorded_return_route import recorded_return_route

def sample(t,x,y):return dict(order=t,xyz_m=[x,y,0])
def inputs():
    return dict(nodes=[dict(id=0,xyz_m=[0,0,0]),dict(id=1,xyz_m=[2,2,0]),dict(id=2,xyz_m=[4,2,0])],
        edges=[dict(source=0,target=1,length_m=4,trace=[sample(0,0,0),sample(1,2,0),sample(2,2,2)]),
               dict(source=1,target=2,length_m=2,trace=[sample(2,2,2),sample(3,4,2)])],
        pending=[sample(3,4,2),sample(4,4,4)],current_node=2,target_node=0,target_order=1)

def test_return_preserves_bend_and_stops_at_recorded_observation():
    data=inputs();before=deepcopy(data);r=recorded_return_route(**data)
    assert [s['xyz_m'] for s in r['samples']]==[[4,4,0],[4,2,0],[2,2,0],[2,0,0]]
    assert r['length_m']==6 and not r['new_edge_inferred']
    assert data==before

def test_unrecorded_gap_is_rejected():
    data=inputs();data['edges'][1]['trace'][0]['xyz_m']=[3,3,0]
    with pytest.raises(ValueError,match='join'):recorded_return_route(**data)

def test_disconnected_graph_not_replaced_with_straight_goal():
    data=inputs();data['edges']=data['edges'][1:]
    with pytest.raises(ValueError,match='no recorded route'):recorded_return_route(**data)

def test_missing_observation_not_guessed_from_nearest_node():
    data=inputs();data['target_order']=1.5
    with pytest.raises(ValueError,match='attachment'):recorded_return_route(**data)
