import numpy as np
from mtare_topo.topology.grouping_sequential_map_v1 import SequentialMap


def feed(g,t,x=0.,world=(0.,4.,8.),continuous=True,frames=None):
    pose=np.eye(4);pose[0,3]=x
    centers=np.asarray([[w-x,0.,0.] for w in world]).reshape(-1,3)
    return g.update(timestamp=t,frame_ids=list(range(t*5,t*5+5)) if frames is None else frames,
        sensor_to_world=pose,centers_sensor=centers,continuous=continuous)


def test_visibility_alone_no_edges_and_overlap_no_confirmation():
    g=SequentialMap();feed(g,0,x=-3)
    a=feed(g,1,x=-3,frames=[1,2,3,4,5])
    assert not any(n['confirmed'] for n in a['nodes'])
    a=feed(g,2,x=-3)
    assert all(n['confirmed'] for n in a['nodes']) and not a['edges']


def test_actual_visit_middle_node_prefix_and_break():
    g=SequentialMap();feed(g,0);saved=feed(g,1)
    feed(g,2,x=2);feed(g,3,x=4);a=feed(g,4,x=8)
    assert [(e['source'],e['target']) for e in a['edges']]==[(0,1),(1,2)]
    assert len(a['edges'][0]['trace'])==3
    h=SequentialMap();feed(h,0)
    assert feed(h,1)==saved
    before=len(a['edges']);a=feed(g,5,x=0,continuous=False)
    assert len(a['edges'])==before
