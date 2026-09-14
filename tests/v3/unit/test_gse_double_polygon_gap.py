"""Pinned diagnostic ray: polygon gap versus ideal-ellipse overlap.

Does not execute or certify the production renderer.
"""
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def test_saved_worst_ray_has_gap_hidden_by_ideal_ellipse():
    case=next(c for c in matrix() if c['case_id']=='double_junction__ellipse__view3')
    origin=np.array([[1.9,0.,.04]])
    direction=np.array([[.05232798574151599,-.9984774386302507,.017452405410415962]])
    end,supported,intervals=declared_union_exit(case,origin,direction)
    first=intervals[0,1,1];next_entry=intervals[0,4,0]
    assert supported[0] and end[0]==first and next_entry>first
    np.testing.assert_allclose([first,next_entry],[1.998137966459527,2.004893607443446],rtol=0,atol=1e-12)
    point=(origin+direction*((first+next_entry)/2))[0]
    assert (point[1]/2)**2+(point[2]/1.5)**2>1
    assert ((point[0]-4)/2)**2+(point[2]/1.5)**2<1
    assert not ((intervals[0,:,0]<(first+next_entry)/2)&(intervals[0,:,1]>(first+next_entry)/2)).any()
