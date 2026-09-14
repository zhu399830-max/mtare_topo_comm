import numpy as np
import pytest
from mtare_topo.evaluation.gse_convex_ray_union import prism_planes,ray_intervals,origin_component_exit,declared_union_exit
from mtare_topo.data.gse_synthetic_matrix import matrix


def test_union_overlap_gap_touch_and_permutation():
    intervals=np.array([[[-2,2],[1,4],[3,6],[7,9]],
                        [[-2,2],[2,4],[3,6],[7,9]],
                        [[1,2],[1,4],[3,6],[7,9]]],float)
    end,valid=origin_component_exit(intervals)
    np.testing.assert_array_equal(valid,[True,True,False])
    np.testing.assert_allclose(end,[6,2,np.nan],equal_nan=True)
    np.testing.assert_allclose(origin_component_exit(intervals[:,::-1])[0],end,equal_nan=True)


def test_prism_entry_from_outside_and_parallel_miss():
    planes=prism_planes([[0,0,0],[10,0,0]],[2,2],2)
    lo,hi=ray_intervals([[-1,0,0],[5,3,0],[5,0,0]],[[1,0,0],[1,0,0],[0,2,0]],planes)
    np.testing.assert_allclose(lo[[0,2]],[1,-1]);np.testing.assert_allclose(hi[[0,2]],[11,1])
    assert lo[1]==np.inf and hi[1]==-np.inf


CASES=[c for c in matrix() if c['program']['type']=='double_junction']
@pytest.mark.parametrize('case',CASES,ids=[c['case_id'] for c in CASES])
def test_double_junction_axis_rays_keep_two_branch_positions(case):
    origins=np.array([[-4,0,0],[4,0,0],[0,0,0]],float)
    directions=np.array([[0,-1,0],[0,-1,0],[0,-1,0]],float)
    end,valid,_=declared_union_exit(case,origins,directions)
    assert valid.all()
    # Independent 2D segment-axis intersection, not ideal-superellipse radius:
    # tiny sin/cos residuals raised to 1/4 matter for the rounded rectangle.
    angles=np.arange(64)*2*np.pi/64;power=case['shape_exponent']
    u=case['half_axes_m'][0]*np.sign(np.cos(angles))*np.abs(np.cos(angles))**(2/power)
    z=case['half_axes_m'][1]*np.sign(np.sin(angles))*np.abs(np.sin(angles))**(2/power)
    intersections=[]
    for i in range(64):
        j=(i+1)%64
        if z[i]==0:intersections.append(u[i])
        elif z[i]*z[j]<0:intersections.append(u[i]-z[i]*(u[j]-u[i])/(z[j]-z[i]))
    expected_wall=-min(intersections)
    np.testing.assert_allclose(end,[30,30,expected_wall],rtol=0,atol=1e-12)
    # Shift all geometry and ray origins together: exit parameters unchanged.
    from copy import deepcopy
    shifted=deepcopy(case);delta=np.array([7,-2,3])
    for e in shifted['program']['edges']:e['points']=(np.array(e['points'])+delta).tolist()
    np.testing.assert_allclose(declared_union_exit(shifted,origins+delta,directions)[0],end)


def test_curved_prism_rejected_not_straightened():
    with pytest.raises(ValueError):prism_planes([[0,0,0],[1,0,0],[1,1,0]],[2,2],2)
