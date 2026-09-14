import numpy as np
import pytest
from mtare_topo.evaluation.gse_cap_branch_precision import cap_branch_precision, summarize_cap_directions


def branch(direction, boundary=False):
    tri=np.array([[[0.,0,1],[1.,0,1],[0.,1,1]]])
    origin=np.array([[.2,.2,1. if boundary else 2.]])
    return cap_branch_precision(tri,origin,np.array([[0.,0,-1.]]),np.array([3]),direction)


def test_three_stable_branches_only_numerical_support():
    reports=[branch(d) for d in ((1,0,0),(0,1,0),(-1,0,0))]
    result=summarize_cap_directions(reports)
    assert result['three_cap_directions_numerically_supported']
    assert not result['label_or_physical_connectivity_certified']


def test_one_stable_two_boundary_cannot_certify_old_anchor():
    reports=[branch((1,0,0)),branch((0,1,0),True),branch((-1,0,0),True)]
    result=summarize_cap_directions(reports)
    assert result['stable_distinct_cap_directions']==1
    assert not result['three_cap_directions_numerically_supported']
    assert reports[1]['unqualified_ray_indices']==[3]


def test_permutation_and_duplicate_direction_do_not_invent_branch():
    reports=[branch((1,0,0)),branch((0,1,0)),branch((1,0,0))]
    assert summarize_cap_directions(reports)==summarize_cap_directions(reports[::-1])
    assert summarize_cap_directions(reports)['stable_distinct_cap_directions']==2
    assert summarize_cap_directions([])['stable_distinct_cap_directions']==0


def test_duplicate_ray_with_unstable_occurrence_stays_unqualified():
    tri=np.repeat(np.array([[[0.,0,1],[1.,0,1],[0.,1,1]]]),2,axis=0)
    result=cap_branch_precision(tri,np.array([[.2,.2,2.],[.2,.2,1.]]),
        np.array([[0.,0,-1.],[0.,0,-1.]]),np.array([7,7]),(1,0,0))
    assert result['stable_entering_ray_indices']==[]
    assert result['unqualified_ray_indices']==[7]


def test_source_precision_required_and_no_fabricated_ray_identity():
    tri=np.array([[[0.,0,1],[1.,0,1],[0.,1,1]]])
    args=(tri,np.array([[.2,.2,2.]]),np.array([[0.,0,-1.]]))
    for ray,direction in (([-1],(1,0,0)),([1.5],(1,0,0)),([1],(0,0,0))):
        with pytest.raises(ValueError):cap_branch_precision(*args,np.array(ray),direction)
    with pytest.raises(ValueError):
        cap_branch_precision(tri.astype(np.float32),*args[1:],np.array([1]),(1,0,0))


def test_leaving_requires_forward_outward_not_entering_or_boundary():
    tri=np.repeat(np.array([[[0.,0,1],[1.,0,1],[0.,1,1]]]),3,axis=0)
    origins=np.array([[.2,.2,0.],[.2,.2,2.],[.2,.2,1.]])
    rays=np.array([[0.,0,1.],[0.,0,-1.],[0.,0,1.]])
    result=cap_branch_precision(tri,origins,rays,np.array([1,2,3]),(0,0,-1),direction_kind='leaving')
    assert result['stable_leaving_ray_indices']==[1]
    assert result['unqualified_ray_indices']==[2,3]
    assert 'stable_entering_ray_indices' not in result
    with pytest.raises(ValueError):
        cap_branch_precision(tri,origins,rays,np.array([1,2,3]),(0,0,-1),direction_kind='either')
