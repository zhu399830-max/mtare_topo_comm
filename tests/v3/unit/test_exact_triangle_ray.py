from fractions import Fraction
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray


def test_exact_interior_edge_and_outside():
    tri=[[2,0,0],[2,1,0],[2,0,1]]
    hit=exact_triangle_ray(tri,[0,.25,.25],[1,0,0])
    assert hit['t']==2 and hit['u']==Fraction(1,4) and hit['v']==Fraction(1,4)
    assert hit['status']=='hit' and not hit['boundary']
    assert exact_triangle_ray(tri,[0,.5,.5],[1,0,0])['boundary']
    assert exact_triangle_ray(tri,[0,1,1],[1,0,0])['status']=='outside_triangle'


def test_winding_permutation_does_not_change_plane_or_hit():
    tri=[[2,0,0],[2,1,0],[2,0,1]]
    a=exact_triangle_ray(tri,[0,.25,.25],[1,0,0])
    b=exact_triangle_ray(tri[::-1],[0,.25,.25],[1,0,0])
    assert a['plane']==b['plane'] and a['t']==b['t']


def test_parallel_and_coplanar_and_degenerate():
    tri=[[2,0,0],[2,1,0],[2,0,1]]
    assert exact_triangle_ray(tri,[0,0,0],[0,1,0])['status']=='parallel'
    assert exact_triangle_ray(tri,[2,0,0],[0,1,0])['status']=='coplanar'
    assert exact_triangle_ray([[0,0,0]]*3,[0,0,0],[1,0,0])['status']=='degenerate'
