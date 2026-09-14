import numpy as np
from mtare_topo.teacher.gse_superellipse_constraints_v1 import local_constraints


def contour(theta,axes=(4.,3.),p=6.):
    trig=np.column_stack((np.cos(theta),np.sin(theta)))
    return np.array(axes)*np.sign(trig)*np.abs(trig)**(2/p)


def test_full_contour_constrains_center_and_circle_angle_does_not_invalidate_center():
    theta=np.arange(80)*2*np.pi/80+.01
    for axes,p in [((4.,3.),6.),((4.,4.),2.)]:
        r=local_constraints(contour(theta,axes,p),center_uv_m=[0,0],half_axes_m=axes,exponent=p)
        assert np.max(np.abs(r['residual']))<1e-12
        assert r['center_profile_singular_values'].min()>1
        assert not r['qualified_label']


def test_small_wall_arc_is_much_less_informative_without_arbitrary_pass_threshold():
    full=local_constraints(contour(np.linspace(0,2*np.pi,80,endpoint=False)+.01),
        center_uv_m=[0,0],half_axes_m=[4,3],exponent=6)
    arc=local_constraints(contour(np.linspace(-.03,.03,80)),
        center_uv_m=[0,0],half_axes_m=[4,3],exponent=6)
    assert arc['center_profile_singular_values'].min()<full['center_profile_singular_values'].min()*1e-4


def test_analytic_jacobian_matches_parameter_perturbations():
    points=np.array([[1.1,.4],[-.7,2.1],[2.4,-1.7]])
    params=np.array([.02,-.03,np.log(4),np.log(3),np.log(5),.17])
    def evaluate(t):
        return local_constraints(points,center_uv_m=10*t[:2],half_axes_m=np.exp(t[2:4]),
                                 exponent=np.exp(t[4]),angle_rad=t[5])
    analytic=evaluate(params)['jacobian']
    for k in range(6):
        delta=np.zeros(6);delta[k]=1e-6
        numeric=(evaluate(params+delta)['residual']-evaluate(params-delta)['residual'])/2e-6
        np.testing.assert_allclose(analytic[:,k],numeric,atol=1e-8,rtol=1e-7)


def test_duplicate_complete_population_does_not_artificially_multiply_information():
    points=contour(np.linspace(0,2*np.pi,40,endpoint=False)+.01)
    def get(x):return local_constraints(x,center_uv_m=[0,0],half_axes_m=[4,3],exponent=6)['center_profile_singular_values']
    np.testing.assert_allclose(get(points),get(np.repeat(points,3,axis=0)),atol=1e-12)
