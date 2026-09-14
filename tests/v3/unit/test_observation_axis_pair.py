import numpy as np
from mtare_topo.representation.gse_observation_axis_pair import estimate_axes, pair_predictions


def surfaces():
    # Two surface orientations share the x axis, but points spread more in yz.
    x, q = np.meshgrid(np.linspace(-.2,.2,3),np.linspace(-1,1,5))
    a = np.c_[x.ravel(),q.ravel(),np.ones(x.size)]
    b = np.c_[x.ravel(),np.ones(x.size),q.ravel()]
    return np.r_[a,b],np.repeat([0,1],len(a)),np.array([[1],[0]])


def test_normal_composition_resolves_axis_not_longest_extent():
    x,m,n = surfaces(); out = estimate_axes(x,m,n,2)
    assert out['NORMAL'].valid.all()
    assert np.allclose(abs(out['NORMAL'].axes[:,0]),1)
    assert np.allclose(out['POINT'].axes[:,0],0)
    assert np.array_equal(out['NORMAL'].support_points,out['POINT'].support_points)


def test_single_surface_normal_does_not_invent_tangent():
    x,m,n = surfaces(); x=x[m==0]
    out = estimate_axes(x,np.zeros(len(x),int),np.array([[-1]]),1)
    assert not out['NORMAL'].valid.any()
    assert out['POINT'].valid.all()
    pred,valid = pair_predictions(out['NORMAL'],np.array([[0]]))
    assert not valid.any() and np.isnan(pred).all()


def test_point_order_and_global_rotation():
    x,m,n=surfaces(); rot=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    a=estimate_axes(x,m,n,2); b=estimate_axes(x[::-1]@rot,m[::-1],n,2)
    for name in a:
        assert np.array_equal(a[name].valid,b[name].valid)
        assert np.allclose(abs(np.sum((a[name].axes@rot)*b[name].axes,axis=1)),1)
        assert np.allclose(pair_predictions(a[name],n)[0],pair_predictions(b[name],n)[0])


def test_duplicate_neighbors_do_not_reweight_support():
    x,m,n=surfaces()
    a=estimate_axes(x,m,n,2); b=estimate_axes(x,m,np.repeat(n,8,axis=1),2)
    for name in a:
        assert np.array_equal(a[name].support_points,b[name].support_points)
        assert np.allclose(a[name].eigenvalues,b[name].eigenvalues)
