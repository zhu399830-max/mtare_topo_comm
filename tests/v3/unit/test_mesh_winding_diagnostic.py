import numpy as np
import pytest
from mtare_topo.evaluation.mesh_winding_diagnostic import oriented_winding


def tetra():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    return vertices[np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])]


def test_inside_outside_orientation_and_multiplicity():
    t=tetra();q=np.array([[.1,.1,.1],[2.,2.,2.]])
    np.testing.assert_allclose(oriented_winding(t,q),[1.,0.],atol=1e-12)
    np.testing.assert_allclose(oriented_winding(t[:,::-1],q),[-1.,0.],atol=1e-12)
    np.testing.assert_allclose(oriented_winding(np.concatenate([t,t]),q),[2.,0.],atol=1e-12)
    np.testing.assert_allclose(oriented_winding(t+17,q+17),[1.,0.],atol=1e-12)


def test_surface_is_not_an_inside_answer():
    with pytest.raises(ValueError):oriented_winding(tetra(),[[0.,0.,0.]])
