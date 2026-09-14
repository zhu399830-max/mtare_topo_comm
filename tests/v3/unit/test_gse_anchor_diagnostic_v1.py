import numpy as np
import pytest

from mtare_topo.teacher.gse_anchor_diagnostic_v1 import local_path


def test_clip_only_first_exit_keeps_reference_order():
    p=np.array([[.1,.1,.1],[20,.1,.1],[-20,.1,.1]])
    result=local_path(p,np.zeros(3),0.)
    assert result.shape==(2,3)
    np.testing.assert_allclose(np.linalg.norm(result[-1]),10.)
    np.testing.assert_array_equal(result[0],p[0])


def test_inside_finite_source_end_not_extended():
    p=np.array([[.1,.1,.1],[2,.1,.1]])
    np.testing.assert_array_equal(local_path(p,np.zeros(3),0.),p)


def test_off_layer_anchor_not_projected_into_roi():
    with pytest.raises(ValueError):local_path(np.array([[0.,0,11],[1,0,11]]),np.zeros(3),0.)


def test_ambiguous_boundary_vertex_not_shifted():
    assert local_path(np.array([[0.,0,0],[10,0,0],[20,0,0]]),np.zeros(3),0.) is None
