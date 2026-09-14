import numpy as np
import pytest
from mtare_topo.representation.observed_query_sampling_v1 import sample_observed_queries


def test_empty_small_and_duplicate_inputs_do_not_pad():
    assert sample_observed_queries(np.empty((0,3))).shape==(0,)
    p=np.array([[1.,0,0],[1.,0,0],[2.,0,0]])
    i=sample_observed_queries(p)
    assert len(i)==2 and np.array_equal(p[i],[[1,0,0],[2,0,0]])


def test_lexical_start_and_ties_independent_of_input_order():
    p=np.array([[0.,1,0],[0,-1,0],[-1,0,0],[1,0,0]])
    expected=p[sample_observed_queries(p)]
    assert np.array_equal(expected,[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]])
    perm=np.array([3,1,0,2])
    assert np.array_equal(p[perm][sample_observed_queries(p[perm])],expected)


def test_source_indices_are_real_and_height_is_retained():
    p=np.array([[0.,0,-3],[0,0,3],[1,0,0]])
    indices=sample_observed_queries(p,maximum=2)
    assert np.array_equal(p[indices],[[0,0,-3],[0,0,3]])
    assert len(set(indices.tolist()))==2


def test_capacity_and_determinism():
    p=np.random.default_rng(0).uniform(-2,2,(300,3))
    a=sample_observed_queries(p);b=sample_observed_queries(p)
    assert len(a)==32 and np.array_equal(a,b) and len(set(a.tolist()))==32


@pytest.mark.parametrize('points',[np.array([[11.,0,0]]),np.array([[np.nan,0,0]]),np.array([[1,2,3]])])
def test_invalid_input_not_silently_clipped(points):
    with pytest.raises(ValueError):sample_observed_queries(points)
