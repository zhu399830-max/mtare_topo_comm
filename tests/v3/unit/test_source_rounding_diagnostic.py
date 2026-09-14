import numpy as np
import pytest
from mtare_topo.evaluation.source_rounding_diagnostic import float32_cells, source_candidates


def run(spans, uncertainty=None):
    spans=np.asarray([spans],float)
    if uncertainty is None: uncertainty=np.zeros(spans.shape[:2],bool)
    return source_candidates(np.array([2.],np.float32),spans,uncertain_operands=uncertainty)


def test_binade_boundary_cell_is_asymmetric():
    low,high=float32_cells(np.array([2.],np.float32))[0]
    assert 2.-low == 2.**-24
    assert high-2. == 2.**-23


def test_single_candidate_is_not_automatically_certificate():
    result=run([[-1,2],[3,4]])
    assert result['singleton_candidate'][0]
    assert not result['surface_uniqueness_certified'][0]


def test_two_different_exits_in_same_storage_cell_remain_ambiguous():
    result=run([[-1,2],[-1,2.+2.**-25]])
    assert result['possible_exit'].tolist()==[[True,True]]
    assert not result['singleton_candidate'][0]


def test_entering_surface_not_misreported_as_exit():
    result=run([[-1,2],[2,3]])
    assert result['possible_entry'].tolist()==[[False,True]]
    assert result['possible_exit'].tolist()==[[True,False]]
    assert not result['singleton_candidate'][0]


def test_tangent_and_explicit_uncertainty_retained():
    assert run([[-1,2],[2,2]])['possible_surface'].all()
    result=run([[-1,2],[np.inf,-np.inf]],np.array([[False,True]]))
    assert result['possible_surface'].all() and not result['singleton_candidate'][0]


def test_closed_rounding_ties_are_conservative():
    low,high=float32_cells(np.array([2.],np.float32))[0]
    assert run([[-1,low],[-1,high]])['possible_surface'].all()


def test_missing_uncertainty_and_invalid_input_rejected():
    with pytest.raises(TypeError): source_candidates(np.array([2.],np.float32),[[[-1,2]]])
    with pytest.raises(ValueError): float32_cells(np.array([2.],np.float64))
    with pytest.raises(ValueError): run([[3,2]])
