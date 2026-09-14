import numpy as np
import pytest
from mtare_topo.data.gse_surface_source_join import join_surface_sources


def records():
    return np.array([[9, 3, .01, .02, 1, 2, 2],
                     [2, 8, 0, 0, 2, 1, 3],
                     [9, 1, .02, .03, 1, 4, 4]], dtype=float)


def test_identity_join_is_not_nearest_source_selection():
    result = join_surface_sources(records(), np.array([2, 9, 11]), np.array([9, 11, 2]))
    assert result.candidates == (('1', '3'), (), ('8',))
    assert result.ambiguous.tolist() == [True, False, False]
    assert result.missing.tolist() == [False, True, False]
    assert not result.point_label_qualified
    assert not hasattr(result, 'evidence_known')
    shuffled = join_surface_sources(records()[::-1], np.array([11, 9, 2]), np.array([9, 11, 2]))
    assert shuffled.candidates == result.candidates
    assert shuffled.scene_residuals_m == result.scene_residuals_m


@pytest.mark.parametrize('change', ['duplicate', 'outside', 'fractional', 'nonfinite', 'negative'])
def test_reject_bad_provenance(change):
    rows = records()
    if change == 'duplicate': rows = np.concatenate([rows, rows[:1]])
    if change == 'outside': rows[0, 0] = 99
    if change == 'fractional': rows[0, 1] = 1.5
    if change == 'nonfinite': rows[0, 2] = np.nan
    if change == 'negative': rows[0, 2] = -1
    with pytest.raises(ValueError):
        join_surface_sources(rows, np.array([2, 9]), np.array([9, 2]))


def test_inventory_drift_fails():
    with pytest.raises(ValueError, match='inventory mismatch'):
        join_surface_sources(records(), np.array([2, 9]), np.array([2, 8]))
