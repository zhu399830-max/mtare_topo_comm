from dataclasses import replace

import numpy as np
import pytest

from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_observed_axis_support_v1 import observed_axis_support


def grid():
    return build_surface_ray_grid(np.array([[.1,.1,.1]]),np.array([[2.1,.1,.1]]),np.array([True]),np.array([0]))


def test_continuous_observed_cells_are_only_axis_proxy():
    result=observed_axis_support(grid(),np.array([[.11,.1,.1],[1.9,.1,.1]]))
    assert result.status=="OBSERVED_AXIS" and result.free_cells==8
    assert result.physical_reachability is None and result.semantic_anchor is None


def test_visible_extremes_do_not_fill_unknown_gap():
    g=build_surface_ray_grid(np.array([[.1,.1,.1],[1.1,.1,.1]]),
        np.array([[.4,.1,.1],[1.4,.1,.1]]),np.array([True,True]),np.array([0,1]))
    result=observed_axis_support(g,np.array([[.11,.1,.1],[1.35,.1,.1]]))
    assert result.status=="UNKNOWN" and result.unknown_cells>0


def test_occupied_is_not_unreachable_opening_label():
    result=observed_axis_support(grid(),np.array([[.11,.1,.1],[2.12,.1,.1]]))
    assert result.status=="UNKNOWN" and result.occupied_cells==1
    assert result.physical_reachability is None


def test_outside_portion_is_not_clipped_into_success():
    result=observed_axis_support(grid(),np.array([[.1,.1,.1],[11.,.1,.1]]))
    assert result.status=="UNKNOWN" and result.outside_observation_cube


def test_reverse_retains_touched_cells():
    p=np.array([[.11,.1,.1],[1.9,.1,.1]])
    assert observed_axis_support(grid(),p).cell_indices==observed_axis_support(grid(),p[::-1]).cell_indices


def test_grid_tampering_rejected():
    g=grid();state=g.state.copy();state[:]=1
    with pytest.raises(ValueError):observed_axis_support(replace(g,state=state),np.array([[.1,.1,.1],[1.,.1,.1]]))


def test_zero_segment_and_integer_path_rejected():
    for p in (np.array([[0,0,0],[1,1,1]]),np.array([[.1,.1,.1],[.1,.1,.1]])):
        with pytest.raises(ValueError):observed_axis_support(grid(),p)
