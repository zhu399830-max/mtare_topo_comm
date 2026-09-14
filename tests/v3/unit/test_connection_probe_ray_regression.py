"""Counterexamples to the ideal-plane probe using existing ray evidence.

No plane IDs, connection pairs or bridge labels enter grid construction.
Queries are software-test assertions only. These tests certify neither ground
support nor robot traversability. No changes to existing grid tolerances.
"""
import numpy as np
import pytest
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.representation.gse_observed_regions import build_observed_regions, query_observed_pair


def evaluate(origins, endpoints, a, b):
    origins = np.asarray(origins, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64)
    grid = build_surface_ray_grid(origins, endpoints, np.ones(len(origins), dtype=bool),
                                  np.zeros(len(origins), dtype=np.int64))
    regions = build_observed_regions(grid)
    return query_observed_pair(regions, grid, a, b)


def test_unobserved_middle_is_not_bridged():
    result = evaluate([[.025,.125,.125],[1.775,.125,.125]],
                      [[.625,.125,.125],[2.425,.125,.125]],
                      [.125,.125,.125],[2.125,.125,.125])
    assert result == {'status':'UNKNOWN','reason':'NO_OBSERVED_CELL_CONNECTION'}


@pytest.mark.parametrize('shift', [0., .001, .000001])
def test_interior_noise_does_not_require_identical_points(shift):
    result = evaluate([[.025,.125+shift,.125]], [[2.425,.125+shift,.125]],
                      [.125,.125,.125], [2.125,.125,.125])
    assert result['status'] == 'SAME_OBSERVED_CELL_COMPONENT'
    assert result['physical_connectivity'] is False


def test_bent_3d_observed_path_not_required_coplanar():
    # Rays overlap before their own first returns; no surface correspondence
    # or shared sampled coordinate is required. This is AIR, not a floor ramp.
    result = evaluate([[.025,.125,.125],[1.125,.125,.125],[1.125,.125,.625]],
                      [[1.625,.125,.125],[1.125,.125,1.125],[2.425,.125,.625]],
                      [.125,.125,.125],[2.125,.125,.625])
    assert result['status'] == 'SAME_OBSERVED_CELL_COMPONENT'
    assert result['physical_connectivity'] is False


def test_stacked_observed_rays_stay_separate():
    result = evaluate([[.025,.125,.125],[.025,.125,2.125]],
                      [[2.425,.125,.125],[2.425,.125,2.125]],
                      [.125,.125,.125],[.125,.125,2.125])
    assert result['status'] == 'UNKNOWN'


def test_first_return_blocks_even_if_other_ray_claims_free():
    result = evaluate([[.025,.125,.125],[.025,.125,.125]],
                      [[2.425,.125,.125],[1.125,.125,.125]],
                      [.125,.125,.125],[2.125,.125,.125])
    assert result['status'] == 'UNKNOWN'


def test_boundary_ray_not_silently_jittered_to_free():
    result = evaluate([[.025,0.,.125]], [[2.425,0.,.125]],
                      [.125,.125,.125],[2.125,.125,.125])
    assert result['status'] == 'UNKNOWN'
