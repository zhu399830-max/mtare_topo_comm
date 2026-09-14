import numpy as np
import pytest

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.semantics.observed_primitive_candidates import observed_primitive_candidates
from mtare_topo.semantics.primitive_relation_nonlearning import PrimitiveRelationBaselineConfig
from mtare_topo.semantics.range_exit_baseline import RangeExitBaselineConfig


def run(p, ids, translation=None, yaw=None, **kwargs):
    return observed_primitive_candidates(p, ids, np.zeros((5, 3)) if translation is None else translation,
        np.zeros(5) if yaw is None else yaw, fit_config=PrimitiveRelationBaselineConfig(),
        exit_config=RangeExitBaselineConfig(smoothing_columns=1, **kwargs))


def scan(columns, frame=4, distance=8.):
    ring, col = np.meshgrid(np.arange(16), np.asarray(columns), indexing='ij')
    ids = frame*11520 + ring.ravel()*720 + col.ravel()
    p = np.asarray(lidar_local_directions())[ring.ravel(), col.ravel()]*distance
    return p, ids


def test_empty_and_short_scan_do_not_force_candidate():
    assert run(np.empty((0, 3)), np.array([], int)) == ()
    p, ids = scan(np.arange(720), distance=2)
    assert run(p, ids) == ()


def test_proposals_keep_fit_rejection_and_real_frame_support():
    p, ids = scan(np.arange(20))
    result = run(p, ids)
    assert len(result) == 1
    assert result[0].contributing_frames == (4,)
    # An arc on one range shell is not a sufficiently long tunnel axis.
    assert result[0].fit.reason == 'insufficient_observed_extent'
    assert result[0].fit.axis_controls_m is None


def test_original_sensor_origin_used_to_recover_range():
    p, ids = scan(np.arange(20), frame=0)
    translation = np.zeros((5, 3)); translation[0] = [-5, 0, 0]
    result = run(p + translation[0], ids, translation)
    assert len(result) == 1  # current-origin distances would be <7 and hide sector
    assert result[0].contributing_frames == (0,)


def test_capacity_fails_instead_of_selecting_top_sectors():
    columns = np.concatenate([np.arange(i, i+10) for i in range(0, 720, 80)])
    p, ids = scan(columns)
    with pytest.raises(ValueError, match='per-frame sector capacity'):
        run(p, ids)


def test_permutation_and_invalid_motion():
    p, ids = scan(np.arange(20))
    assert run(p, ids) == run(p[::-1], ids[::-1])
    yaw = np.zeros(5); yaw[-1] = 1
    with pytest.raises(ValueError, match='current transform'): run(p, ids, yaw=yaw)
    with pytest.raises(ValueError, match='ray indices'): run(p, np.zeros_like(ids))


def test_full_proposal_to_supported_fit_uses_observed_returns():
    p, ids = scan(np.arange(15, 85), distance=1.)
    # Synthetic observed wall y=2, with varying depth along its length.
    p = p * (2 / p[:, 1])[:, None]
    result = run(p, ids)
    assert len(result) == 1
    fit = result[0].fit
    assert fit.reason == 'observed_surface_fit_only'
    assert fit.axis_controls_m is not None
    assert set(fit.selected_ray_indices).issubset(set(ids))
    assert not fit.connectivity_verified
