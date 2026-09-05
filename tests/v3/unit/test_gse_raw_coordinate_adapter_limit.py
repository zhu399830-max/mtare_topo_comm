"""Oracle tensor weights expose readout limits; nothing here is learned."""
import numpy as np
import pytest
import torch

from mtare_topo.representation.primitive_relation_model import _token_xyz


def _cloud():
    points = torch.zeros(1, 5, 16, 720, 3, dtype=torch.float64)
    valid = torch.zeros(1, 5, 16, 720, dtype=torch.bool)
    # Token 0 has two different returns; token 1 has three.
    points[0, 0, 0, 0] = torch.tensor((2., 0., -2.))
    points[0, 0, 1, 1] = torch.tensor((2., 0., 2.))
    valid[0, 0, 0, 0] = valid[0, 0, 1, 1] = True
    for row, xyz in enumerate(((8., 1., -1.), (8., 1., 0.), (8., 1., 1.))):
        points[0, 0, row, 4] = torch.tensor(xyz)
        valid[0, 0, row, 4] = True
    return points, valid


def _raw_weighted(points, valid, token_weights, *, divide_group_counts):
    grouped_points = points.reshape(1, 5, 16, 180, 4, 3)
    grouped_valid = valid.reshape(1, 5, 16, 180, 4)
    counts = grouped_valid.sum(dim=(2, 4)).clamp_min(1)
    weights = token_weights[:, :, None, :, None] * grouped_valid
    if divide_group_counts:
        weights = weights / counts[:, :, None, :, None]
    weights = weights / weights.sum(dim=(1, 2, 3, 4), keepdim=True)
    return (weights[..., None] * grouped_points).sum(dim=(1, 2, 3, 4))


def test_uniform_per_valid_point_partition_exactly_reproduces_old_token_mean_readout():
    points, valid = _cloud()
    pooled, pooled_valid = _token_xyz(points, valid)
    assert pooled_valid.sum() == 2
    token_weights = torch.zeros(1, 5, 180, dtype=torch.float64)
    token_weights[0, 0, :2] = torch.tensor((.25, .75), dtype=torch.float64)
    old = (token_weights[..., None] * pooled).sum(dim=(1, 2))
    raw = _raw_weighted(points, valid, token_weights, divide_group_counts=True)
    torch.testing.assert_close(raw, old, atol=1e-14, rtol=0.)
    torch.testing.assert_close(old, torch.tensor([[6.5, .75, 0.]], dtype=torch.float64), atol=1e-14, rtol=0.)


def test_broadcast_without_count_division_changes_group_weights_by_return_count():
    points, valid = _cloud()
    pooled, _ = _token_xyz(points, valid)
    weights = torch.zeros(1, 5, 180, dtype=torch.float64)
    weights[0, 0, :2] = .5
    old = (weights[..., None] * pooled).sum(dim=(1, 2))
    raw = _raw_weighted(points, valid, weights, divide_group_counts=False)
    # Equal token weight becomes 2/5 vs 3/5 because valid return counts differ.
    expected = .4 * pooled[0, 0, 0] + .6 * pooled[0, 0, 1]
    torch.testing.assert_close(raw[0], expected)
    assert not torch.allclose(raw, old)
    assert old[0, 0] == 5
    assert raw[0, 0] == pytest.approx(5.6)


def test_distinct_oracle_point_weights_can_select_layers_that_token_mean_cannot():
    points, valid = _cloud()
    pooled, _ = _token_xyz(points, valid)
    assert pooled[0, 0, 0, 2] == 0
    pair = torch.stack((points[0, 0, 0, 0], points[0, 0, 1, 1]))
    # Explicit oracle weights demonstrate representability, not trained segmentation.
    upper_oracle = torch.tensor((0., 1.), dtype=torch.float64) @ pair
    lower_oracle = torch.tensor((1., 0.), dtype=torch.float64) @ pair
    assert upper_oracle[2] == 2 and lower_oracle[2] == -2
    assert not torch.equal(upper_oracle, pooled[0, 0, 0])
    assert not torch.equal(lower_oracle, pooled[0, 0, 0])


def test_raw_convex_combinations_cannot_reach_axis_beyond_one_sided_wall():
    points = torch.tensor([[x, 2., z] for x in (-2., 0., 2.) for z in (-1., 1.)], dtype=torch.float64)
    query_axis = torch.tensor((0., 0., 0.), dtype=torch.float64)
    rng = np.random.default_rng(42)
    weights = torch.from_numpy(rng.uniform(size=(100, len(points))))
    weights /= weights.sum(dim=1, keepdim=True)
    reconstructed = weights @ points
    torch.testing.assert_close(reconstructed[:, 1], torch.full((100,), 2., dtype=torch.float64))
    assert torch.all(torch.linalg.vector_norm(reconstructed - query_axis, dim=1) >= 2. - 1e-12)
    # This separating plane proves the result for ALL convex weights, not just the random examples.
    unit_normal = torch.tensor((0., -1., 0.), dtype=torch.float64)
    distance_lower_bound = query_axis @ unit_normal - (points @ unit_normal).max()
    assert distance_lower_bound == 2
