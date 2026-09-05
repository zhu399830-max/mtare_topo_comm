import numpy as np

from mtare_topo.evaluation.primitive_attachment_observability import (
    endpoint_support_gaps,
    summarize_attachment_observations,
    unique_attachment_indices,
    unique_attachment_observations,
    world_to_sensor,
)


def test_world_to_sensor_batched_yaw() -> None:
    points = np.asarray([[[[2.0, 1.0, 0.0]]], [[[1.0, 2.0, 0.0]]]])
    origins = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    transformed = world_to_sensor(points, origins, np.asarray([0.0, 90.0]))
    np.testing.assert_allclose(transformed[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(transformed[1, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)


def test_endpoint_gaps_and_unique_undirected_pairs() -> None:
    endpoints = np.asarray([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    ])
    axis = np.zeros((1, 32, 3, 3), dtype=np.float64)
    axis[0, 0] = [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    axis[0, 1] = [[2.1, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    index = np.full((1, 32), -1, dtype=np.int64); index[0, :2] = [0, 1]
    mask = index >= 0
    gaps, observed = endpoint_support_gaps(
        axis_control_current_sensor_m=axis,
        primitive_index=index,
        primitive_mask=mask,
        primitive_endpoints_world_m=endpoints,
        sensor_xyz_m=np.zeros((1, 3)),
        yaw_deg=np.zeros(1),
    )
    np.testing.assert_allclose(gaps[0, :2], [[0.5, 0.0], [0.1, 0.0]], atol=1e-12)
    assert np.isinf(gaps[0, 2:]).all()

    neighbors = np.full((1, 32, 2, 3), -1, dtype=np.int8)
    neighbors[0, 0, 1, 0] = 2
    neighbors[0, 1, 0, 0] = 1
    np.testing.assert_array_equal(unique_attachment_indices(neighbors), [[0, 1, 2]])
    observations = unique_attachment_observations(
        endpoint_neighbor=neighbors,
        endpoint_gap_m=gaps,
        observed_endpoint_sensor_m=observed,
    )
    assert observations.pair_count == 1
    np.testing.assert_allclose(observations.first_gap_m, [0.0])
    np.testing.assert_allclose(observations.second_gap_m, [0.1])
    np.testing.assert_allclose(observations.observed_endpoint_separation_m, [0.1])
    summary = summarize_attachment_observations(observations)
    assert summary["support_bands"]["0.10m"]["both_endpoints"] == 1
    assert summary["support_bands"]["0.10m"]["both_fraction"] == 1.0


def test_inactive_index_contract_fails_closed() -> None:
    axis = np.zeros((1, 32, 3, 3))
    index = np.full((1, 32), -1); index[0, 3] = 0
    mask = np.zeros((1, 32), dtype=bool)
    try:
        endpoint_support_gaps(
            axis_control_current_sensor_m=axis,
            primitive_index=index,
            primitive_mask=mask,
            primitive_endpoints_world_m=np.zeros((1, 2, 3)),
            sensor_xyz_m=np.zeros((1, 3)),
            yaw_deg=np.zeros(1),
        )
    except ValueError as error:
        assert "inactive" in str(error)
    else:
        raise AssertionError("invalid inactive primitive index was accepted")
