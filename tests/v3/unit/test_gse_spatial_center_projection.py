import numpy as np
import pytest

from mtare_topo.evaluation.gse_spatial_center_projection import (
    combine_seed_projections,
    project_local_vectors,
)


def test_route_local_projection_and_seed_uncertainty() -> None:
    sensor = np.asarray([[10.0, 20.0, 2.0], [1.0, 2.0, 3.0]])
    basis = np.asarray([
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        np.eye(3),
    ])
    vector = np.asarray([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    projected = project_local_vectors(vector, sensor, basis)
    np.testing.assert_allclose(projected, [[7.0, 22.0, 6.0], [5.0, 7.0, 9.0]])

    seeds = np.stack((vector - 1.0, vector, vector + 1.0))
    combined = combine_seed_projections(seeds, sensor, basis)
    np.testing.assert_allclose(combined["predicted_local_vector_m"], vector)
    np.testing.assert_allclose(combined["projected_center_xyz_m"], projected)
    np.testing.assert_allclose(
        combined["position_uncertainty_m"], np.sqrt(2.0), atol=1e-6,
    )


def test_degenerate_runtime_basis_falls_back_to_sensor() -> None:
    sensor = np.asarray([[4.0, 5.0, 6.0]])
    basis = np.zeros((1, 3, 3))
    seeds = np.ones((3, 1, 3))
    combined = combine_seed_projections(seeds, sensor, basis)
    np.testing.assert_array_equal(combined["projected_center_xyz_m"], sensor)
    np.testing.assert_array_equal(combined["position_uncertainty_m"], [0.0])


def test_projection_rejects_population_drift() -> None:
    with pytest.raises(ValueError, match="shape"):
        project_local_vectors(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 3, 3)))


def test_float32_world_coordinate_rounding_stays_sub_millimetre() -> None:
    sensor = np.asarray([[507.95980835, 227.60353088, -0.62996298]], dtype=np.float32)
    basis = np.asarray([[
        [0.982976258, -0.183732659, 0.0],
        [0.183732659, 0.982976258, 0.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    seeds = np.asarray([
        [[2.62460136, -0.49104330, 0.71656674]],
        [[12.0, -0.44852272, 0.60379827]],
        [[2.09676194, -0.38711911, 0.76851952]],
    ], dtype=np.float32)
    combined = combine_seed_projections(seeds, sensor, basis)
    assert 0.0 < float(combined["linear_identity_max_abs_m"]) < 1e-4
