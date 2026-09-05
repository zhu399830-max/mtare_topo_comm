from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[3] / "tools/v3/execute_gse_low_support_endpoint_observability_audit_v1.py"
SPEC = importlib.util.spec_from_file_location("gse_low_support_observability", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_probability_uses_five_event_interface() -> None:
    value = MODULE._probability(np.asarray([0.0]), np.asarray([[0.0, 0.0]]))
    assert value.shape == (1, 5)
    assert np.all(value[:, 3:] == 0.0)
    assert np.allclose(value.sum(axis=1), 1.0)


def test_geometry_signature_is_permutation_invariant() -> None:
    rng = np.random.default_rng(3)
    token = rng.normal(size=(2, 3, 6, 40)).astype(np.float32)
    token[..., 0] = rng.uniform(0.05, 1.0, size=(2, 3, 6))
    first = MODULE._geometry_signature(token)
    permuted = token[:, :, [4, 0, 5, 2, 1, 3]]
    np.testing.assert_allclose(first, MODULE._geometry_signature(permuted), rtol=1e-6, atol=1e-6)


def test_nearest_labels_uses_fit_standardization_and_stable_order() -> None:
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    labels = np.asarray([0, 1, 2])
    nearest, distance = MODULE._nearest_labels(reference, labels, np.asarray([[0.9, 0.0]]), k=2)
    assert nearest.tolist() == [[1, 0]]
    assert distance[0, 0] < distance[0, 1]


def test_endpoint_outcome_separates_class_error_from_missing_proposal() -> None:
    probability = np.zeros((3, 5), dtype=np.float32)
    probability[0, :3] = (0.01, 0.98, 0.01)
    probability[1, :3] = (0.6, 0.1, 0.3)
    probability[2, :3] = (0.01, 0.01, 0.98)
    assert MODULE._endpoint_outcome(np.asarray([0]), 2, probability) == "event_misclassified"
    assert MODULE._endpoint_outcome(np.asarray([1]), 2, probability) == "proposal_missing"
    assert MODULE._endpoint_outcome(np.asarray([2]), 2, probability) == "correct_event_proposal"
