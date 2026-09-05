from __future__ import annotations

import pytest
import torch

from execute_gse_relational_exit_transport_readiness_v1r import flatnonzero_compat


def test_flatnonzero_compat_matches_expected_vector_indices() -> None:
    value = torch.tensor([False, True, False, True, True])
    assert flatnonzero_compat(value).tolist() == [1, 3, 4]


def test_flatnonzero_compat_rejects_matrix() -> None:
    with pytest.raises(ValueError, match="one vector"):
        flatnonzero_compat(torch.ones(2, 2, dtype=torch.bool))
