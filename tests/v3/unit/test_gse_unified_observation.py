from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_unified_observation import (
    align_by_global_sequence_index,
    replace_normalized_slope,
)


def test_alignment_uses_global_identity_not_row_order() -> None:
    actual = align_by_global_sequence_index(
        np.asarray([7, 2, 9]), np.asarray([9, 7, 2]), np.asarray([90.0, 70.0, 20.0])
    )
    assert np.array_equal(actual, np.asarray([70.0, 20.0, 90.0], dtype=np.float32))


def test_alignment_rejects_missing_or_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="identity contract"):
        align_by_global_sequence_index(
            np.asarray([1, 2]), np.asarray([1, 1]), np.asarray([3.0, 4.0])
        )


def test_replacement_changes_only_normalized_slope() -> None:
    original = np.arange(3 * 146, dtype=np.float32).reshape(3, 146) / 1000.0
    unified, audit = replace_normalized_slope(
        original, np.asarray([3, 8, 5]), np.asarray([5, 3, 8]),
        np.asarray([-4.5, 9.0, 18.0], dtype=np.float32),
    )
    assert audit.changed_columns == (10,)
    assert audit.unchanged_columns_byte_exact
    assert np.array_equal(unified[:, 10], np.asarray([0.2, 0.4, -0.1], dtype=np.float32))
    assert original[:, :10].tobytes() == unified[:, :10].tobytes()
    assert original[:, 11:].tobytes() == unified[:, 11:].tobytes()


def test_replacement_rejects_nonfinite_or_out_of_domain_slope() -> None:
    original = np.zeros((2, 146), dtype=np.float32)
    with pytest.raises(ValueError):
        replace_normalized_slope(original, np.asarray([1, 2]), np.asarray([1, 2]), np.asarray([0.0, 46.0]))
    with pytest.raises(ValueError):
        replace_normalized_slope(original, np.asarray([1, 2]), np.asarray([1, 2]), np.asarray([0.0, np.nan]))
