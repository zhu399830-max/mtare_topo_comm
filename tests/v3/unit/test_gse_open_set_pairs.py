import numpy as np
import pytest

from mtare_topo.data.gse_open_set_pairs import (
    cohort_number,
    compact_global_sequence_identity,
    corrective_partition,
    nearest_past_open_set_pairs,
)


def _row(i, xyz, valid, parent="S01_flat_tree_small_C01"):
    return {
        "parent_id": parent,
        "world_sequence_row": i,
        "global_sequence_index": 100 + i,
        "sensor_xyz_m": xyz,
        "association_valid": valid,
    }


def test_partition_is_frozen_to_c01_c08():
    assert cohort_number("S10_3d_complex_C08") == 8
    assert corrective_partition("S01_flat_tree_small_C06") == "fit"
    assert corrective_partition("S01_flat_tree_small_C07") == "selection"
    with pytest.raises(ValueError, match="C01--C08"):
        corrective_partition("S01_flat_tree_small_C09")


def test_open_set_pair_is_nearest_strictly_past_with_stable_tie_break():
    rows = [
        _row(0, [0.0, 0.0, 0.0], True),
        _row(1, [2.0, 0.0, 0.0], True),
        _row(2, [1.0, 0.0, 0.0], False),
        _row(3, [30.0, 0.0, 0.0], False),
    ]
    pairs = nearest_past_open_set_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["anchor_global_sequence_index"] == 102
    assert pairs[0]["paired_global_sequence_index"] == 100
    assert pairs[0]["spatial_distance_m"] == 1.0


def test_open_set_builder_is_deterministic_under_input_permutation():
    rows = [
        _row(0, [0.0, 0.0, 0.0], True),
        _row(1, [1.0, 0.0, 0.0], False),
        _row(2, [2.0, 0.0, 0.0], False),
    ]
    assert nearest_past_open_set_pairs(rows) == nearest_past_open_set_pairs(rows[::-1])


def test_duplicate_sequence_row_fails_closed():
    rows = [_row(0, [0.0, 0.0, 0.0], True), _row(0, [1.0, 0.0, 0.0], False)]
    with pytest.raises(ValueError, match="duplicate"):
        nearest_past_open_set_pairs(rows)


def test_sparse_global_ids_map_to_compact_rows_without_losing_identity():
    ordered, compact = compact_global_sequence_identity([208227, 0, 2, 100002])
    np.testing.assert_array_equal(ordered, np.asarray([0, 2, 100002, 208227]))
    assert compact == {0: 0, 2: 1, 100002: 2, 208227: 3}
    left_global = np.asarray([100002, 208227])
    right_global = np.asarray([0, 2])
    left_compact = np.asarray([compact[int(value)] for value in left_global])
    right_compact = np.asarray([compact[int(value)] for value in right_global])
    np.testing.assert_array_equal(ordered[left_compact], left_global)
    np.testing.assert_array_equal(ordered[right_compact], right_global)


def test_compact_global_ids_reject_duplicates_not_gaps():
    with pytest.raises(ValueError, match="unique"):
        compact_global_sequence_identity([0, 2, 2, 5])
