"""Synthetic full identity indices only: no shard, sensor, or checkpoint I/O."""
from copy import deepcopy

import numpy as np
import pytest

from mtare_topo.data.gse_review_source_index_v1 import (
    VARIANTS, SOURCE_SEQUENCE_POPULATION, validate_review_source_indices,
)


def inputs(*, lengths=(30, 12), parent="S01_synthetic_C01"):
    partition = "c07" if parent.endswith("C07") else "fit"
    sa, ta, sx, tx = {}, {}, {}, {}
    traversals = [parent + f":edge{i}:d{i % 2}" for i in range(len(lengths))]
    rows, tids, sequence_ids = [], [], []
    frame_start, seq_start = 0, 100
    for length, traversal in zip(lengths, traversals):
        q = max(0, length - 4)
        rows.extend(tuple(range(frame_start + j, frame_start + j + 5)) for j in range(q))
        tids.extend([traversal] * q)
        sequence_ids.extend(range(seq_start, seq_start + q))
        frame_start += length; seq_start += q
    for index, variant in enumerate(VARIANTS):
        sa[variant] = {"schema_version": "primitive_relation_p1a_sensor_shard_v1", "parent_id": parent,
            "partition": partition, "geometry_realization": variant, "sensor_shape": [16, 720],
            "maximum_range_m": 50., "student_pose_input_forbidden": True, "traversal_ids": list(traversals)}
        ta[variant] = {"schema_version": "primitive_relation_p1b_teacher_shard_v1", "parent_id": parent,
            "partition": partition, "geometry_realization": variant, "window_frames": 5, "maximum_slots": 32,
            "student_identity_input_forbidden": True, "traversal_ids": list(tids)}
        sx[variant] = {"global_frame_index": np.arange(1000, 1000 + sum(lengths), dtype=np.int64),
            "local_frame_index": np.concatenate([np.arange(n, dtype=np.int32) for n in lengths]),
            "traversal_index": np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(lengths)]),
            "route_arc_m": np.concatenate([np.arange(n, dtype=np.float64) * (1 + index * .1) for n in lengths])}
        tx[variant] = {"source_global_sequence_index": np.asarray(sequence_ids, dtype=np.int64),
            "variant_global_sequence_index": np.asarray(sequence_ids, dtype=np.int64) + index * SOURCE_SEQUENCE_POPULATION,
            "frame_row": np.asarray(rows, dtype=np.int32).reshape(-1, 5)}
    return {"parent_id": parent, "sensor_attrs_by_variant": sa, "teacher_attrs_by_variant": ta,
            "sensor_arrays_by_variant": sx, "teacher_arrays_by_variant": tx}


def test_full_index_validates_and_enumerates_without_nominating_structures():
    args = inputs()
    report = validate_review_source_indices(**args)
    assert report.partition == "fit"
    assert report.counts["logical_sequences"] == 34
    assert report.counts["variant_observations"] == 102
    assert report.counts["unique_variant_raw_frames"] == 126
    assert report.counts["eligible_21_decision_windows"] == 6
    assert report.counts["structure_candidates_nominated"] == report.counts["structure_labels_created"] == 0
    assert report.short_intervals[0]["sequence_count"] == 8 and report.short_intervals[0]["missing"] == 13
    windows = tuple(report.intervals[0].iter_windows())
    assert len(windows) == 6
    for window in windows:
        assert all(len(v.sequence_rows) == len(v.frame_rows) == len(v.source_sequence_ids) == 21 for v in window.variants)
        assert len(set(f for row in window.variants[0].frame_rows for f in row)) == 25
        assert len({v.source_sequence_ids for v in window.variants}) == 1
    assert windows[0].variants[0].decision_arc_m[0] == 4.
    assert windows[0].variants[1].decision_arc_m[0] == 4.4
    assert report.continuity_requires_source_audit and report.metadata_truth_requires_source_reader
    assert report.duration_s is None


def test_repeated_validation_and_mapping_order_are_deterministic_no_input_mutation():
    args = inputs(); before = deepcopy(args)
    a = validate_review_source_indices(**args)
    b = validate_review_source_indices(**{k: dict(reversed(list(v.items()))) if type(v) is dict else v for k, v in args.items()})
    assert a == b
    for group in ("sensor_arrays_by_variant", "teacher_arrays_by_variant"):
        for variant in VARIANTS:
            for field in args[group][variant]: np.testing.assert_array_equal(args[group][variant][field], before[group][variant][field])
    args["sensor_arrays_by_variant"][VARIANTS[0]]["route_arc_m"][:] = 1000
    assert a.intervals[0].variants[0].decision_arc_m[0] == 4.  # output snapshot, not borrowed mutable arrays


@pytest.mark.parametrize("length", [1, 4, 5, 24])
def test_short_traversal_is_normal_insufficiency_not_fill_or_error(length):
    report = validate_review_source_indices(**inputs(lengths=(length,)))
    assert report.counts["eligible_21_decision_windows"] == 0
    assert report.short_intervals[0]["sequence_count"] == max(0, length - 4)
    assert tuple(report.intervals[0].iter_windows()) == ()


def test_c07_is_separate_source_partition_not_strict_test():
    report = validate_review_source_indices(**inputs(parent="S02_synthetic_C07", lengths=(25,)))
    assert report.partition == "c07" and report.counts["eligible_21_decision_windows"] == 1


@pytest.mark.parametrize("fault", ["C08", "missing_variant", "schema", "partition", "parent", "variant",
    "bool_window", "pose_boundary", "scan_field", "geometry_field", "float_identity", "bool_identity",
    "global_duplicate", "unsigned_reverse", "local_gap", "traversal_bounds", "duplicate_traversal",
    "arc_nan", "arc_reverse", "frame_future", "frame_cross_traversal", "source_duplicate", "source_limit",
    "variant_offset", "paired_source_mismatch", "paired_frames_mismatch", "teacher_traversal_length",
    "missing_window", "shuffled_window", "source_gap"])
def test_drift_and_leakage_boundaries(fault):
    a = inputs(); v = VARIANTS[0]
    sa, ta = a["sensor_attrs_by_variant"][v], a["teacher_attrs_by_variant"][v]
    sx, tx = a["sensor_arrays_by_variant"][v], a["teacher_arrays_by_variant"][v]
    if fault == "C08": a = inputs(parent="S01_synthetic_C08")
    elif fault == "missing_variant": a["sensor_arrays_by_variant"].pop(VARIANTS[1])
    elif fault == "schema": ta["schema_version"] = "other"
    elif fault == "partition": sa["partition"] = "c07"
    elif fault == "parent": ta["parent_id"] = "S02_synthetic_C01"
    elif fault == "variant": sa["geometry_realization"] = "c1_mixed"
    elif fault == "bool_window": ta["window_frames"] = True
    elif fault == "pose_boundary": sa["student_pose_input_forbidden"] = False
    elif fault == "scan_field": sx["range_m"] = np.zeros((42, 16, 720), dtype=np.float32)
    elif fault == "geometry_field": tx["axis_control_current_sensor_m"] = np.zeros((34, 32, 3, 3))
    elif fault == "float_identity": tx["source_global_sequence_index"] = tx["source_global_sequence_index"].astype(float)
    elif fault == "bool_identity": sx["traversal_index"] = sx["traversal_index"].astype(bool)
    elif fault == "global_duplicate": sx["global_frame_index"][1] = sx["global_frame_index"][0]
    elif fault == "unsigned_reverse": sx["global_frame_index"] = sx["global_frame_index"][::-1].astype(np.uint64)
    elif fault == "local_gap": sx["local_frame_index"][1] = 10
    elif fault == "traversal_bounds": sx["traversal_index"][0] = 2
    elif fault == "duplicate_traversal": sa["traversal_ids"].append(sa["traversal_ids"][0])
    elif fault == "arc_nan": sx["route_arc_m"][0] = np.nan
    elif fault == "arc_reverse": sx["route_arc_m"][1] = sx["route_arc_m"][0]
    elif fault == "frame_future": tx["frame_row"][0, -1] = 42
    elif fault == "frame_cross_traversal": tx["frame_row"][0] = np.arange(28, 33)
    elif fault == "source_duplicate": tx["source_global_sequence_index"][1] = tx["source_global_sequence_index"][0]
    elif fault == "source_limit": tx["source_global_sequence_index"] += SOURCE_SEQUENCE_POPULATION
    elif fault == "variant_offset": tx["variant_global_sequence_index"] += 1
    elif fault == "paired_source_mismatch":
        tx["source_global_sequence_index"] += 1; tx["variant_global_sequence_index"] += 1
    elif fault == "paired_frames_mismatch": sx["global_frame_index"] += 1
    elif fault == "teacher_traversal_length": ta["traversal_ids"].pop()
    elif fault == "missing_window":
        for key in tx: tx[key] = tx[key][1:]
        ta["traversal_ids"] = ta["traversal_ids"][1:]
    elif fault == "shuffled_window": tx["frame_row"][[0, 1]] = tx["frame_row"][[1, 0]]
    else:
        tx["source_global_sequence_index"][1:] += 1; tx["variant_global_sequence_index"][1:] += 1
    with pytest.raises(ValueError): validate_review_source_indices(**a)


def test_p1a_lookup_table_can_reorder_without_changing_actual_identity():
    a = inputs(); v = VARIANTS[1]
    a["sensor_attrs_by_variant"][v]["traversal_ids"].reverse()
    a["sensor_arrays_by_variant"][v]["traversal_index"] = 1 - a["sensor_arrays_by_variant"][v]["traversal_index"]
    assert validate_review_source_indices(**a) == validate_review_source_indices(**inputs())
