from dataclasses import FrozenInstanceError, fields, replace

import pytest

from mtare_topo.representation.gse_structure_context import CausalStructureContext


def context(**changes):
    values = dict(
        timestamp_s=1.0,
        coordinate_frame="robot/current",
        source_frame_ids=(2, 3, 5, 7, 8),
        source_timestamps_s=(0.1, 0.3, 0.6, 0.8, 1.0),
        sequence_id="sequence_0",
    )
    return CausalStructureContext(**(values | changes))


def test_valid_five_frame_context_is_immutable_and_has_no_gt_or_graph_fields():
    item = context()
    assert item.source_frame_ids == (2, 3, 5, 7, 8)
    assert {field.name for field in fields(item)} == {
        "timestamp_s", "coordinate_frame", "source_frame_ids", "source_timestamps_s", "sequence_id"
    }
    with pytest.raises(FrozenInstanceError):
        item.timestamp_s = 2.0
    with pytest.raises(TypeError):
        item.source_frame_ids[0] = 99
    with pytest.raises(TypeError):
        CausalStructureContext(**{field.name: getattr(item, field.name) for field in fields(item)}, node_id=4)


def test_delayed_publication_and_integer_clock_are_valid_without_consecutive_ids():
    item = context(timestamp_s=5, source_timestamps_s=(0, 1, 2, 3, 4))
    assert item.timestamp_s == 5.0 and type(item.timestamp_s) is float
    assert all(type(value) is float for value in item.source_timestamps_s)


def test_future_frame_is_rejected_and_replace_revalidates():
    with pytest.raises(ValueError, match="future frame"):
        replace(context(), timestamp_s=0.9)


@pytest.mark.parametrize("frame_ids", [
    (2, 3, 3, 7, 8), (2, 3, 8, 7, 9), (2, 3, 4, 5),
    [2, 3, 5, 7, 8], (True, 3, 5, 7, 8), (-1, 3, 5, 7, 8),
    (2.0, 3, 5, 7, 8), ("2", 3, 5, 7, 8),
])
def test_duplicate_unordered_or_invalid_frame_identity_rejected(frame_ids):
    with pytest.raises(ValueError, match="source_frame_ids"):
        context(source_frame_ids=frame_ids)


@pytest.mark.parametrize("times", [
    (0.1, 0.3, 0.3, 0.8, 1.0), (0.1, 0.6, 0.3, 0.8, 1.0),
    (0.1, 0.3, 0.6, 0.8), [0.1, 0.3, 0.6, 0.8, 1.0],
    (0.1, 0.3, float("nan"), 0.8, 1.0),
    (0.1, 0.3, 0.6, 0.8, float("inf")),
    (0.1, 0.3, 0.6, 0.8, True), (0.1, 0.3, 0.6, 0.8, "1.0"),
])
def test_invalid_source_timestamps_rejected(times):
    with pytest.raises(ValueError, match="source_timestamps_s"):
        context(source_timestamps_s=times)


@pytest.mark.parametrize("timestamp", [True, "1", float("nan"), float("inf"), -(10**1000)])
def test_invalid_observation_timestamp_rejected(timestamp):
    with pytest.raises(ValueError, match="timestamp_s"):
        context(timestamp_s=timestamp)


@pytest.mark.parametrize("name", ["coordinate_frame", "sequence_id"])
@pytest.mark.parametrize("value", ["", " ", " map", "base link", "map\x00", None, 3])
def test_invalid_frame_or_sequence_identifier_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        context(**{name: value})
