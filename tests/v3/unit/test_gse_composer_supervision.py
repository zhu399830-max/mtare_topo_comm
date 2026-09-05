from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.teacher.gse_composer_supervision import (
    materialize_composer_world_supervision,
    summarize_composer_supervision,
)


def _row(index, traversal, sequence, event="corridor", identity=None, arc=None, curvature=0.0):
    return {
        "global_sequence_index": index, "traversal_id": traversal,
        "sequence_index": sequence, "event": event, "identity": identity,
        "traversal_arc_m": float(sequence if arc is None else arc),
        "curvature_per_m": curvature,
    }


def test_histories_are_left_padded_past_only_and_never_cross_traversal() -> None:
    rows = [_row(0, "a", 0), _row(1, "a", 1), _row(2, "b", 0), _row(3, "b", 1)]
    result = materialize_composer_world_supervision(np.arange(4), rows, [])
    np.testing.assert_array_equal(result.history_row_index[1], [-1, -1, -1, 0, 1])
    np.testing.assert_array_equal(result.valid_history_mask[2], [False, False, False, False, True])
    assert result.history_row_index[3, -2:].tolist() == [2, 3]


def test_transition_fraction_and_turn_curvature_anchor_are_preserved() -> None:
    rows = [
        _row(0, "a", 0),
        _row(1, "a", 1, "turn", "turn0", arc=1, curvature=.1),
        _row(2, "a", 2, "turn", "turn0", arc=2, curvature=.4),
        _row(3, "a", 3, "turn", "turn0", arc=3, curvature=.2),
        _row(4, "a", 4, "geometry_transition", "node0", arc=4),
    ]
    timing = [{
        "traversal_id": "a", "sequence_index": 4,
        "metres_after_boundary": 2.5, "boundary_inside_five_frame_history": True,
    }]
    result = materialize_composer_world_supervision(np.arange(5), rows, timing)
    assert not result.backprojection_valid[1]
    assert result.backprojection_valid[2] and result.backprojection_steps_ago[2] == 0
    assert result.backprojection_valid[3] and result.backprojection_steps_ago[3] == 1
    assert result.backprojection_valid[4] and result.backprojection_steps_ago[4] == 2.5
    summary = summarize_composer_supervision(result)
    assert summary["backprojection_frames"] == {"turn": 2, "geometry_transition": 1}


def test_transition_outside_history_is_retained_as_event_but_not_backprojection() -> None:
    rows = [_row(0, "a", 0, "geometry_transition", "node0")]
    timing = [{
        "traversal_id": "a", "sequence_index": 0,
        "metres_after_boundary": 4.5, "boundary_inside_five_frame_history": False,
    }]
    result = materialize_composer_world_supervision(np.arange(1), rows, timing)
    assert result.metric_target[0] == 2
    assert not result.backprojection_valid[0]


def test_missing_transition_timing_and_future_history_fail_closed() -> None:
    rows = [_row(0, "a", 0, "geometry_transition", "node0")]
    with pytest.raises(RuntimeError, match="lacks timing"):
        materialize_composer_world_supervision(np.arange(1), rows, [])


def test_materializer_plot_accepts_categorical_summary(tmp_path) -> None:
    from materialize_gse_composer_supervision_v1 import _plot

    _plot(tmp_path, {
        "event_frames": {"corridor": 10, "junction": 2, "terminal": 1, "turn": 1, "geometry_transition": 1},
        "backprojection_frames": {"turn": 1, "geometry_transition": 1},
    })
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"gse_composer_supervision_v1.{suffix}").is_file()
