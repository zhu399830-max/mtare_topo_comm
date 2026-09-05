from __future__ import annotations

import pytest

from mtare_topo.evaluation.gse_causal_event_supervision import (
    contiguous_structural_episodes,
    transition_supervision_alignment,
)


def _row(index: int, event: str, identity: str | None, traversal: str = "w:e:d0") -> dict:
    return {
        "traversal_id": traversal,
        "sequence_index": index,
        "frame_index": index + 4,
        "traversal_arc_m": float(index + 4),
        "event": event,
        "identity": identity,
    }


def test_structural_episode_partition_respects_corridor_and_identity() -> None:
    rows = [
        _row(0, "corridor", None),
        _row(1, "turn", "a"),
        _row(2, "turn", "a"),
        _row(3, "corridor", None),
        _row(4, "turn", "a"),
        _row(5, "junction", "b"),
    ]
    episodes = contiguous_structural_episodes(rows)
    assert [(item.event, item.identity, item.frame_count) for item in episodes] == [
        ("turn", "a", 2),
        ("turn", "a", 1),
        ("junction", "b", 1),
    ]


def test_structural_episode_rejects_corridor_identity_drift() -> None:
    with pytest.raises(ValueError, match="identity"):
        contiguous_structural_episodes([_row(0, "corridor", "bad")])


def test_transition_alignment_separates_proposal_and_confirmation() -> None:
    teacher = [_row(index, "geometry_transition", "cp") for index in range(4)]
    labels = [
        {
            "parent_id": "w",
            "traversal_id": "w:e:d0",
            "sequence_index": index,
            "identity": "cp",
            "traversal_arc_m": float(index + 4),
        }
        for index in range(4)
    ]
    point = {
        "emitted": True,
        "identity_assignment": {"identity": "cp"},
        "causal_matches": [[{
            "traversal_id": "w:e:d0",
            "emission_arc_m": 7.0,
            "causal_delay_m": 3.5,
        }], [{
            "traversal_id": "w:e:d1",
            "emission_arc_m": 7.0,
            "causal_delay_m": 3.5,
        }]],
    }
    teacher += [_row(index, "corridor", None, "w:e:d1") for index in range(4)]
    # The second direction belongs to the same final identity in the real data;
    # include its final rows/labels here so exact coverage is exercised.
    teacher[-4:] = [_row(index, "geometry_transition", "cp", "w:e:d1") for index in range(4)]
    labels += [
        {
            "parent_id": "w",
            "traversal_id": "w:e:d1",
            "sequence_index": index,
            "identity": "cp",
            "traversal_arc_m": float(index + 4),
        }
        for index in range(4)
    ]
    result = transition_supervision_alignment(
        teacher_rows=teacher,
        proof_points=[point],
        proof_labels=labels,
        history_lengths=(5, 8),
    )
    assert result["final_transition_labels"] == 8
    assert result["labels_before_closed_confirmation"] == 6
    assert result["labels_at_closed_confirmation"] == 2
    assert result["history_available_at_confirmation"] == {"5": 2, "8": 2}

