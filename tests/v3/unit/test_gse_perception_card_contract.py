from __future__ import annotations

import json
from pathlib import Path

from mtare_topo.governance import validate_data_card


ROOT = Path(__file__).resolve().parents[3]
CARD = ROOT / "configs/v3/gate3/data_cards/gse_perception_validation_v1.json"


def test_perception_validation_card_has_exact_population_and_isolation() -> None:
    card = json.loads(CARD.read_text(encoding="utf-8"))
    report = validate_data_card(card)
    assert report.passed, report.errors
    assert len(card["worlds"]["train"]) == 80
    trajectories = card["trajectories"]
    assert len(trajectories) == 10
    assert sum(row["sequence_count"] for row in trajectories) == 24462
    assert sum(row["unique_frame_count"] for row in trajectories) == 32678
    assert sum(row["directed_traversal_count"] for row in trajectories) == 2054
    assert abs(sum(row["distance_m"] for row in trajectories) - 31654.752491194) < 1e-9
    assert all(row["world"].endswith("_C09") for row in trajectories)
    assert {key: card["leakage_audit"][key] for key in (
        "train_worlds_read",
        "strict_test_worlds_read",
        "mtare_worlds_read",
        "future_frames_as_model_input",
        "gt_identity_as_model_input",
        "model_updates",
        "optimizer_steps",
        "checkpoint_selection_changes",
    )} == {
        "train_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "future_frames_as_model_input": 0,
        "gt_identity_as_model_input": 0,
        "model_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_selection_changes": 0,
    }
