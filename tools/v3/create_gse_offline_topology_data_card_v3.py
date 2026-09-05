#!/usr/bin/env python3
"""Materialize the operation-bound C09 V3 Data Card from the audited V2 card."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v2.json"
DESTINATION = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v3.json"


def main() -> int:
    if DESTINATION.exists():
        raise RuntimeError("C09 V3 Data Card already exists")
    card = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    selection_worlds = [f"S{family:02d}_{suffix}_C{copy_id:02d}" for family, suffix in (
        (1, "flat_tree_small"), (2, "3d_tree_small"),
        (3, "flat_unicyclic_small"), (4, "3d_unicyclic_small"),
        (5, "flat_branch_medium"), (6, "3d_branch_medium"),
        (7, "flat_loop_rich"), (8, "3d_loop_rich"),
        (9, "flat_complex"), (10, "3d_complex"),
    ) for copy_id in (7, 8)]
    card.update({
        "card_id": "gse_offline_topology_validation_v3",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V3",
        "purpose": "C09 causal topology validation of the frozen three-seed distance-aware exit-token ensemble, with unchanged structural grid, deployable baselines, scientific gates, and zero C10/M-TARE reads.",
        "estimated_cost": {
            "compute": "One batched three-seed frozen verifier pass over all unique <=16 m C09 spatial candidate pairs, followed by 59,442,660 typed graph updates across 24,300 world/config/seed replays; zero optimizer/model updates.",
            "wall_time_hours": 8.0,
            "disk_gb": 5.0,
        },
    })
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-26T00:00:00+08:00",
        "scope": "One immutable C09 V3 topology validation over 10 worlds, 2,054 directed traversals and 24,462 sequences; fixed three-seed equal-weight verifier, threshold 0.9431912302970886, 16 m candidate domain, four methods and 243 graph settings.",
        "authorized_operations": ["topology_replay"],
        "authorized_gates": [4],
        "confirmation_reference": "The user explicitly granted standing authorization for the complete GSE-Graph paper workflow and instructed execution without repeated approval prompts.",
    }
    card["worlds"]["threshold_calibration"] = selection_worlds
    card["worlds"]["checkpoint_selection"] = selection_worlds
    card["source"]["raw_sources"].extend([
        "Immutable V2 exit-token verifier component run: all three best checkpoints and normalizations are consumed despite the sealed single-seed aggregate FAIL.",
        "Sealed PASS distance-aware ensemble calibration V1: equal seed weights, threshold 0.9431912302970886 and maximum candidate distance 16 m were selected only on C07-C08.",
    ])
    card["sampling"].update({
        "association_fit_pairs": 135232,
        "association_selection_pairs": 45372,
        "association_selection_pairs_within_16m": 31469,
        "association_selection_positive_pairs_within_16m": 12813,
        "association_selection_negative_pairs_within_16m": 18656,
        "association_operating_point": {
            "threshold": 0.9431912302970886,
            "precision": 0.9900537634408603,
            "false_accept_rate": 0.009946236559139785,
            "recall": 0.28744244127058455,
        },
    })
    card["teacher"]["planner_consistency_plan"] = (
        "Graph builders receive only current/past typed observations, current pose and pair-local frozen verifier scores. "
        "Precomputed scores use exactly two observations plus their measured distance and never use Teacher identity, labels, future aggregates or graph state. "
        "Nodes are event-triggered or metric anchors; edges appear only after recorded physical traversal."
    )
    card["methods"] = {
        "main": "Risk-calibrated GSE observations with a frozen three-seed equal-weight exit-token verifier; candidates are limited to the predeclared graph radius and hard 16 m deployment domain, accepted at the frozen 0.9431912302970886 threshold, and ambiguous accepted matches create provisional nodes.",
        "baselines": [
            "The same GSE structural observations with deterministic distance/exit-angle association.",
            "Frozen M1D exit-only observations with rule association.",
            "Deterministic non-learning geometry observations with rule association.",
        ],
        "diagnostic": "GT-TNG evaluator-only ceiling.",
        "fallback": "If the gate fails, preserve the immutable FAIL and diagnose node generation versus association fragmentation; do not tune the verifier, threshold, candidate domain, data, Teacher or planner on C09.",
    }
    card["evidence"] = [
        "All 243 sweep rows and selected configuration for each of four methods.",
        "Per selected world/seed nodes, trace-verified edges, decision traces and every evaluated frozen association candidate with seed scores, ensemble score, distance and rejection reason.",
        "Node/edge F1, association precision and false-loop rate, connected-component and cycle-rank bias, source verification, raw log, environment, RUN_STATE and SHA-256 seal.",
        "After PASS only: paper PNG/PDF/SVG/CSV/JSON/provenance bundle; failed runs retain compact diagnostic evidence.",
    ]
    DESTINATION.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DESTINATION.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
