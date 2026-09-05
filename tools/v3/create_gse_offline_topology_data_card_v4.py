#!/usr/bin/env python3
"""Create the operation-bound C09 node-gated V4 topology Data Card."""

from __future__ import annotations

import copy
import json

from _bootstrap import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v3.json"
DESTINATION = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_offline_topology_validation_v4.json"


def main() -> int:
    if DESTINATION.exists():
        raise RuntimeError("C09 V4 Data Card already exists")
    card = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    card.update({
        "card_id": "gse_offline_topology_validation_v4",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4",
        "purpose": "C09 causal topology corrective with the separately frozen open-set node-generation gate before the frozen pair association ensemble; execute all baselines first so a main-method safety failure cannot erase comparison evidence.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user",
        "approved_at": "2026-08-26T17:25:00+08:00",
        "authorized_gates": [4], "authorized_operations": ["topology_replay"],
        "scope": "One immutable C09 V4 replay over the same 10 worlds/2054 traversals/24462 sequences, four methods and 243 graph settings. Add only the C01-C08-frozen matchability threshold 0.982292910416921 before V3 node creation; pair threshold/domain and all gates unchanged.",
        "confirmation_reference": "Standing user authorization for the full GSE-Graph paper workflow and autonomous corrective execution without repeated approval prompts.",
    }
    card["source"]["raw_sources"].append(
        "Sealed PASS node-matchability ensemble calibration V1: three frozen matchability heads, equal weights, threshold 0.982292910416921 selected only on C07-C08, zero C09/C10/M-TARE."
    )
    card["sampling"]["node_matchability_operating_point"] = {
        "selection_observations": 45942,
        "positive": 13693,
        "negative": 32249,
        "threshold": 0.982292910416921,
        "precision": 0.9900559353635798,
        "false_accept_rate": 0.009944064636420136,
        "recall": 0.3490104432921931,
    }
    card["teacher"]["planner_consistency_plan"] = (
        "For every current observation, the frozen three-seed matchability mean first decides whether a predicted non-corridor event may trigger a node. "
        "Only accepted node events enter the independently frozen pair verifier; no identity, future observation or graph outcome is consumed. "
        "Ambiguous association remains provisional and edges remain physical-traversal-only."
    )
    card["methods"]["main"] = (
        "Risk-calibrated GSE structural observations; frozen equal-weight matchability gate at 0.982292910416921 for non-corridor node triggers; "
        "then frozen equal-weight exit-token pair verifier at 0.9431912302970886 within 16 m; unchanged graph grid and traversal-only edges."
    )
    card["methods"]["execution_order"] = (
        "Complete GSE-rule, exit-only-rule and nonlearning-rule baselines before the full GSE method; this changes only evidence availability, not selection or gates."
    )
    card["methods"]["fallback"] = (
        "If no main configuration satisfies association safety or the final graph gate, seal FAIL after preserving completed baselines; do not change either threshold, seed weights, radius, Teacher, frames or planner."
    )
    card["evidence"].append(
        "Node-generation decision for every selected replay frame: all three matchability scores, equal-weight score, frozen threshold and acceptance/rejection."
    )
    DESTINATION.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DESTINATION.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
