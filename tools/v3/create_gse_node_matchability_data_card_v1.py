#!/usr/bin/env python3
"""Create the operation-bound node-matchability calibration Data Card."""

from __future__ import annotations

import copy
import json

from _bootstrap import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_distance_aware_ensemble_calibration_v1.json"
DESTINATION = PROJECT_ROOT / "configs/v3/gate4/data_cards/gse_node_matchability_ensemble_calibration_v1.json"


def main() -> int:
    if DESTINATION.exists():
        raise RuntimeError("node-matchability Data Card already exists")
    card = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    card.update({
        "card_id": "gse_node_matchability_ensemble_calibration_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1",
        "purpose": "Freeze a three-seed open-set structural-node generation threshold from the already-trained V2 matchability heads on C07-C08 only, before another C09 topology replay.",
        "approval": {
            "status": "APPROVED", "approved_by": "user",
            "approved_at": "2026-08-26T17:15:00+08:00",
            "authorized_gates": [4], "authorized_operations": ["threshold_calibration"],
            "scope": "One immutable read-only calibration over 188126 C01-C08 observations: 142184 retained fit provenance and exactly 45942 C07-C08 threshold-selection observations. Equal-weight three-seed matchability only; zero C09/C10/M-TARE.",
            "confirmation_reference": "Standing user authorization for autonomous GSE-Graph corrective work after the reported C09 V3 failure decomposition.",
        },
    })
    card["source"]["raw_sources"] = [
        "Sealed V2 scientific-FAIL component run: three frozen observation feature arrays, three normalizations, three best verifier checkpoints and the exact 188126-row pair-cache observation manifest.",
        "No C09 output, topology result, Teacher identity or planner state is consumed by threshold selection.",
    ]
    card["source"]["partial_reuse"] = "READ_ONLY_FROZEN_MATCHABILITY_HEADS_AND_C01_C08_OBSERVATION_FEATURES_ZERO_UPDATE"
    card["sampling"]["node_gate_target"] = "association_valid: objective structural identity present versus explicit open-set corridor observation"
    card["sampling"]["node_gate_selection_observations"] = 45942
    card["sampling"]["node_gate_selection_positive"] = 13693
    card["sampling"]["node_gate_selection_negative"] = 32249
    card["split"].update({
        "fit": "C01-C06 matchability-head training provenance only; no new gradient or selection.",
        "checkpoint_and_threshold_selection": "All three V2 checkpoints remain frozen; exactly one equal-weight matchability threshold is selected on C07-C08.",
        "future_validation": "C09 topology may be replayed only after this operating point is sealed; C09 cannot change the threshold or seed weights.",
    })
    card["teacher"].update({
        "student_input": "Each frozen head consumes only its normalized 146D deployment observation feature. The calibrator averages three sigmoid matchability scores; no pair candidate, pose, identity or future frame is an input.",
        "planner_consistency_plan": "Online node generation first applies the frozen matchability ensemble to the current observation. Only accepted structural events may trigger a node; accepted nodes then use the independently frozen pair verifier for association, and physical traversal remains required for edges.",
    })
    card["methods"] = {
        "main": "Float64 arithmetic mean of the three frozen V2 observation-matchability sigmoid scores; one non-vacuous threshold.",
        "baseline": "Each single-seed score remains diagnostic; no seed is selected and no learned ensemble weight is allowed.",
        "fallback": "If no threshold satisfies all aggregate/family gates, keep the node gate disabled and preserve the C09 V3 FAIL; do not read C09 to choose a new threshold.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "aggregate": "precision>=0.98, false accept<=0.01, recall>=0.25",
        "per_family": "all ten topology families have non-empty acceptance, precision>=0.95 and recall>=0.10",
        "nonvacuous": True,
    }
    card["estimated_cost"] = {
        "wall_time_hours": 0.17, "disk_gb": 0.1, "host_ram_gb": 2,
        "gpu_memory_gb": 0,
        "compute": "CPU inference through only the already-trained matchability MLP for 188126 observations x 3 seeds; zero backbone inference, optimizer, LiDAR, simulator or C09 read.",
    }
    card["evidence"] = {
        "machine_metrics": "Exact fit/selection positive-negative counts, three score arrays, one arithmetic ensemble, selected threshold, aggregate/per-family metrics, hashes, environment, raw log, RUN_STATE and full seal.",
        "complete_visual_review": "No paper figure is emitted from development threshold calibration.",
        "failure_policy": "Any source/count/score drift, missing family, learned weight, gate failure, model update or C09/C10/M-TARE read seals FAIL.",
    }
    card["retention"] = "Retain compact selection scores, summary, config, log and seal; source feature/checkpoint arrays stay in the sealed V2 run."
    DESTINATION.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DESTINATION.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
