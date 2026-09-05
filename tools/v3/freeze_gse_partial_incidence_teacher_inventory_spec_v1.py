#!/usr/bin/env python3
"""Freeze the partial-incidence reliability Teacher inventory."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_partial_incidence_teacher_inventory_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_partial_incidence_teacher_inventory_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_partial_incidence_teacher_inventory_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("partial-incidence inventory card/spec exists; overwrite is forbidden")
    current = "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
    teacher = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl"
    pair_cache = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"
    objective = "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0/artifacts/teacher/event_center_teacher.npz"
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T17:10:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable C01-C08 read-only partial-incidence Teacher inventory; zero training/inference/C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "gse_partial_incidence_teacher_inventory_v1",
        "title": "Partially observed two-edge junction reliability Teacher inventory",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_PARTIAL_INCIDENCE_TEACHER_INVENTORY_V1",
        "approval": approval,
        "purpose": "Prove that fit-only objective labels are sufficient for a small reliability head limited to junction hypotheses supported by exactly two physical edges.",
        "operation": "read-only Teacher inventory and no-training capacity boundary",
        "worlds": {"fit": "60 worlds S01-S10 C01-C06", "selection": "20 worlds S01-S10 C07-C08", "forbidden": "C09, C10, strict test and M-TARE"},
        "sampling": {
            "raw_observations": 188126, "fit_observations": 142184, "selection_observations": 45942,
            "expected_fit_committed_hypotheses": 718, "expected_selection_committed_hypotheses": 234,
            "expected_fit_two_edge_junctions": 106, "expected_fit_positive_negative": "84/22",
            "expected_selection_two_edge_junctions": 35, "expected_selection_positive_negative": "28/7",
            "independent_unit": "One union-replay committed junction hypothesis with exactly two unique physical-edge identities",
        },
        "trajectories": {"source": "All sealed directed traversals in C01-C08", "spacing_m": 1.0, "temporal_context": "5 causal frames, no future"},
        "split_logic": "C01-C06 alone decides data sufficiency and any later reliability training. C07-C08 is inventory-only selection evidence and cannot set features, thresholds or checkpoints.",
        "teacher": {
            "source": "Post-replay objective 3D one-to-one TNG node match within the unchanged 4m cap",
            "positive": "Committed two-edge junction hypothesis assigned to one objective node",
            "negative": "Unmatched or duplicate committed two-edge junction hypothesis",
            "runtime_isolation": "Objective identity and center are never association or commit inputs",
        },
        "methods": {
            "main": "Old learned verifier OR unanimous three-seed <=4m metric support, followed by physical incidence counting. Terminal nodes remain unchanged; junction support 1/2/3/4 is audited.",
            "baseline": "Frozen old-verifier trace-commit graph and the parameter-free three-physical-edge junction rule.",
            "fallback": "If fit has fewer than 100/80/20 total/positive/negative two-edge junctions, fewer than 8 negative families or 5 negative strata, do not train.",
        },
        "acceptance": {
            "population": "Exact 188126 observations, 718/234 union committed hypotheses and 106/35 fit/selection two-edge junctions with 84/22 and 28/7 labels.",
            "coverage": "Fit negatives cover at least 8 topology families and 5 C-strata.",
            "capacity": "The no-training three-physical-edge rule reaches fit node/edge precision >=0.98 and recall >=0.25.",
            "isolation": "Zero optimizer/model inference/update and zero C09/C10/M-TARE reads.",
            "evidence": "Full fit/selection JSONL, rule CSV, summary, PNG/PDF/SVG, provenance, logs and seal.",
        },
        "leakage_audit": {"future_frames": 0, "selection_used_for_training": 0, "c09_worlds": 0, "c10_worlds": 0, "mtare_worlds": 0},
        "estimated_cost": {"compute": "CPU read-only", "wall_time_hours": 0.25, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "retention": "Retain compact hypothesis manifests, metrics, all figure formats, provenance, logs and seal.",
    }
    write_json(CARD, card)
    inputs = [
        teacher, pair_cache, objective, f"{current}/RUN_STATE.json", f"{current}/artifacts/evidence_sha256.txt",
        f"{current}/artifacts/replay/action_ensemble.npz", f"{current}/artifacts/replay/association_pairs.npz",
        f"{current}/artifacts/projection/spatial_center_ensemble_all_rows.npz",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "incidence_module": "src/mtare_topo/evaluation/gse_partial_incidence_reliability.py",
        "objective_scorer": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "executor": "tools/v3/execute_gse_partial_incidence_teacher_inventory_v1.py",
        "runner": "tools/v3/run_gse_partial_incidence_teacher_inventory_v1.py",
        "freezer": "tools/v3/freeze_gse_partial_incidence_teacher_inventory_spec_v1.py",
        "unit_test": "tests/v3/unit/test_gse_partial_incidence_reliability.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "operation": "audit", "date": "20260828",
        "slug": "gse_partial_incidence_teacher_inventory_v1", "seed": 0,
        "question": "Is fit-only evidence sufficient to learn reliability only for partially observed two-edge junctions?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "estimated_cost": card["estimated_cost"], "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": approval,
        "estimated_resources": {"wall_time_minutes": 15, "ram_gb": 4, "disk_gb": .1, "gpu_count": 0},
        "acceptance_criteria": list(card["acceptance"].values()),
        "frozen_inputs": {value: _sha(PROJECT_ROOT / value) for value in inputs},
        "frozen_tools": {name: {"path": value, "sha256": _sha(PROJECT_ROOT / value)} for name, value in tools.items()},
        "expected_evidence": ["Exact two-edge junction Teacher population and coverage.", "Incidence-rule capacity curves.", "Machine-readable stop/go decision for a later small reliability head."],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_partial_incidence_teacher_inventory_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
