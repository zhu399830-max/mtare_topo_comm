#!/usr/bin/env python3
"""Freeze the factorized association Teacher manifest Data Card and run spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_teacher_manifest_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_association_teacher_manifest_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_inventory_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
INVENTORY = "results/gate3_semantics/gate3_20260827_gse_factorized_association_inventory_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update({
        "card_id": "gse_factorized_association_teacher_manifest_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_TEACHER_MANIFEST_V1",
        "purpose": "Prove one split-isolated identity-balanced positive and same-event/degree structural-alias negative for every C01-C08 junction/terminal identity before association training.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-27T23:59:30+08:00", "authorized_operations": ["audit"],
        "authorized_gates": [3],
        "scope": "One immutable CPU-only C01-C08 read-only Teacher manifest proof; zero optimization, inference, C09/C10/strict/M-TARE or graph/planner access.",
        "confirmation_reference": "User explicitly instructed automatic evidence-supported decisions without routine approval prompts.",
    }
    card["source"] = {
        "raw_sources": [
            "Sealed corrected C01-C08 Teacher: 188126 causal observations.",
            "Sealed old pair cache used only for exact observation alignment and association-valid mask.",
            "Sealed failed factorized inventory proving old negative coverage defect.",
        ],
        "license_or_allowed_use": "Local research use of project procedural worlds and locally generated Teacher evidence.",
        "corrected_teacher_run": TEACHER, "frozen_component_run": VERIFIER,
        "inventory_defect_run": INVENTORY,
    }
    card["sampling"] = {
        "raw_frame_count": 252430, "effective_sample_count": 188126,
        "fit_observations": 142184, "selection_observations": 45942,
        "spatial_interval_m": 1.0,
        "independent_units": "1066 decision identities: fit 792, selection 274; junction 563, terminal 503.",
        "temporal_context": "Current plus up to four strictly earlier rows in the same directed traversal; never crosses reset.",
        "rule": "One query, one positive and one hard negative per identity. Positive hierarchy is different physical edge, reverse view, distinct observation, then fixed 180/720 circular shift for exactly five singleton full-history terminals. Negative is a different identity with same split/event/incident degree and nearest masked objective geometry profile.",
    }
    card["split"] = {
        "fit": "C01-C06 only: 60 worlds, 142184 observations, 792 decision identities.",
        "selection": "C07-C08 only: 20 worlds, 45942 observations, 274 decision identities.",
        "strict_test": "C09-C10 and M-TARE worlds are forbidden and read count must remain zero.",
        "leakage_audit": "No pair may cross fit/selection. Objective geometry chooses Teacher negatives only and is forbidden from student input. Identity/world/parent labels are never student features.",
    }
    card["teacher"] = {
        "source": "TNG decision identity/event/incident degree plus spline/mesh objective geometry for masked hard-negative selection.",
        "student_forbidden_fields": ["identity", "parent_id", "world_id", "event_teacher", "objective_geometry_profile"],
        "singleton_policy": "Five short terminal identities retain a deterministic circular-shift augmentation pair, reported separately and never counted as physical revisit evidence.",
    }
    card["methods"] = {
        "main": "Identity-balanced structural-alias pair manifest with split-isolated positives and nearest same-event/degree objective-geometry hard negatives.",
        "baseline": "Old runtime pair cache: fit 23359 positive/27 negative and selection 8083 positive/71 negative decision pairs.",
        "fallback": "If any identity lacks a valid pair or leakage occurs, stop association training and redesign the Teacher rather than delete identities or enlarge runtime radius.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 792 fit and 274 selection identities, 1066 manifest units and 2132 pair records.",
        "positive": "Every identity has one identity-exact positive; exactly five explicitly marked circular-shift singleton terminals and all other 1061 use physical cross-view/distinct observations.",
        "negative": "Every identity has one finite-distance different-identity negative with identical partition, event and incident degree.",
        "coverage": "All ten topology families occur in both partitions; limited five-history profiles remain exactly fit 15/selection 5 with masks.",
        "forbidden": "Optimizer, inference, C09, C10, strict-test and M-TARE counts all zero.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only deterministic Teacher manifest proof", "disk_gb": .25,
        "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": .25,
    }
    card["retention"] = "Retain complete JSONL/CSV manifest, metrics, PNG/PDF/SVG and source JSON, environment, raw log, RUN_STATE and SHA-256 seal for paper evidence."
    card["evidence"] = {
        "machine_metrics": "Exact identity/pair/kind/event/degree/family counts, hard-negative distance ranges, masks and zero forbidden operations.",
        "complete_visual_review": "Hard-negative distance and positive construction figure retained in raster/vector forms with source JSON.",
        "failure_policy": "Any source/alignment/population/pair/split/mask/finite-distance or singleton-count drift seals FAIL; no repair inside the run.",
    }
    card["failure_policy"] = card["evidence"]["failure_policy"]
    write_json(CARD, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt", f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{INVENTORY}/RUN_STATE.json", f"{INVENTORY}/metrics/summary.json",
        f"{INVENTORY}/artifacts/evidence_sha256.txt",
    ]
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "teacher_proposal": "docs/GSE_FACTORIZED_ASSOCIATION_TEACHER_PROPOSAL_V1.md",
        "inventory_contract": "src/mtare_topo/evaluation/gse_factorized_association_inventory.py",
        "teacher_contract": "src/mtare_topo/teacher/gse_factorized_association_teacher.py",
        "unit_tests": "tests/v3/unit/test_gse_factorized_association_teacher.py",
        "executor": "tools/v3/execute_gse_factorized_association_teacher_manifest_v1.py",
        "runner": "tools/v3/run_gse_factorized_association_teacher_manifest_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_association_teacher_manifest_spec_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_factorized_association_teacher_manifest_v1",
        "seed": 0, "operation": "audit",
        "question": "Can every C01-C08 decision identity receive one split-isolated positive and one same-event/degree objective-geometry-nearest hard negative without test leakage?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            "Exactly 1066 identity units (fit 792/selection 274) and 2132 pair records are emitted.",
            "Every positive is identity-exact; exactly five singleton terminals use marked circular-shift augmentation and the other 1061 use observed cross-view/distinct positives.",
            "Every negative is a different identity in the same partition/event/degree with finite objective-profile distance; all ten families occur in both partitions.",
            "Limited history masks equal fit 15/selection 5 and all optimizer/inference/C09/C10/strict/M-TARE counts are zero.",
        ],
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184, "fit_identities": 792,
            "selection_worlds": 20, "selection_observations": 45942,
            "selection_identities": 274, "manifest_units": 1066, "pair_records": 2132,
            "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0,
            "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Complete identity-balanced JSONL/CSV, exact metrics, PNG/PDF/SVG with source JSON, environment, command, raw log, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "960s", PYTHON,
            "tools/v3/run_gse_factorized_association_teacher_manifest_v1.py",
            "--spec", str(SPEC), "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
