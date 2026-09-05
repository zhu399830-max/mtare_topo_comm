#!/usr/bin/env python3
"""Freeze the action-conditioned geometry-state feasibility card and spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_action_conditioned_state_feasibility_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_action_conditioned_state_feasibility_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_action_conditioned_state_feasibility_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_training_v1r.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
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
        "card_id": "gse_action_conditioned_state_feasibility_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_ACTION_CONDITIONED_STATE_FEASIBILITY_V1",
        "purpose": "Test whether frozen continuous geometry and exit/action state jointly create safer, more complete causal topology-node proposals than either source alone.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-27T23:55:00+08:00",
        "authorized_operations": ["training"], "authorized_gates": [3],
        "scope": "One immutable CPU-only C01-C08 frozen-feature feasibility proof; zero training, model inference, C09/C10/strict/M-TARE access or graph/planner tuning.",
        "confirmation_reference": "User granted standing authority to select and execute the evidence-supported best in-scope route without routine approval prompts.",
    }
    card["source"] = {
        "raw_sources": [
            "Three sealed C01-C08 frozen 146D deployment-feature arrays, 188126 observations each.",
            "Sealed corrected causal Teacher V1R, used only for episode scoring and never for state scores.",
            "Sealed C01-C06/C07-C08 partition cache; no raw LiDAR, checkpoint, C09/C10 or M-TARE artifact is read.",
        ],
        "license_or_allowed_use": "Local research use of project procedural Cano worlds and locally trained frozen outputs with full provenance.",
        "frozen_feature_run": VERIFIER,
        "corrected_teacher_run": TEACHER,
        "partial_reuse": "READ_ONLY_FROZEN_OBSERVATION_FEATURE_COMPONENTS_FROM_FAILED_ASSOCIATION_RUN; THIS DOES_NOT_REQUALIFY_THAT_RUN",
    }
    card["split"] = {
        "fit": "C01-C06: 60 worlds, 142184 observations; robust scales and one 99th-percentile corridor threshold per method.",
        "selection": "C07-C08: 20 disjoint worlds, 45942 observations; metrics only, no adaptation.",
        "strict_test": "C09/C10 and M-TARE remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
    }
    card["sampling"] = {
        "raw_frame_count": 252430,
        "effective_sample_count": 188126,
        "effective_causal_observations": 188126,
        "fit_observations": 142184,
        "selection_observations": 45942,
        "spatial_interval_m": 1.0,
        "independent_units": "80 programmatic topology parents and 16076 observed directed traversals; events are 5306 contiguous same-traversal Teacher episodes.",
        "temporal_context": "At each eligible observation, compare the means of the previous six and current six observations; first eleven rows of each traversal are unavailable.",
        "rule": "No resampling. Geometry uses columns 5:12; action uses 141:146; combined is equal-group RMS after fit-only robust scaling.",
        "event_episodes": {"junction": 3424, "terminal": 994, "turn": 740, "geometry_transition": 148},
        "fit_episodes": 3956,
        "selection_episodes": 1350,
    }
    card["teacher"] = {
        "source": "Sealed corrected causal Teacher V1R from TNG/spline/mesh.",
        "student_input": "No Teacher field enters a state score. Identity, event and episode membership are evaluation-only.",
        "episode_supervision": "A trigger matches only when its selected peak row lies inside one true structural episode; contiguous accepted responses collapse to one proposal.",
        "planner_consistency_plan": "If feasible, a stable state change proposes a provisional node; a graph edge still requires physical traversal.",
    }
    card["leakage_audit"] = {
        "world_isolation": "C01-C06 fit and C07-C08 selection are disjoint; C09/C10/M-TARE are absent.",
        "future_frames_excluded": True,
        "gt_identity_excluded_at_inference": True,
        "teacher_input_separation": "Teacher event/episode/identity are evaluation-only.",
        "normalization": "Per-feature robust center/scale from eligible C01-C06 deltas only.",
        "test_excluded_from_supervised_training": True,
        "test_excluded_from_threshold_calibration": True,
        "test_excluded_from_normalization": True,
        "test_excluded_from_checkpoint_selection": True,
        "test_excluded_from_ssl": True,
        "test_excluded_from_teacher_calibration": True,
        "test_excluded_from_augmentation_tuning": True,
    }
    card["methods"] = {
        "main": "Equal-group combination of causal six-versus-six geometry and exit/action discrepancies, with a fit-only higher 99th-percentile corridor threshold.",
        "baseline": "The same detector using geometry alone and exit/action summary alone under independently fitted but identical one-percent corridor budgets.",
        "fallback": "If combined state does not improve safe episode recall or misses turn/transition identities, stop or redesign this route before any new training.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "runtime_trigger": "Selection precision >=0.98 and false triggers per eligible corridor observation <=0.01.",
        "episode": "Combined selection episode recall exceeds the stronger single-source baseline by >=0.05.",
        "identity": "Combined covers at least one turn identity and one geometry-transition identity.",
        "reporting": "Report all three methods, all four event identity coverages and both fit/selection results; no post-hoc winner selection.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only frozen-feature analytic proof", "disk_gb": .25,
        "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": .25,
    }
    card["retention"] = "Retain metrics, all trigger rows, robust scales, paper-ready PNG/PDF/SVG, logs, source integrity, RUN_STATE and seal. No cache or checkpoint is created."
    card["evidence"] = {
        "machine_metrics": "Exact populations, thresholds, precision, false-trigger fraction, episode recall, event identity coverage, zero inference/optimization/forbidden reads and complete seal.",
        "complete_visual_review": "One paper-ready three-panel method comparison retained as PNG/PDF/SVG with source JSON.",
        "failure_policy": "Any source/split/Teacher/alignment drift, future crossing, forbidden read, nonfinite score, resource violation or unmet scientific criterion seals FAIL without adaptation.",
    }
    card["failure_policy"] = card["evidence"]["failure_policy"]
    write_json(CARD, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt", f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.append(f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy")
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_review": "docs/GSE_ACTION_CONDITIONED_GEOMETRY_STATE_REVIEW_V1.md",
        "state_evaluator": "src/mtare_topo/evaluation/gse_action_conditioned_state.py",
        "episode_reference": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "executor": "tools/v3/execute_gse_action_conditioned_state_feasibility_v1.py",
        "runner": "tools/v3/run_gse_action_conditioned_state_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_action_conditioned_state_feasibility_spec_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_action_conditioned_state_feasibility_v1", "seed": 0,
        "operation": "audit",
        "question": "Can frozen continuous geometry and exit/action state jointly create safer, more complete structural node proposals than either source alone?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            "Exact C01-C06 60/142184 and C07-C08 20/45942 populations, 3956/1350 episodes and selection identities 146/128/95/17.",
            "Combined selection precision >=0.98, false triggers per eligible corridor observation <=0.01, and episode recall gain >=0.05 over the stronger single-source baseline.",
            "Combined covers at least one turn and one geometry-transition identity; zero inference/optimizer/C09/C10/strict/M-TARE reads and complete paper-ready evidence seal.",
        ],
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184, "fit_episodes": 3956,
            "selection_worlds": 20, "selection_observations": 45942, "selection_episodes": 1350,
            "model_inference_frames": 0, "optimizer_steps": 0, "c09_worlds_read": 0,
            "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Fit-only robust scales and thresholds, all three method metrics, all trigger rows, paper-ready PNG/PDF/SVG with source JSON, environment, log, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "CPU-only frozen-feature analytic proof", "disk_gb": .25, "wall_time_hours": .25},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "960s", PYTHON,
            "tools/v3/run_gse_action_conditioned_state_feasibility_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
