from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.governance import (
    create_run,
    preflight,
    update_status_fields,
    validate_data_card,
    validate_sensor_contract_pilot_card,
    validate_sensor_smoke_card,
)


def make_status(gate: int = 0) -> dict:
    return {
        "schema_version": "master_plan_v3_status_v1",
        "current_gate": gate,
        "running_experiment": None,
        "gate_status": "GATE_MIXED",
        "advance_authorized": False,
    }


def make_spec(gate: int = 0, operation: str = "infrastructure") -> dict:
    return {
        "schema_version": "v3_run_spec_v1",
        "gate": gate,
        "date": "20260808",
        "slug": "governance_contract_test",
        "operation": operation,
        "question": "Does governance reject an unsafe run?",
        "method": "Deterministic contract validation",
        "baseline": "No preflight validation",
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "test_fixture",
            "approved_at": "2026-08-08",
            "scope": "Governance unit test only",
            "confirmation_reference": "unit-test fixture",
        },
        "seed": 0,
        "command": ["python3", "-c", "print('not executed by create_run')"],
        "acceptance_criteria": ["Expected validation outcome is observed"],
        "expected_evidence": ["Validation report"],
        "estimated_cost": {
            "disk_gb": 0,
            "wall_time_hours": 0,
            "compute": "CPU-only validation",
        },
    }


def make_approved_data_card() -> dict:
    return {
        "schema_version": "v3_data_card_v1",
        "card_id": "gate1_example_approved",
        "purpose": "Contract-test-only synthetic metadata; no real data is exported.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "test_fixture",
            "approved_at": "2026-08-08",
            "scope": "Synthetic metadata used only by the governance unit test",
            "authorized_operations": ["training"],
            "authorized_gates": [2],
            "confirmation_reference": "unit-test fixture, not a real user approval",
        },
        "source": {
            "raw_sources": ["synthetic_contract_fixture"],
            "license_or_allowed_use": "Local test fixture only",
        },
        "worlds": {
            "train": ["mine_dev_a"],
            "validation": ["tunnel_dev_b"],
            "ssl": [],
            "normalization": ["mine_dev_a"],
            "teacher_calibration": ["mine_dev_a"],
            "threshold_calibration": ["tunnel_dev_b"],
            "augmentation_tuning": ["mine_dev_a"],
            "checkpoint_selection": ["tunnel_dev_b"],
            "strict_test": ["cave_test_a"],
        },
        "trajectories": [
            {
                "id": "mine_dev_a_traj01",
                "world": "mine_dev_a",
                "split": "train",
                "independent": True,
                "duration_s": 120.0,
                "distance_m": 80.0,
                "spatial_coverage_m": 65.0,
            },
            {
                "id": "tunnel_dev_b_traj01",
                "world": "tunnel_dev_b",
                "split": "validation",
                "independent": True,
                "duration_s": 90.0,
                "distance_m": 55.0,
                "spatial_coverage_m": 48.0,
            },
        ],
        "sampling": {
            "raw_frame_count": 1000,
            "effective_sample_count": 200,
            "effective_structure_event_count": 20,
            "spatial_interval_m": 0.5,
            "rule": "Select causally valid frames separated by at least 0.5 m.",
            "structure_event_counts": {
                "junction": 8,
                "dead_end": 4,
                "width_change": 8,
            },
        },
        "teacher": {
            "source": "Synthetic planner-consistent fixture",
            "valid_mask": "Only directions evaluated by the fixture planner are valid",
            "planner_consistency_plan": "Compare every fixture label with planner output",
        },
        "split": {
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": "Fixture names do not refer to project data",
        },
        "leakage_audit": {
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "estimated_cost": {
            "disk_gb": 1.0,
            "wall_time_hours": 2.0,
            "compute": "CPU export and one GPU for later training",
        },
    }


def make_topology_replay_card() -> dict:
    card = make_approved_data_card()
    card["approval"]["authorized_operations"] = ["topology_replay"]
    card["approval"]["authorized_gates"] = [4]
    return card


def make_diagnostic_geometry_audit_card() -> dict:
    card = make_topology_replay_card()
    card["trajectories"] = [
        {
            "id": "mine_dev_a_failure_subset",
            "world": "mine_dev_a",
            "split": "train",
            "independent": False,
            "duration_s": 120.0,
            "distance_m": 80.0,
            "spatial_coverage_m": 65.0,
        }
    ]
    card["sampling"].update(
        {
            "raw_frame_count": 9,
            "effective_sample_count": 9,
            "effective_structure_event_count": 0,
            "parameter_evaluation_count": 549,
            "structure_event_counts": {"junction": 0, "terminal": 0},
        }
    )
    card["diagnostic_geometry_audit"] = {
        "enabled": True,
        "formal_dataset_false": True,
        "source_trajectory_independent": True,
        "pose_subset_nonindependent": True,
        "parameter_candidates_not_samples": True,
        "parameter_evaluation_count": 549,
        "zero_training": True,
        "zero_model_selection": True,
    }
    return card


def write_card(project_root: Path, card: dict) -> str:
    card_path = project_root / "cards" / "approved.json"
    card_path.parent.mkdir(parents=True)
    card_path.write_text(json.dumps(card), encoding="utf-8")
    return str(card_path.relative_to(project_root))


def make_approved_sensor_smoke_card() -> dict:
    return {
        "schema_version": "v3_sensor_smoke_card_v1",
        "card_id": "gate0_sensor_smoke_fixture",
        "purpose": "One-world diagnostic sensor fixture; creates no formal dataset.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "test_fixture",
            "approved_at": "2026-08-10",
            "scope": "Exactly one world and four diagnostic poses",
            "authorized_operations": ["sensor_smoke"],
            "authorized_gates": [0],
            "confirmation_reference": "unit-test fixture",
        },
        "source": {
            "raw_sources": ["existing_world_fixture"],
            "license_or_allowed_use": "Local diagnostic fixture only",
            "world_id": "world_fixture_000",
        },
        "scope": {
            "formal_dataset": False,
            "split": "NONE_DIAGNOSTIC_SMOKE_ONLY",
            "world_count": 1,
            "pose_count": 4,
            "training_samples": 0,
            "validation_samples": 0,
            "test_samples": 0,
            "trajectories": 0,
            "no_training": True,
            "no_model_selection": True,
            "no_threshold_calibration": True,
            "benchmark_worlds_excluded": True,
        },
        "sampling": {
            "planned_pose_count": 4,
            "role_counts": {"tunnel": 2, "junction": 1, "terminal": 1},
            "selection_rule": "Deterministic role-stratified poses",
            "sensor_contract": "Fixed synthetic lidar contract",
            "label_contract": "Ground-truth axis direction diagnostic",
        },
        "evidence": {
            "machine_metrics": "Per-pose validity and schema metrics",
            "complete_visual_review": "All four poses are rendered without cherry-picking",
            "failure_policy": "Stop without substituting a world or backend",
        },
        "estimated_cost": {
            "disk_gb": 0.1,
            "wall_time_hours": 0.1,
            "compute": "Short CPU/GPU diagnostic",
        },
    }


def make_approved_sensor_contract_pilot_card() -> dict:
    return {
        "schema_version": "v3_sensor_contract_pilot_card_v1",
        "card_id": "gate0_sensor_contract_pilot_fixture",
        "purpose": "Two-world zero-training sensor contract fixture.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "test_fixture",
            "approved_at": "2026-08-10",
            "scope": "Two parents, four anchors and eight observations",
            "authorized_operations": ["sensor_contract_pilot"],
            "authorized_gates": [0],
            "confirmation_reference": "unit-test fixture",
        },
        "source": {
            "raw_sources": ["procedural_fixture"],
            "license_or_allowed_use": "Local contract fixture only",
            "benchmark_exclusion": "No benchmark world is present",
        },
        "scope": {
            "formal_dataset": False,
            "split": "NONE_DIAGNOSTIC_CONTRACT_PILOT_ONLY",
            "topology_parent_count": 2,
            "mesh_world_count": 2,
            "canonical_anchor_count": 4,
            "diagnostic_observation_count": 8,
            "training_samples": 0,
            "validation_samples": 0,
            "test_samples": 0,
            "trajectories": 0,
            "models": 0,
            "no_training": True,
            "no_model_selection": True,
            "no_threshold_calibration": True,
            "no_ai_labels": True,
            "benchmark_worlds_excluded": True,
        },
        "generation": {
            "parent_ids": ["parent_a", "parent_b"],
            "topology_seeds": [1, 2],
            "geometry_seeds": [101, 102],
            "retry_policy": "No retry or replacement",
            "variant_policy": "One perception mesh per parent",
        },
        "sampling": {
            "anchors_per_parent": 2,
            "views_per_anchor": 2,
            "view_yaw_offsets_deg": [-15.0, 15.0],
            "rays_per_observation": 11520,
            "sensor_contract": "Fixed range image",
            "label_contract": "Objective spline labels",
            "storage_contract": "One shard per world",
        },
        "evidence": {
            "machine_metrics": "All parent and observation checks",
            "complete_visual_review": "All observations represented",
            "failure_policy": "Stop without replacing a parent",
        },
        "estimated_cost": {
            "disk_gb": 0.1,
            "wall_time_hours": 0.1,
            "compute": "CPU contract fixture",
        },
    }


class GovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_valid_gate0_infrastructure_preflight_passes(self) -> None:
        report = preflight(make_spec(), make_status(), self.project_root)

        self.assertTrue(report.passed)
        self.assertEqual(report.errors, ())
        self.assertEqual(
            report.output_dir,
            self.project_root
            / "results"
            / "gate0_baseline"
            / "gate0_20260808_governance_contract_test_seed0",
        )

    def test_cross_gate_run_is_rejected(self) -> None:
        report = preflight(make_spec(gate=0), make_status(gate=1), self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(
            any("does not match current Gate" in error for error in report.errors)
        )

    def test_run_without_user_authorization_is_rejected(self) -> None:
        spec = make_spec()
        spec["user_authorization"]["status"] = "DRAFT"

        report = preflight(spec, make_status(), self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(
            any("user_authorization.status" in error for error in report.errors)
        )

    def test_training_without_approved_data_card_is_rejected(self) -> None:
        report = preflight(
            make_spec(gate=2, operation="training"),
            make_status(gate=2),
            self.project_root,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("data_card" in error for error in report.errors))

    def test_approved_corrective_data_export_passes_at_gate2(self) -> None:
        card = make_approved_data_card()
        card["approval"]["authorized_operations"] = ["data_export"]
        card["approval"]["authorized_gates"] = [2]
        spec = make_spec(gate=2, operation="data_export")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_data_card(card)
        run_report = preflight(spec, make_status(gate=2), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_approved_semantic_teacher_data_export_passes_at_gate3(self) -> None:
        card = make_approved_data_card()
        card["approval"]["authorized_operations"] = ["data_export"]
        card["approval"]["authorized_gates"] = [3]
        spec = make_spec(gate=3, operation="data_export")
        spec["data_card"] = write_card(self.project_root, card)

        report = preflight(spec, make_status(gate=3), self.project_root)

        self.assertTrue(report.passed, report.errors)

    def test_topology_replay_without_approved_data_card_is_rejected(self) -> None:
        report = preflight(
            make_spec(gate=4, operation="topology_replay"),
            make_status(gate=4),
            self.project_root,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("data_card" in error for error in report.errors))

    def test_approved_topology_replay_data_card_passes(self) -> None:
        card = make_topology_replay_card()
        spec = make_spec(gate=4, operation="topology_replay")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_data_card(card)
        run_report = preflight(spec, make_status(gate=4), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_approved_shadow_data_card_passes_at_gate5(self) -> None:
        card = make_topology_replay_card()
        card["approval"]["authorized_operations"] = ["shadow"]
        card["approval"]["authorized_gates"] = [5]
        spec = make_spec(gate=5, operation="shadow")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_data_card(card)
        run_report = preflight(spec, make_status(gate=5), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_diagnostic_geometry_audit_allows_honest_nonindependent_subset(self) -> None:
        card = make_diagnostic_geometry_audit_card()
        spec = make_spec(gate=4, operation="topology_replay")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_data_card(card)
        run_report = preflight(spec, make_status(gate=4), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertFalse(card_report.warnings, card_report.warnings)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_nonindependent_subset_without_diagnostic_contract_is_rejected(self) -> None:
        card = make_diagnostic_geometry_audit_card()
        card.pop("diagnostic_geometry_audit")

        report = validate_data_card(card)

        self.assertFalse(report.passed)
        self.assertTrue(any("independent must be true" in error for error in report.errors))

    def test_diagnostic_parameter_candidates_cannot_be_claimed_as_samples(self) -> None:
        card = make_diagnostic_geometry_audit_card()
        card["sampling"]["effective_sample_count"] = 549

        report = validate_data_card(card)

        self.assertFalse(report.passed)
        self.assertTrue(any("parameter candidates are not samples" in error for error in report.errors))

    def test_ai_annotation_without_approved_data_card_is_rejected(self) -> None:
        report = preflight(
            make_spec(gate=1, operation="ai_annotation"),
            make_status(gate=1),
            self.project_root,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("data_card" in error for error in report.errors))

    def test_sensor_smoke_without_approved_smoke_card_is_rejected(self) -> None:
        report = preflight(
            make_spec(gate=0, operation="sensor_smoke"),
            make_status(gate=0),
            self.project_root,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("data_card" in error for error in report.errors))

    def test_approved_zero_dataset_sensor_smoke_card_passes(self) -> None:
        card = make_approved_sensor_smoke_card()
        spec = make_spec(gate=0, operation="sensor_smoke")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_sensor_smoke_card(card)
        run_report = preflight(spec, make_status(gate=0), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_approved_multi_world_sensor_contract_pilot_passes(self) -> None:
        card = make_approved_sensor_contract_pilot_card()
        spec = make_spec(gate=0, operation="sensor_contract_pilot")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_sensor_contract_pilot_card(card)
        run_report = preflight(spec, make_status(gate=0), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_sensor_contract_pilot_rejects_count_drift(self) -> None:
        card = make_approved_sensor_contract_pilot_card()
        card["scope"]["diagnostic_observation_count"] = 7

        report = validate_sensor_contract_pilot_card(card)

        self.assertFalse(report.passed)
        self.assertTrue(any("parents * anchors * views" in error for error in report.errors))

    def test_ai_annotation_card_without_annotation_contract_is_rejected(self) -> None:
        card = make_approved_data_card()
        card["approval"]["authorized_operations"] = ["ai_annotation"]
        card["approval"]["authorized_gates"] = [1]
        spec = make_spec(gate=1, operation="ai_annotation")
        spec["data_card"] = write_card(self.project_root, card)

        report = preflight(spec, make_status(gate=1), self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(any("annotation must be an object" in error for error in report.errors))

    def test_approved_ai_annotation_contract_passes(self) -> None:
        card = make_approved_data_card()
        card["approval"]["authorized_operations"] = ["ai_annotation"]
        card["approval"]["authorized_gates"] = [1]
        card["annotation"] = {
            "labeler_name": "frozen_test_labeler",
            "labeler_version": "v1",
            "prompt_reference": "prompts/structural_role_v1.txt",
            "input_bundle_contract": "annotation_bundle_v1",
            "output_schema": "structural_attributes_v1",
            "conflict_policy": "Send objective-label conflicts to human review",
            "abstain_policy": "Exclude abstained fields from supervision",
            "planned_ai_sample_count": 300,
            "planned_human_gold_count": 150,
            "strict_test_excluded": True,
            "raw_responses_preserved": True,
        }
        spec = make_spec(gate=1, operation="ai_annotation")
        spec["data_card"] = write_card(self.project_root, card)

        report = preflight(spec, make_status(gate=1), self.project_root)

        self.assertTrue(report.passed, report.errors)

    def test_draft_data_card_is_rejected(self) -> None:
        card = make_approved_data_card()
        card["approval"]["status"] = "DRAFT"
        spec = make_spec(gate=2, operation="training")
        spec["data_card"] = write_card(self.project_root, card)

        report = preflight(spec, make_status(gate=2), self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(any("approval.status" in error for error in report.errors))

    def test_strict_test_leakage_is_rejected(self) -> None:
        card = make_approved_data_card()
        card["worlds"]["ssl"] = ["cave_test_a"]
        spec = make_spec(gate=2, operation="training")
        spec["data_card"] = write_card(self.project_root, card)

        report = preflight(spec, make_status(gate=2), self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(any("strict test leakage" in error for error in report.errors))

    def test_approved_world_disjoint_data_card_passes(self) -> None:
        card = make_approved_data_card()
        spec = make_spec(gate=2, operation="training")
        spec["data_card"] = write_card(self.project_root, card)

        card_report = validate_data_card(card)
        run_report = preflight(spec, make_status(gate=2), self.project_root)

        self.assertTrue(card_report.passed, card_report.errors)
        self.assertTrue(run_report.passed, run_report.errors)

    def test_create_run_writes_layout_but_does_not_execute(self) -> None:
        spec = make_spec()
        sentinel = self.project_root / "command_was_executed"
        spec["command"] = [
            "python3",
            "-c",
            f"from pathlib import Path; Path({str(sentinel)!r}).write_text('bad')",
        ]

        output_dir = create_run(spec, make_status(), self.project_root)

        self.assertFalse(sentinel.exists())
        for child in ("config", "logs", "metrics", "previews", "artifacts"):
            self.assertTrue((output_dir / child).is_dir())
        state = json.loads(
            (output_dir / "RUN_STATE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["state"], "CREATED_NOT_EXECUTED")

        second_report = preflight(spec, make_status(), self.project_root)
        self.assertFalse(second_report.passed)
        self.assertTrue(
            any("refusing to overwrite" in error for error in second_report.errors)
        )

    def test_status_update_cannot_change_gate(self) -> None:
        status = make_status(gate=0)

        updated = update_status_fields(
            status,
            latest_result="Governance tests passed",
            next_action="Draft Gate-0 interface contract",
        )

        self.assertEqual(updated["current_gate"], 0)
        self.assertFalse(updated["advance_authorized"])
        self.assertEqual(updated["latest_result"], "Governance tests passed")

    def test_existing_running_experiment_blocks_new_run(self) -> None:
        status = make_status(gate=0)
        status["running_experiment"] = "gate0_another_run"

        report = preflight(make_spec(), status, self.project_root)

        self.assertFalse(report.passed)
        self.assertTrue(any("marked running" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
