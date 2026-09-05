"""Static governance tests for the approved Cano CPU raycast contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROPOSAL = ROOT / "configs/v3/gate0/cano_cpu_raycast_lidar_contract_v1.proposal.json"
CARD = ROOT / "configs/v3/gate0/data_cards/cano_cpu_raycast_lidar_contract_v1.json"
RUNNER = ROOT / "tools/v3/run_cano_cpu_raycast_lidar_contract.py"
EXECUTOR = ROOT / "tools/v3/execute_cano_cpu_raycast_lidar_contract.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_user_approval_is_bound_to_exact_zero_training_scope() -> None:
    proposal = _json(PROPOSAL)
    card = _json(CARD)
    assert proposal["approval"]["status"] == "APPROVED"
    assert card["approval"]["status"] == "APPROVED"
    assert card["approval"]["authorized_operations"] == ["sensor_smoke"]
    assert card["approval"]["authorized_gates"] == [0]
    assert proposal["scope"]["formal_dataset_samples"] == 0
    assert proposal["scope"]["training_samples"] == 0
    assert proposal["scope"]["models"] == 0


def test_card_freezes_exact_24_pose_and_16x720_contract() -> None:
    card = _json(CARD)
    assert card["scope"]["pose_count"] == 24
    assert card["scope"]["role_counts"] == {
        "tunnel_interior": 8,
        "junction_transition": 8,
        "terminal_approach": 8,
    }
    sensor = card["sampling"]["sensor_parameters"]
    assert len(sensor["elevation_deg"]) == 16
    assert sensor["azimuth_columns"] == 720
    assert sensor["near_range_m"] == 0.3
    assert sensor["far_range_m"] == 50.0


def test_executor_enforces_analytic_near_far_and_two_fresh_scenes() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")
    assert "def _analytic_control" in source
    assert "raw >= NEAR_RANGE_M" in source
    assert "raw <= MAX_RANGE_M" in source
    assert "scene_first = _build_scene(mesh)" in source
    assert "scene_second = _build_scene(mesh)" in source
    assert "replay_max_diff <= 1e-6" in source
    assert "reference_max_diff <= 1e-5" in source


def test_executor_preserves_clearance_branch_los_and_all_sample_visuals() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")
    assert "minimum_clearance >= 0.8" in source
    assert "target_distance - 0.25" in source
    assert "full_world_pose_map.png" in source
    assert "all_24_range_label_contact_sheet.png" in source
    assert "included_sample_ids" in source
    assert "Idealized CPU first-return raycasting does not establish Gazebo or real-LiDAR parity." in source


def test_runner_has_no_retry_and_freezes_sources_tools_environment() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("_run_executor(run_dir)") == 1
    assert "frozen_world_hashes" in source
    assert "prior_reference_manifest_sha256" in source
    assert "pip_freeze_evidence_sha256" in source
    assert '"formal_dataset_samples": 0' in source
    assert '"training_samples": 0' in source
    assert '"models": 0' in source
    assert '"isaac_runs": 0' in source
    assert '"gazebo_runs": 0' in source
    assert "docker" not in source.lower()


def test_runner_refuses_scope_drift_and_requires_all_evidence() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "spec.get(\"data_scope\") != expected_scope" in source
    assert "diagnostic_scan_npz" in source
    assert "scan_count == 24" in source
    assert "result_bytes <= int(0.05 * 1024**3)" in source
    assert "evidence_sha256.txt" in source
