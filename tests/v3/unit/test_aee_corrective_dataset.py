from __future__ import annotations

import numpy as np
import pytest
import importlib.util
from pathlib import Path

from mtare_topo.data.aee_corrective_dataset import (
    AEE_FRAMES_PER_TRAJECTORY,
    TOTAL_CORRECTIVE_FRAMES,
    aee_corrective_frame_indices,
    audit_corrective_sensor_shard,
    build_corrective_cano_cluster_eligibility,
    enumerate_aee_corrective_trajectories,
    monotonic_nearest_stamp_pairs,
    select_corrective_cano_clusters,
    select_qualified_corrective_cano_clusters,
)
from mtare_topo.data.aee_sensor_operator_parity import (
    resample_aee_gazebo_organized_azimuth,
)


ROOT = Path(__file__).resolve().parents[3]


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_aee_corrective_trajectory_v1",
        ROOT / "tools/v3/collect_aee_corrective_trajectory_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cano_v1r_executor():
    spec = importlib.util.spec_from_file_location(
        "execute_aee_corrective_cano_dataset_v1r",
        ROOT / "tools/v3/execute_aee_corrective_cano_dataset_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_v1r2_runner():
    spec = importlib.util.spec_from_file_location(
        "run_aee_corrective_sensor_teacher_dataset_v1r2",
        ROOT / "tools/v3/run_aee_corrective_sensor_teacher_dataset_v1r2.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(tunnel: int, index: int, *, junction: str | None = None, terminal: str | None = None):
    junctions = [] if junction is None else [junction]
    terminals = [] if terminal is None else [terminal]
    return {
        "cluster_id": f"t{tunnel}_k{index:03d}",
        "tunnel_id": tunnel,
        "center_arc_m": 2.5 + 5.0 * index,
        "junction_event_ids": junctions,
        "terminal_event_ids": terminals,
        "junction_event_distance_m": {} if junction is None else {junction: float(index)},
        "terminal_event_distance_m": {} if terminal is None else {terminal: float(index)},
        "primary_role": "terminal" if terminals else "junction" if junctions else "interior",
    }


def _qualification(cluster_id: str, failed_frames: tuple[int, ...] = ()):
    rows = []
    for frame_index in range(5):
        passed = frame_index not in failed_frames
        rows.append(
            {
                "frame_index": frame_index,
                "eligible": passed,
                "teacher_eligible": passed,
                "native_clearance_passed": True,
                "minimum_horizontal_clearance_m": 1.0,
                "branch_los": [passed],
            }
        )
    return build_corrective_cano_cluster_eligibility(cluster_id, rows)


def _valid_shard(frames: int = AEE_FRAMES_PER_TRAJECTORY):
    raw_range = np.full((frames, 16, 350), 2.0, dtype=np.float32)
    raw_valid = np.ones_like(raw_range, dtype=np.uint8)
    raw_range[:, 0, 10] = 130.0
    raw_valid[:, 0, 10] = 0
    source_valid = raw_valid.astype(bool) & (raw_range >= 0.3) & (raw_range <= 50.0)
    source_range = np.where(source_valid, raw_range, np.float32(50.0))
    model_range, model_valid = resample_aee_gazebo_organized_azimuth(source_range, source_valid)
    return {
        "raw_range_m": raw_range,
        "raw_valid_mask": raw_valid,
        "range_m": model_range,
        "valid_mask": model_valid,
        "sensor_xyz_m": np.zeros((frames, 3), dtype=np.float64),
        "sensor_orientation_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
        "yaw_deg": np.zeros(frames, dtype=np.float64),
        "frame_id": np.asarray([f"f{index:03d}" for index in range(frames)]),
    }


def test_registry_places_both_inspected_aee_worlds_in_corrective_train():
    trajectories = enumerate_aee_corrective_trajectories()
    assert len(trajectories) == 10
    assert [item.world for item in trajectories[:5]] == ["tunnel"] * 5
    assert [item.world for item in trajectories[5:]] == ["garage"] * 5
    assert all(item.split == "corrective_train" for item in trajectories)
    assert TOTAL_CORRECTIVE_FRAMES == 11_000


def test_aee_selection_is_exact_stride_30_and_outcome_independent():
    indices = aee_corrective_frame_indices()
    np.testing.assert_array_equal(indices, np.arange(0, 3000, 30))
    assert len(indices) == 100 and indices[-1] == 2970
    with pytest.raises(ValueError):
        aee_corrective_frame_indices(2999)


def test_monotonic_nearest_pairing_skips_raw_warmup_and_freezes_first_pairs():
    raw = [0, 100, 200, 300, 400, 500]
    registered = [198, 302, 399, 503]
    pairs = monotonic_nearest_stamp_pairs(
        raw, registered, maximum_delta_ns=10, required_pairs=4
    )
    assert [item.pair_index for item in pairs] == [0, 1, 2, 3]
    assert [item.raw_message_index for item in pairs] == [2, 3, 4, 5]
    assert [item.absolute_delta_ns for item in pairs] == [2, 2, 1, 3]


def test_monotonic_nearest_pairing_is_unique_ordered_and_tie_deterministic():
    pairs = monotonic_nearest_stamp_pairs(
        [90, 110, 190, 210], [100, 200], maximum_delta_ns=20, required_pairs=2
    )
    assert [item.raw_message_index for item in pairs] == [0, 2]
    assert all(right.raw_message_index > left.raw_message_index for left, right in zip(pairs, pairs[1:]))
    assert pairs == monotonic_nearest_stamp_pairs(
        [90, 110, 190, 210], [100, 200], maximum_delta_ns=20, required_pairs=2
    )


def test_monotonic_nearest_pairing_rejects_boundary_and_input_drift():
    with pytest.raises(RuntimeError, match="only 0"):
        monotonic_nearest_stamp_pairs([0], [10], maximum_delta_ns=10, required_pairs=1)
    with pytest.raises(ValueError, match="raw timestamps"):
        monotonic_nearest_stamp_pairs([0, 0], [0], maximum_delta_ns=10, required_pairs=1)
    with pytest.raises(ValueError, match="registered timestamps"):
        monotonic_nearest_stamp_pairs([0], [1, 0], maximum_delta_ns=10, required_pairs=1)


def test_cano_selector_covers_events_tunnels_and_replays_exactly():
    candidates = []
    for tunnel in range(4):
        for index in range(40):
            candidates.append(
                _candidate(
                    tunnel,
                    index,
                    junction="j0" if tunnel in (0, 1) and index == 1 else None,
                    terminal=f"e{tunnel}" if index == 39 else None,
                )
            )
    first, audit = select_corrective_cano_clusters(candidates)
    second, replay = select_corrective_cano_clusters(candidates)
    assert audit["passed"] and replay == audit
    assert [item["cluster_id"] for item in first] == [item["cluster_id"] for item in second]
    assert len(first) == 100 and audit["selected_frames"] == 500
    selected_ids = {item["cluster_id"] for item in first}
    assert "t0_k001" in selected_ids
    assert all(f"t{tunnel}_k039" in selected_ids for tunnel in range(4))


def test_cano_selector_rejects_mandatory_overflow_without_dropping_events():
    candidates = []
    for index in range(120):
        candidates.append(_candidate(0, index, terminal=f"e{index}"))
    for index in range(120):
        candidates.append(_candidate(1, index))
    with pytest.raises(RuntimeError, match="mandatory structural coverage"):
        select_corrective_cano_clusters(candidates)


def test_eligibility_first_rejects_whole_cluster_and_preserves_complete_lattice_event():
    candidates = [_candidate(0, index, junction="j0" if index in (1, 2) else None) for index in range(130)]
    qualifications = [
        _qualification(item["cluster_id"], (4,) if item["cluster_id"] == "t0_k001" else ())
        for item in candidates
    ]
    selected, audit = select_qualified_corrective_cano_clusters(candidates, qualifications)
    selected_ids = {item["cluster_id"] for item in selected}
    assert "t0_k001" not in selected_ids
    assert "t0_k002" in selected_ids
    assert len(selected) == 100 and audit["passed"]
    assert audit["candidate_lattice_clusters"] == 130
    assert audit["eligible_clusters"] == 129 and audit["ineligible_clusters"] == 1
    assert audit["ineligible_frames"] == 1
    assert audit["complete_lattice_events"] == 1
    assert audit["uncovered_complete_lattice_events"] == []
    assert audit["eligibility_uses_complete_development_geometry"]
    assert audit["structural_scoring_blind_to_observations"]
    assert "selection_blind_to_observations" not in audit
    assert "selection_blind_to_observations" not in audit["structural_selector_audit"]


def test_eligibility_first_fails_if_complete_lattice_event_has_no_safe_candidate():
    candidates = [_candidate(0, index, terminal="e0" if index == 3 else None) for index in range(130)]
    qualifications = [
        _qualification(item["cluster_id"], (0,) if item["cluster_id"] == "t0_k003" else ())
        for item in candidates
    ]
    with pytest.raises(RuntimeError, match="complete-lattice structural events"):
        select_qualified_corrective_cano_clusters(candidates, qualifications)


def test_eligibility_first_requires_exact_identity_and_is_order_deterministic():
    candidates = [_candidate(tunnel, index) for tunnel in range(2) for index in range(80)]
    qualifications = [_qualification(item["cluster_id"]) for item in candidates]
    first, first_audit = select_qualified_corrective_cano_clusters(candidates, qualifications)
    second, second_audit = select_qualified_corrective_cano_clusters(
        list(reversed(candidates)), list(reversed(qualifications))
    )
    assert [item["cluster_id"] for item in first] == [item["cluster_id"] for item in second]
    assert first_audit == second_audit
    with pytest.raises(ValueError, match="match the complete candidate lattice"):
        select_qualified_corrective_cano_clusters(candidates, qualifications[:-1])


def test_cluster_eligibility_is_exactly_five_atomic_frames():
    result = _qualification("cluster", (2,))
    assert not result.eligible and result.rejected_frame_count == 1
    assert result.to_dict()["frames"][2]["eligible"] is False
    with pytest.raises(ValueError, match="exactly 5"):
        build_corrective_cano_cluster_eligibility("cluster", result.to_dict()["frames"][:-1])


def test_corrective_shard_requires_raw_and_exact_resampled_views():
    payload = _valid_shard()
    audit = audit_corrective_sensor_shard(payload, 100)
    assert audit["passed"]
    assert audit["raw_to_model_resample_bitwise_exact"]

    missing = dict(payload)
    missing.pop("raw_range_m")
    assert not audit_corrective_sensor_shard(missing, 100)["passed"]

    drift = dict(payload)
    drift["range_m"] = payload["range_m"].copy()
    drift["range_m"][0, 0, 0] += 0.01
    result = audit_corrective_sensor_shard(drift, 100)
    assert not result["passed"] and not result["raw_to_model_resample_bitwise_exact"]


def test_corrective_shard_rejects_duplicate_identity_and_bad_raw_sentinel():
    payload = _valid_shard()
    payload["frame_id"] = np.asarray(["duplicate"] * 100)
    assert not audit_corrective_sensor_shard(payload, 100)["passed"]
    payload = _valid_shard()
    payload["raw_range_m"][0, 0, 10] = 50.0
    assert not audit_corrective_sensor_shard(payload, 100)["passed"]


def test_collector_decodes_frozen_organized_message_without_legacy_fill():
    collector = _load_collector()

    class Field:
        def __init__(self, name, offset, datatype):
            self.name, self.offset, self.datatype = name, offset, datatype

    class Message:
        width = 16
        height = 350
        point_step = 22
        row_step = 352
        is_bigendian = False
        is_dense = False
        fields = [Field("x", 0, 7), Field("y", 4, 7), Field("z", 8, 7), Field("ring", 16, 4)]
        data = bytes(350 * 16 * 22)

    rows = []
    for _azimuth in range(350):
        for ring in range(16):
            rows.append((2.0, 0.0, 0.0, ring))
    rows[10 * 16 + 3] = (np.nan, np.nan, np.nan, 3)

    class Decoder:
        @staticmethod
        def read_points(message, field_names, skip_nans):
            assert field_names == ("x", "y", "z", "ring") and skip_nans is False
            return iter(rows)

    raw_range, raw_valid, model_range, model_valid, audit = collector.decode_raw_message(Message(), Decoder)
    assert raw_range.shape == raw_valid.shape == (16, 350)
    assert model_range.shape == model_valid.shape == (16, 720)
    assert raw_valid[3, 10] == 0 and raw_range[3, 10] == 130.0
    assert audit["raw_records"] == 5600
    source = (ROOT / "tools/v3/collect_aee_corrective_trajectory_v1.py").read_text(encoding="utf-8")
    assert 'RAW_TOPIC = "/velodyne_points"' in source
    assert "organized_aee_gazebo_points_to_ranges" in source
    assert "resample_aee_gazebo_organized_azimuth" in source
    assert "fill_unmeasured_azimuth_gaps" not in source


def test_collector_rejects_unorganized_or_field_layout_drift():
    collector = _load_collector()

    class Field:
        def __init__(self, name, offset, datatype):
            self.name, self.offset, self.datatype = name, offset, datatype

    class Message:
        width = 5600
        height = 1
        point_step = 22
        row_step = 123200
        is_bigendian = False
        is_dense = True
        fields = [Field("x", 0, 7), Field("y", 4, 7), Field("z", 8, 7), Field("ring", 16, 4)]
        data = bytes(350 * 16 * 22)

    with pytest.raises(RuntimeError, match="organized layout"):
        collector.validate_raw_message_layout(Message())


def test_cano_executor_freezes_selection_before_rays_and_suppresses_validation_previews():
    source = (ROOT / "tools/v3/execute_aee_corrective_cano_dataset_v1.py").read_text(encoding="utf-8")
    select_at = source.index("selected, selector_audit = select_corrective_cano_clusters(candidates)")
    scene_at = source.index("scene_a = scene_from_mesh")
    assert select_at < scene_at
    assert "aee_gazebo_lidar_directions_sensor" in source
    assert '"raw_range_m"' in source and '"range_m"' in source
    assert '"validation_previews": 0' in source
    assert '"c09_reads": 0' in source and '"c10_reads": 0' in source
    assert "retry" not in source.lower()


def test_cano_v1r_creates_per_world_metric_directory_before_execution(tmp_path):
    executor = _load_cano_v1r_executor()
    metrics_root = executor.initialize_v1r_output_directories(tmp_path)
    assert metrics_root == tmp_path.resolve() / "metrics/cano"
    assert metrics_root.is_dir()
    metric = metrics_root / "world.json"
    metric.write_text("{}\n", encoding="utf-8")
    assert metric.read_text(encoding="utf-8") == "{}\n"


def test_cano_v1r2_orders_complete_eligibility_before_structural_selection_and_materialization():
    source = (ROOT / "tools/v3/execute_aee_corrective_cano_dataset_v1r2.py").read_text(encoding="utf-8")
    scene_at = source.index("scene_a = scene_from_mesh")
    qualification_at = source.index("qualifications = []")
    selection_at = source.index("selected, selector_audit = select_qualified_corrective_cano_clusters")
    shard_at = source.index("shard = create_shard")
    assert scene_at < qualification_at < selection_at < shard_at
    assert 'run / "metrics/cano"' in source
    assert 'run / "artifacts/cano_eligibility.jsonl"' in source
    assert '"candidate_clusters": 6_058' in source
    assert '"eligible_clusters": 6_043' in source
    assert '"ineligible_clusters": 15' in source
    assert '"ineligible_frames": 21' in source
    assert '"qualification_replay_exact"' in source
    assert "select_corrective_cano_clusters(candidates)" not in source
    assert "retry" not in source.lower()


def test_aee_teacher_uses_objective_complete_map_and_exact_100_frame_identity():
    source = (ROOT / "tools/v3/generate_aee_corrective_teacher_shard_v1.py").read_text(encoding="utf-8")
    assert "LayeredGTMapOracle.from_ply_and_dae" in source
    assert "audit_corrective_sensor_shard(sensor, AEE_FRAMES_PER_TRAJECTORY)" in source
    assert "aee_corrective_frame_indices()" in source
    assert "decode_direction_components" in source
    assert '"teacher_queries": len(direction)' in source
    assert '"c09_reads": 0' in source and '"c10_reads": 0' in source
    assert "torch" not in source and "checkpoint" not in source


def test_formal_runner_has_exact_11000_aggregation_and_no_retry():
    source = (ROOT / "tools/v3/run_aee_corrective_sensor_teacher_dataset_v1.py").read_text(encoding="utf-8")
    assert '"total_frames": TOTAL_CORRECTIVE_FRAMES' in source
    assert '"corrective_train_frames": 6_000' in source
    assert '"corrective_validation_frames": 5_000' in source
    assert '"teacher_labels": 11_000' in source
    assert "enumerate_aee_corrective_trajectories()" in source
    assert "verify_source_seal" in source
    assert "verify_archive" in source
    assert '"c09_reads": 0' in source and '"c10_reads": 0' in source
    assert "retry" not in source.lower()


def test_v1r2_runner_enforces_eligibility_aggregate_and_resource_contracts():
    source = (ROOT / "tools/v3/run_aee_corrective_sensor_teacher_dataset_v1r2.py").read_text(encoding="utf-8")
    assert '"candidate_clusters": 6_058' in source
    assert '"candidate_frames_qualified": 30_290' in source
    assert '"eligible_clusters": 6_043' in source
    assert '"ineligible_clusters": 15' in source
    assert '"ineligible_frames": 21' in source
    assert "run_monitored_process" in source
    assert "RSS_LIMIT_BYTES = 4 * 1024**3" in source
    assert '"--memory"' in source and '"--memory-swap"' in source
    assert "resource_traces" in source and "within_rss_limit" in source


def test_v1r3_pairing_does_not_version_the_stable_ownership_interface():
    source = (ROOT / "tools/v3/collect_aee_corrective_trajectory_v1r3.py").read_text(encoding="utf-8")
    assert '"schema_version": "aee_corrective_host_ownership_handoff_v1"' in source
    assert '"status": "PASS_AEE_CORRECTIVE_HOST_OWNERSHIP_HANDOFF_V1"' in source
    assert "HOST_OWNERSHIP_HANDOFF_V1R3" not in source


def test_v1r2_runner_applies_real_memory_contract_and_records_monitor_evidence(tmp_path):
    runner = _load_v1r2_runner()
    command = runner._add_docker_memory_contract(["docker", "run", "--rm", "image"])
    assert command == [
        "docker", "run", "--memory", str(4 * 1024**3), "--memory-swap", str(4 * 1024**3), "--rm", "image"
    ]
    logs = tmp_path / "logs"
    logs.mkdir()
    runner.run_logged(["/bin/true"], "resource-smoke", logs / "resource_smoke.log", 5.0)
    resource = (tmp_path / "metrics/resources/resource_smoke.json").read_text(encoding="utf-8")
    assert '"within_rss_limit": true' in resource
    assert (tmp_path / "metrics/resource_traces/resource_smoke.jsonl").is_file()


def test_corrective_handoff_is_scoped_to_new_evidence_tree():
    source = (ROOT / "tools/v3/collect_aee_corrective_trajectory_v1r.py").read_text(encoding="utf-8")
    assert 'Path("/evidence/aee_trajectories")' in source
    assert "follow_symlinks=False" in source
    assert "PASS_AEE_CORRECTIVE_HOST_OWNERSHIP_HANDOFF_V1" in source
