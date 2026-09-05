from pathlib import Path
import ast
import sys


ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "tools/v3/semantic_topology_global_node_v3.py"


def source():
    return NODE.read_text(encoding="utf-8")


def test_ros_node_is_syntax_valid_and_has_exact_sync():
    ast.parse(source())
    text = source()
    assert "message_filters.TimeSynchronizer" in text
    assert '"/registered_scan"' in text
    assert '"/state_estimation_at_scan"' in text


def test_ros_node_uses_full_orientation_and_frozen_runtime():
    text = source()
    assert "sensor_orientation_xyzw=orientation" in text
    assert "OnlineTopologyPlannerRuntime(GRAPH_CONFIG, PLANNER_CONFIG)" in text
    assert "evaluator_gt_edge_id" not in text


def test_ros_node_replaces_only_public_global_topics():
    text = source()
    for topic in (
        '"/way_point"',
        '"/runtime"',
        '"/map_clearing"',
        '"/sensor_coverage_planner/exploration_finish"',
        '"/free_paths"',
    ):
        assert topic in text
    for forbidden in ("/cmd_vel", "/terrain_map", "/path"):
        assert f'Publisher("{forbidden}"' not in text


def test_ros_node_requires_deployment_checkpoint_identity_and_hash():
    text = source()
    assert "m1d_ros_deployment_state_v1" in text
    assert "deployment checkpoint hash drift" in text
    assert 'payload.get("mode") not in SUPPORTED_DEPLOYMENT_MODES' in text
    assert 'SUPPORTED_DEPLOYMENT_MODES = ("M1D", "M1D_AEE_HEAD_ADAPTED_V1", COMPOSITE_V9_MODE)' in text


def test_ros_node_applies_v9_composite_contract() -> None:
    text = source()
    assert "apply_composite_v9_semantics" in text
    assert 'record["composite_semantics"]' in text
    assert "V9 deployment runtime contract mismatch" in text


def test_v9_uses_exact_raw_organized_sensor_operator() -> None:
    text = source()
    assert 'Subscriber("/velodyne_points"' in text
    assert "message_filters.ApproximateTimeSynchronizer" in text
    assert "MAXIMUM_RAW_POSE_STAMP_DELTA_SEC = 0.1" in text
    assert "aee_organized_pointcloud2_to_range_image" in text
    assert 'self.input_operator = "aee_organized_velodyne_16x350_to_16x720_v1"' in text
