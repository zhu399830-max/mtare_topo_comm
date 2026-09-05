from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from mtare_topo.evaluation.closed_loop_recording import (
    TopicObservation,
    audit_topic_inventory,
    load_topic_contract,
    rosbag_record_command,
)
from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_cases,
    load_development_matrix,
    paired_case_audit,
)


ROOT = Path(__file__).resolve().parents[3]
LAUNCH = ROOT / "configs/v3/gate5/roslaunch"


def _load_oracle_node():
    path = ROOT / "tools/v3/layered_gt_map_global_node_v1.py"
    spec = importlib.util.spec_from_file_location("layered_gt_map_global_node_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeded_vehicle_launch_injects_exact_gazebo_seed() -> None:
    root = ET.parse(LAUNCH / "vehicle_simulator_seeded.launch").getroot()
    arguments = {item.attrib["name"]: item.attrib for item in root.findall("arg")}
    assert arguments["gazebo_seed"]["default"] == "0"
    includes = root.findall("include")
    gazebo = next(item for item in includes if "gazebo_ros" in item.attrib["file"])
    passed = {item.attrib["name"]: item.attrib["value"] for item in gazebo.findall("arg")}
    assert passed["extra_gazebo_args"] == "--seed $(arg gazebo_seed)"


def test_seeded_system_keeps_local_stack_and_excludes_tare_planner() -> None:
    path = LAUNCH / "system_seeded.launch"
    root = ET.parse(path).getroot()
    executable_contract = " ".join(
        value for element in root.iter() for value in element.attrib.values()
    )
    assert "tare_planner" not in executable_contract
    includes = [item.attrib["file"] for item in root.findall("include")]
    assert any("local_planner" in value for value in includes)
    assert any("terrain_analysis" in value for value in includes)
    assert any("sensor_scan_generation" in value for value in includes)
    seeded = next(item for item in root.findall("include") if "vehicle_simulator_seeded" in item.attrib["file"])
    passed = {item.attrib["name"]: item.attrib["value"] for item in seeded.findall("arg")}
    assert passed["gazebo_seed"] == "$(arg gazebo_seed)"


def test_oracle_prediction_adapter_preserves_all_four_heads() -> None:
    module = _load_oracle_node()

    class Prediction:
        direction_logits = np.arange(720, dtype=np.float32)
        count_probabilities = np.arange(6, dtype=np.float32) / 15.0
        role_probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
        z_role = np.arange(128, dtype=np.float32)

    adapted = module.oracle_semantic_prediction(Prediction())
    assert np.array_equal(adapted.direction_logits, Prediction.direction_logits)
    assert np.array_equal(adapted.count_probabilities, Prediction.count_probabilities)
    assert np.array_equal(adapted.role_probabilities, Prediction.role_probabilities)
    assert np.array_equal(adapted.z_role, Prediction.z_role)


def test_oracle_and_m1d_nodes_share_public_topics_and_completion_policy() -> None:
    oracle = (ROOT / "tools/v3/layered_gt_map_global_node_v1.py").read_text(encoding="utf-8")
    m1d = (ROOT / "tools/v3/semantic_topology_global_node_v3.py").read_text(encoding="utf-8")
    for topic in (
        "/way_point",
        "/runtime",
        "/sensor_coverage_planner/exploration_finish",
        "/map_clearing",
        "/free_paths",
    ):
        assert topic in oracle
        assert topic in m1d
    assert 'candidate_complete if self.shadow else False' in oracle
    assert 'candidate_complete if self.shadow else False' in m1d
    assert 'open("x", encoding="utf-8"' in oracle


def test_recording_contract_is_identical_and_lossless_for_all_methods(tmp_path: Path) -> None:
    contract = load_topic_contract(ROOT / "configs/v3/gate5/closed_loop_recording_topics_v1.json")
    command = rosbag_record_command(contract, output_bag=tmp_path / "run.bag")
    assert command[:6] == ("rosbag", "record", "--lz4", "--buffsize=2048", "-O", str(tmp_path / "run.bag"))
    assert command[6:] == tuple(topic.name for topic in contract)
    assert "/sensor_coverage_planner/exploration_finish" in command


def test_aee_recording_contract_adds_raw_model_input() -> None:
    contract = load_topic_contract(ROOT / "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json")
    raw = next(item for item in contract if item.name == "/velodyne_points")
    assert raw.required and raw.type == "sensor_msgs/PointCloud2"


def test_recording_inventory_fails_missing_empty_or_wrong_type() -> None:
    contract = load_topic_contract(ROOT / "configs/v3/gate5/closed_loop_recording_topics_v1.json")
    observed = {topic.name: TopicObservation(topic.type, 1) for topic in contract if topic.required}
    assert audit_topic_inventory(contract, observed).passed
    observed.pop("/cmd_vel")
    observed["/path"] = TopicObservation("sensor_msgs/PointCloud2", 1)
    observed["/runtime"] = TopicObservation("std_msgs/Float32", 0)
    audit = audit_topic_inventory(contract, observed)
    assert not audit.passed
    assert audit.missing_topics == ("/cmd_vel",)
    assert audit.empty_required_topics == ("/runtime",)
    assert audit.type_mismatches == ("/path:sensor_msgs/PointCloud2!=nav_msgs/Path",)


def test_development_matrix_is_exactly_paired_without_test_worlds() -> None:
    matrix = load_development_matrix(
        ROOT / "configs/v3/gate5/mtare_single_robot_development_matrix_v1.proposal.json"
    )
    cases = enumerate_cases(matrix)
    audit = paired_case_audit(cases)
    assert audit == {
        "schema_version": "mtare_single_robot_pairing_audit_v1",
        "case_count": 50,
        "paired_world_seed_count": 10,
        "methods_per_pair": 5,
        "all_pairs_complete_and_ordered": True,
        "total_simulated_runtime_sec": 30000,
    }
    assert {case.world for case in cases} == {"tunnel", "garage"}
    assert not any("c09" in case.case_id.lower() or "c10" in case.case_id.lower() for case in cases)
    assert [case.method_id for case in cases[:5]] == [
        "original_mtare",
        "m1d_seed0",
        "m1d_seed1",
        "m1d_seed2",
        "layered_gt_map_oracle",
    ]


def test_development_matrix_keeps_environment_and_model_seeds_separate() -> None:
    matrix = load_development_matrix(
        ROOT / "configs/v3/gate5/mtare_single_robot_development_matrix_v1.proposal.json"
    )
    cases = enumerate_cases(matrix)
    m1d = [case for case in cases if case.method_family == "m1d_topology"]
    assert {case.environment_seed for case in m1d} == {11, 23, 37, 53, 71}
    assert {case.checkpoint_seed for case in m1d} == {0, 1, 2}
    assert len(m1d) == 30


def test_planner_seed_patch_is_hash_bound_and_requires_private_parameter() -> None:
    patcher = (
        ROOT / "configs/v3/gate5/mtare_planner_seed_v1/apply_mtare_planner_seed_v1.py"
    ).read_text(encoding="utf-8")
    launch = (ROOT / "configs/v3/gate5/roslaunch/explore_seeded.launch").read_text(encoding="utf-8")
    assert patcher.count('"src/') >= 5
    assert "std::random_device" in patcher and "rand()" in patcher
    assert "uncontrolled random sources remain" in patcher
    assert 'getParam("planner_seed"' in patcher
    assert '<arg name="planner_seed"/>' in launch
    assert '<param name="planner_seed" value="$(arg planner_seed)" type="int"/>' in launch
