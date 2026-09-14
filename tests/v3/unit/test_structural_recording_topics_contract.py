"""Future topic profiles: real existing loader/recorder, synthetic inventories."""
import hashlib
import json
from pathlib import Path

from mtare_topo.evaluation.closed_loop_recording import (
    METHODS, TopicObservation, load_topic_contract, rosbag_record_command, audit_topic_inventory,
)
from mtare_topo.integration.structural_recording_requirements import (
    recording_readiness, BRIDGE_TOPICS,
)


ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / 'configs/v3/gate6/closed_loop_recording_topics_gse_common_v1.json'
REPLAY = ROOT / 'configs/v3/gate6/closed_loop_recording_topics_gse_native_replay_v1.json'


def synthetic_inventory(contract):
    return {t.name: TopicObservation(t.type, 3000) for t in contract if t.required}


def test_existing_loader_accepts_separate_profiles_with_same_recording_population():
    common, replay = load_topic_contract(COMMON), load_topic_contract(REPLAY)
    assert tuple(t.name for t in common) == tuple(t.name for t in replay)
    assert len(common) == 21
    for a, b in zip(common, replay):
        assert (a.name, a.type, a.purpose) == (b.name, b.type, b.purpose)
        assert (a.required, b.required) == ((False, True) if a.name in BRIDGE_TOPICS else (a.required, a.required))
    required = {t.name: t.type for t in common if t.required}
    assert required['/velodyne_points'] == 'sensor_msgs/PointCloud2'
    assert required['/state_estimation'] == 'nav_msgs/Odometry'
    assert required['/state_estimation_at_scan'] == 'nav_msgs/Odometry'


def test_real_record_command_captures_raw_tf_and_native_even_for_disabled_baseline(tmp_path):
    command = rosbag_record_command(load_topic_contract(COMMON), output_bag=tmp_path/'synthetic.bag')
    assert command[:2] == ('rosbag', 'record')
    assert '--lz4' in command
    for topic in ('/velodyne_points', '/state_estimation', '/tf', '/tf_static', *BRIDGE_TOPICS):
        assert command.count(topic) == 1
    assert not (tmp_path/'synthetic.bag').exists()  # command construction, never execution


def test_original_common_pass_survives_no_bridge_but_replay_gap_is_visible():
    common, replay = load_topic_contract(COMMON), load_topic_contract(REPLAY)
    observed = synthetic_inventory(common)
    assert audit_topic_inventory(common, observed).passed
    replay_audit = audit_topic_inventory(replay, observed)
    assert not replay_audit.passed and set(replay_audit.missing_topics) == set(BRIDGE_TOPICS)
    ready = recording_readiness({k: {'type': v.type, 'count': v.messages} for k, v in observed.items()},
                                external_pose_binding=True, case_method='original')
    assert ready['model_readiness']['ready']
    assert not ready['native_route_replay_readiness']['ready']
    assert ready['native_route_replay_readiness']['does_not_invalidate_original_baseline']


def test_full_bridge_inventory_pass_and_missing_raw_cannot_hide_behind_sync_count():
    contract = load_topic_contract(REPLAY)
    observed = synthetic_inventory(contract)
    assert audit_topic_inventory(contract, observed).passed
    del observed['/velodyne_points']
    result = audit_topic_inventory(contract, observed)
    assert not result.passed and result.missing_topics == ('/velodyne_points',)
    assert observed['/state_estimation_at_scan'].messages == 3000


def test_zero_bridge_or_wrong_raw_type_fail_the_relevant_profile():
    common, replay = load_topic_contract(COMMON), load_topic_contract(REPLAY)
    observed = synthetic_inventory(replay)
    observed[BRIDGE_TOPICS[1]] = TopicObservation('std_msgs/String', 0)
    assert audit_topic_inventory(common, observed).passed
    assert audit_topic_inventory(replay, observed).empty_required_topics == (BRIDGE_TOPICS[1],)
    observed['/velodyne_points'] = TopicObservation('sensor_msgs/PointCloud', 3000)
    assert not audit_topic_inventory(common, observed).passed


def test_tf_or_external_pose_is_explicit_second_check_not_legacy_loader_claim():
    observed = synthetic_inventory(load_topic_contract(COMMON))
    assert audit_topic_inventory(load_topic_contract(COMMON), observed).passed
    inventory = {k: {'type': v.type, 'count': v.messages} for k, v in observed.items()}
    assert not recording_readiness(inventory, external_pose_binding=False, case_method='original')['model_readiness']['ready']
    inventory['/tf_static'] = {'type': 'tf2_msgs/TFMessage', 'count': 1}
    result = recording_readiness(inventory, external_pose_binding=False, case_method='original')
    assert result['model_readiness']['ready']
    assert not result['model_readiness']['payload_and_temporal_binding_verified']


def test_contracts_do_not_create_approval_or_claim_organized_shape():
    for path in (COMMON, REPLAY):
        value = json.loads(path.read_text())
        assert value['status'] == 'SPECIFICATION_ONLY_NO_EXECUTION_AUTHORIZATION'
        assert tuple(value['methods']) == METHODS  # unchanged old loader registry
        assert 'approval' not in value
        assert value['policy']['legacy_loader_does_not_enforce_conditional_policy']
        assert value['policy']['retain_native_fields_headers_and_timestamps']
        assert not value['policy']['registered_scan_is_raw_substitute']
        assert not value['policy']['all_sync_frames_imply_model_input_ready']
        assert not value['policy']['approval_or_run_created']


def test_legacy_contract_bytes_untouched():
    for path, expected in [
        ('configs/v3/gate5/closed_loop_recording_topics_v1.json', '22d4f53620352b0eb65edeb0e33e3c475de25fc9db9c825f458cdc2c379e3eae'),
        ('configs/v3/gate6/closed_loop_recording_topics_aee_v2.json', '301266d8f1432e5e6f54ef1ab89efc832904c7355c5a479f3a718f350b5af961'),
    ]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
