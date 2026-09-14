"""Synthetic topic inventories only; no ROS, bag or payload access."""
import pytest

from mtare_topo.integration.structural_recording_requirements import (
    recording_readiness, RAW_TOPIC, POSE_TOPIC, BRIDGE_TOPICS,
)


def inventory():
    return {RAW_TOPIC: {'type': 'sensor_msgs/PointCloud2', 'count': 3000},
            POSE_TOPIC: {'type': 'nav_msgs/Odometry', 'count': 3000},
            '/tf_static': {'type': 'tf2_msgs/TFMessage', 'count': 1}}


def test_original_without_bridge_preserved_and_inventory_not_payload_proof():
    result = recording_readiness(inventory(), external_pose_binding=False, case_method='original')
    assert result['model_readiness']['ready']
    assert not result['model_readiness']['payload_and_temporal_binding_verified']
    assert not result['native_route_replay_readiness']['ready']
    assert len(result['native_route_replay_readiness']['reasons']) == 4
    assert result['native_route_replay_readiness']['does_not_invalidate_original_baseline']
    assert not result['baseline_validity_assessed']


def test_3000_registered_and_sync_frames_cannot_replace_missing_raw():
    inv = inventory()
    del inv[RAW_TOPIC]
    inv['/registered_scan'] = {'type': 'sensor_msgs/PointCloud2', 'count': 3000}
    inv['/synchronized_frames'] = {'type': 'std_msgs/String', 'count': 3000}
    result = recording_readiness(inv, external_pose_binding=True, case_method='original')
    assert result['model_readiness']['reasons'] == ['MISSING_TOPIC:/velodyne_points']
    assert not result['baseline_validity_assessed']


def test_bridge_requires_all_published_native_exchange_topics():
    inv = inventory()
    result = recording_readiness(inv, external_pose_binding=False, case_method='bridge')
    assert result['model_readiness']['ready']
    assert len(result['native_route_replay_readiness']['reasons']) == 4
    inv.update({t: {'type': 'std_msgs/String', 'count': 2} for t in BRIDGE_TOPICS})
    assert recording_readiness(inv, external_pose_binding=False, case_method='bridge')['native_route_replay_readiness']['ready']
    inv[BRIDGE_TOPICS[1]]['count'] = 0
    result = recording_readiness(inv, external_pose_binding=False, case_method='bridge')
    assert not result['native_route_replay_readiness']['ready']
    assert not result['baseline_validity_assessed']  # timeout episode not declared invalid


def test_recorded_bridge_does_not_supply_missing_sensor_input():
    inv = {t: {'type': 'std_msgs/String', 'count': 20} for t in BRIDGE_TOPICS}
    result = recording_readiness(inv, external_pose_binding=True, case_method='bridge')
    assert not result['native_route_replay_readiness']['ready']
    assert not result['native_route_replay_readiness']['model_prerequisites_ready']


def test_tf_or_explicit_binding_required_not_inferred_from_odometry():
    inv = inventory()
    del inv['/tf_static']
    assert not recording_readiness(inv, external_pose_binding=False, case_method='original')['model_readiness']['ready']
    result = recording_readiness(inv, external_pose_binding=True, case_method='original')
    assert result['model_readiness']['ready']
    assert result['model_readiness']['pose_binding_mode'] == 'external_declared'
    inv['/tf'] = {'type': 'tf2_msgs/TFMessage', 'count': 50}
    assert recording_readiness(inv, external_pose_binding=False, case_method='original')['model_readiness']['ready']


@pytest.mark.parametrize('topic,kind,count', [(RAW_TOPIC, 'sensor_msgs/PointCloud2', 4),
    (RAW_TOPIC, 'sensor_msgs/PointCloud', 3000), (POSE_TOPIC, 'geometry_msgs/Pose', 3000),
    (POSE_TOPIC, 'nav_msgs/Odometry', 0), ('/tf_static', 'std_msgs/String', 1)])
def test_wrong_type_or_insufficient_population_fails(topic, kind, count):
    inv = inventory(); inv[topic] = {'type': kind, 'count': count}
    assert not recording_readiness(inv, external_pose_binding=False, case_method='original')['model_readiness']['ready']


@pytest.mark.parametrize('value', [None, {'type': 'x', 'count': -1},
    {'type': 'x', 'count': True}, {'type': 'x', 'count': 1.5}, {'count': 5}])
def test_invalid_inventory_rejected(value):
    inv = inventory(); inv[RAW_TOPIC] = value
    with pytest.raises(ValueError): recording_readiness(inv, external_pose_binding=False, case_method='bridge')


def test_explicit_case_and_binding_types_required():
    with pytest.raises(ValueError): recording_readiness(inventory(), external_pose_binding=1, case_method='original')
    with pytest.raises(ValueError): recording_readiness(inventory(), external_pose_binding=True, case_method='unknown')
