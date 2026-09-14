"""Read-only topic-inventory prerequisites, not payload or baseline scoring.

No bag I/O. Counts cannot establish synchronization, organized ray layout,
five-frame pose interpolation, proper extrinsics, causal binding or usable model
features. external_pose_binding is an explicit producer declaration (e.g. a
documented commanded-pose convention); it is never inferred from topic names.
Missing structural topics in an original baseline do not invalidate that run.
"""
from collections.abc import Mapping


RAW_TOPIC = '/velodyne_points'
POSE_TOPIC = '/state_estimation'
TF_TOPICS = ('/tf', '/tf_static')
BRIDGE_TOPICS = tuple('/sensor_coverage_planner/native_structure/' + suffix
                      for suffix in ('candidates', 'advice', 'decision', 'feedback'))


def recording_readiness(inventory, *, external_pose_binding, case_method):
    """Check mapping ``topic -> {'type': ROS type, 'count': integer}``.

    ``ready`` means these necessary inventory conditions only. Both /tf and
    /tf_static may exist; one nonempty correctly typed topic suffices to request
    a later transform-chain check. It does not prove that required transforms
    are available. An explicit external binding bypasses only TF presence.

    Zero recorded advice can be a valid timeout/fallback episode. Such a run is
    not rejected as an experiment, but does not have a recorded advice exchange
    for the full structural-advice replay requested here.
    """
    if type(external_pose_binding) is not bool or case_method not in ('original', 'bridge'):
        raise ValueError('explicit bool pose binding and original/bridge case method required')
    if not isinstance(inventory, Mapping):
        raise ValueError('topic inventory mapping required')
    for topic, value in inventory.items():
        if (type(topic) is not str or not topic.startswith('/') or
                not isinstance(value, Mapping) or set(value) != {'type', 'count'} or
                type(value['type']) is not str or not value['type'] or
                type(value['count']) is not int or value['count'] < 0):
            raise ValueError('inventory requires exact type/count with nonnegative integer count')

    def failures(topic, message_type, minimum=1):
        value = inventory.get(topic)
        if value is None:
            return [f'MISSING_TOPIC:{topic}']
        result = []
        if value['type'] != message_type:
            result.append(f'WRONG_TYPE:{topic}:expected={message_type}')
        if value['count'] < minimum:
            result.append(f'INSUFFICIENT_MESSAGES:{topic}:minimum={minimum}')
        return result

    model_reasons = failures(RAW_TOPIC, 'sensor_msgs/PointCloud2', 5)
    model_reasons += failures(POSE_TOPIC, 'nav_msgs/Odometry')
    if not external_pose_binding and not any(not failures(t, 'tf2_msgs/TFMessage') for t in TF_TOPICS):
        model_reasons.append('MISSING_TF_OR_EXPLICIT_EXTERNAL_POSE_BINDING')
    model = dict(ready=not model_reasons, reasons=model_reasons,
                 scope='necessary_topic_inventory_only',
                 pose_binding_mode='external_declared' if external_pose_binding else 'tf_requires_payload_validation',
                 registered_scan_is_raw_substitute=False,
                 payload_and_temporal_binding_verified=False)
    reasons = [reason for topic in BRIDGE_TOPICS
               for reason in failures(topic, 'std_msgs/String')]
    # Original bags may also be proposed for replay: expose their missing bridge
    # traces without retroactively changing original recording validity.
    route = dict(ready=not reasons and model['ready'], reasons=reasons,
                 model_prerequisites_ready=model['ready'],
                 does_not_invalidate_original_baseline=True,
                 status='INVENTORY_READY' if not reasons and model['ready'] else 'INVENTORY_INCOMPLETE')
    return dict(model_readiness=model, native_route_replay_readiness=route,
                case_method=case_method, baseline_validity_assessed=False,
                no_payload_read=True)
