"""Reference competition correction plus source-bound terminal relations."""
from .gse_joint_reference_targets_v2 import _produce_joint_v2
from .gse_joint_reference_targets_v5 import _produce_terminal_joint
from .gse_reference_opening_targets_v3 import produce_opening_reference_targets


def _base(bundle, raw_interfaces):
    return _produce_joint_v2(bundle, raw_interfaces, require_all_rays=False,
                            opening_producer=produce_opening_reference_targets)


def produce_joint_reference_targets(bundle, raw_interfaces):
    result = _produce_terminal_joint(bundle, raw_interfaces, _base)
    result['producer_version'] = 'joint_partial_reference_v6_reference_competition'
    return result
