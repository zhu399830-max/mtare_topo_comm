"""Joint partial targets with corrected axis-reference contour ownership."""
from .gse_joint_reference_targets_v2 import _produce_joint_v2
from .gse_reference_opening_targets_v2 import produce_opening_reference_targets


def produce_joint_reference_targets(bundle,raw_interfaces):
    result=_produce_joint_v2(bundle,raw_interfaces,require_all_rays=False,
        opening_producer=produce_opening_reference_targets)
    result['producer_version']='joint_partial_reference_v4_axis_contour'
    return result
