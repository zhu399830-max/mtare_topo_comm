"""Distinguish absent ray witnesses from contradictory node evidence.

At least one unique same-source direct witness is still required. Evidence of
another/intervening node vetoes the relation even when that node is unlabelled.
Original V1/V2 behavior remains available; no complete-label claim is added.
"""
from .gse_joint_reference_targets_v2 import _produce_joint_v2


def produce_joint_reference_targets(bundle,raw_interfaces):
    result=_produce_joint_v2(bundle,raw_interfaces,require_all_rays=False)
    result['producer_version']='joint_partial_reference_v3_missing_is_not_conflict'
    return result
