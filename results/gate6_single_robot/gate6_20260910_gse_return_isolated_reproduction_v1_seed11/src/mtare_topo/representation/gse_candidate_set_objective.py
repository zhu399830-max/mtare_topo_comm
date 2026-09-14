"""Versioned candidate set objective; uses the tested one-to-one control.

Only the loss sees targets. Objective validity does not qualify the detector.
"""
from mtare_topo.evaluation.gse_candidate_set_loss_control import one_to_one_hard_negative_control


def candidate_set_loss(logits,positions,expected):
    if logits.ndim!=2 or logits.shape[0]!=1:
        raise ValueError('single-observation loss required')
    return one_to_one_hard_negative_control(logits[0],positions,expected,complete_region=True)
