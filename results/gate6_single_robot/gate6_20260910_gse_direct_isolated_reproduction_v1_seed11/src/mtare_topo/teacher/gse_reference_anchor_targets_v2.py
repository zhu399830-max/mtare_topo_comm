"""Versioned addition of source-interior evidence; original V1 stays callable."""
from .gse_reference_anchor_targets_v1 import _produce_junction_reference_targets
from .gse_interior_branch_evidence_v1 import bound_interior_evidence


def produce_junction_reference_targets(bundle,raw_result):
    interior=bound_interior_evidence(bundle,raw_result)
    result=_produce_junction_reference_targets(bundle,raw_result,interior)
    result['producer_version']='junction_partial_reference_v2'
    result['interior_ambiguous_frame_slots']=interior['ambiguous_frame_slots']
    return result
