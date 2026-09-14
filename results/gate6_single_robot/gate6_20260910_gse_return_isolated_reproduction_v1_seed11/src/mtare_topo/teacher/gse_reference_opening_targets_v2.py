"""Versioned source-axis/contour correction, non-owning evidence stays unknown."""
from .gse_reference_opening_targets_v1 import _produce_opening_reference_targets
from .gse_window_opening_diagnostic_v2 import diagnose_observation


def produce_opening_reference_targets(bundle):
    result=_produce_opening_reference_targets(bundle,diagnose_observation,require_axis_ownership=True)
    result['producer_version']='opening_partial_reference_v2_axis_contour'
    return result
