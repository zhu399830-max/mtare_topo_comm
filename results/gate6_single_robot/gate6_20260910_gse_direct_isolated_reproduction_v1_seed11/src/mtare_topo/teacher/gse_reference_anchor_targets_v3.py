"""Experimental lateral-entry addition; mesh settings must be source-bound."""
from .gse_reference_anchor_targets_v1 import _produce_junction_reference_targets
from .gse_interior_branch_evidence_v1 import bound_interior_evidence
from .gse_lateral_branch_evidence_v1 import bound_lateral_evidence


def produce_junction_reference_targets(bundle, raw, *, axial_spacing_m, angular_segments, field_spacing_m,
                                      qualify_cap_precision=False):
    if type(qualify_cap_precision) is not bool:
        raise ValueError('explicit boolean cap precision policy required')
    interior = bound_interior_evidence(bundle, raw, field_spacing_m=field_spacing_m)
    lateral = bound_lateral_evidence(bundle, raw,
        axial_spacing_m=axial_spacing_m, angular_segments=angular_segments,
        field_spacing_m=field_spacing_m)
    settings=dict(axial_spacing_m=axial_spacing_m,angular_segments=angular_segments) if qualify_cap_precision else None
    options={} if settings is None else dict(cap_source_settings=settings)
    result = _produce_junction_reference_targets(bundle, raw, interior, additional=lateral,**options)
    result['producer_version'] = 'junction_partial_reference_v3_lateral'
    if qualify_cap_precision:
        result['producer_version'] += '_source_precision'
    result['lateral_ambiguous_ray_indices'] = lateral['ambiguous_ray_indices']
    return result
