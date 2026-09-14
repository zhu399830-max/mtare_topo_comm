"""Experimental joint lateral entry, terminal positives and exclusions.

Requires explicit original operand mesh and distance-field settings. Full annotation and real
population qualification are not implied by this implementation.
"""
from functools import partial
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from .gse_reference_anchor_targets_v3 import produce_junction_reference_targets
from .gse_reference_opening_targets_v3 import produce_opening_reference_targets
from .gse_joint_reference_targets_v2 import _produce_joint_v2, _add_interior_correspondence
from .gse_joint_reference_targets_v5 import _produce_terminal_joint
from .gse_joint_reference_targets_v7 import _produce_terminal_exclusion


def produce_joint_reference_targets(bundle, raw, *, axial_spacing_m, angular_segments, field_spacing_m,
                                    qualify_cap_precision=False):
    if type(qualify_cap_precision) is not bool:
        raise ValueError('explicit boolean cap precision policy required')
    anchor_options=dict(qualify_cap_precision=True) if qualify_cap_precision else {}
    anchor = partial(produce_junction_reference_targets,
        axial_spacing_m=axial_spacing_m, angular_segments=angular_segments,
        field_spacing_m=field_spacing_m,**anchor_options)

    def base(b, r):
        result = _produce_joint_v2(b, r, require_all_rays=False,
            opening_producer=produce_opening_reference_targets, anchor_producer=anchor)
        # Only incoming, source-owned surface rays establish opening ownership.
        # Departure supports an anchor branch but is not an opening witness.
        _add_interior_correspondence(result, b, r,
            witness_field='surface_entry_witness_ray_indices')
        result['target_record_sha256'] = canonical_sha(result['record'])
        return result

    def positive(b, r):
        return _produce_terminal_joint(b, r, base, field_spacing_m=field_spacing_m)

    exclusion_options=dict(cap_source_settings=dict(axial_spacing_m=axial_spacing_m,
        angular_segments=angular_segments)) if qualify_cap_precision else {}
    result = _produce_terminal_exclusion(bundle, raw, positive,
        departure_field='surface_departure_witness_ray_indices', field_spacing_m=field_spacing_m,**exclusion_options)
    result['producer_version'] = 'joint_partial_reference_v8_lateral_terminal_exclusion'
    result['geometry_evidence_settings'] = dict(axial_spacing_m=axial_spacing_m,
        angular_segments=angular_segments, field_spacing_m=field_spacing_m,
        intersections='original_float32_rays_complete_operand_surfaces',
        recast_first_returns=False)
    if qualify_cap_precision:
        result['producer_version'] += '_source_precision'
        result['geometry_evidence_settings']['cap_precision_policy']='source_float64_float32_interval_v1'
    return result
