"""Source-bound partial terminal correspondence; V4 remains unchanged."""
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .gse_joint_reference_targets_v4 import produce_joint_reference_targets as previous
from .gse_directed_interface_binding_v1 import interpret_bound_result
from .gse_terminal_window_component_v1 import terminal_window_component
from .gse_terminal_relation_witness_v1 import terminal_relation_witness
from .swept_superellipse_field import SweptSuperellipseProvenanceField


def produce_joint_reference_targets(bundle, raw_interfaces):
    return _produce_terminal_joint(bundle, raw_interfaces, previous)


def _produce_terminal_joint(bundle, raw_interfaces, base_producer, *, field_spacing_m=0.01):
    # Validate original frames, float32 ray parameters and every interface hit.
    interpret_bound_result(bundle, raw_interfaces)
    result = base_producer(bundle, raw_interfaces)
    _, primitives = load_p1a_realized_construction(bundle['construction_teacher_only'])
    ids = [p.primitive_id for p in primitives]
    if ids != bundle['codebook_teacher_only']['primitive_ids']:
        raise ValueError('operand distance order differs from source inventory')
    sensor = bundle['sensor_teacher_only']
    distances = SweptSuperellipseProvenanceField(primitives, spacing_m=field_spacing_m).operand_signed_distances_sparse(sensor['sensor_xyz_m'])
    record, provenance = result['record'], result['teacher_provenance']
    candidates = {i: [] for i in range(len(record['openings']))}
    diagnostics = []
    for ti, terminal in enumerate(provenance['terminals']):
        source, side = terminal['endpoint_key_teacher_only']
        component = terminal_window_component(primitives[ids.index(source)].centerline_xyz_m,
            endpoint_index=side, center_m=sensor['sensor_xyz_m'][-1])
        ai = provenance['terminal_anchor_start'] + ti
        for oi, opening in enumerate(provenance['openings']):
            if opening['primitive_id_teacher_only'] != source:
                continue
            entry = dict(anchor_index=ai, opening_index=oi, component=component)
            if (component['status'] != 'REFERENCE_COMPONENT_ONLY'
                    or opening['reference_arc_m'] != component['opening_reference_arc_m']):
                diagnostics.append(dict(entry, status='UNKNOWN_DIFFERENT_LOCAL_COMPONENT'))
                continue
            evidence = terminal_relation_witness(source_id=source,
                node_id=terminal['node_id_teacher_only'],
                cap_rays=[w['ray_index'] for w in terminal['witnesses']],
                opening_rays=opening['crossing_ray_indices'], source_ids=ids,
                origin_distances=distances, interfaces=raw_interfaces['interfaces_teacher_only'],
                interface_hits=raw_interfaces['raw_interface_intersections'],
                first_return=bundle['student']['ranges_m'], frame_rows=bundle['source']['frame_rows'])
            diagnostics.append(dict(entry, **evidence))
            if evidence['status'] == 'TWO_SIDED_SOURCE_WITNESS_ONLY':
                candidates[oi].append(ai)
    for oi, anchors in candidates.items():
        existing = {i for i, v in enumerate(record['membership'][oi]) if v is True}
        combined = existing | set(anchors)
        if len(combined) == 1 and anchors:
            record['membership'][oi][anchors[0]] = True
            provenance['relations'][oi] = dict(opening_index=oi, anchor_index=anchors[0],
                status='PARTIAL_TERMINAL_SOURCE_COMPONENT_CORRESPONDENCE',
                evidence_indices=[i for i, d in enumerate(diagnostics)
                    if d['opening_index'] == oi and d['anchor_index'] == anchors[0]
                    and d['status'] == 'TWO_SIDED_SOURCE_WITNESS_ONLY'])
        elif len(combined) > 1:
            record['membership'][oi] = [None] * len(record['anchors'])
            provenance['relations'][oi] = dict(opening_index=oi, anchor_index=None,
                status='UNKNOWN_COMPETING_TERMINAL_CORRESPONDENCE',
                candidate_anchor_indices=sorted(combined))
    provenance['terminal_relations'] = diagnostics
    result['producer_version'] = 'joint_partial_reference_v5_terminal_correspondence'
    # Implementation exists, but neither reference quality nor complete labels
    # are certified by the presence of a positive correspondence.
    result['missing_tasks'] = ['complete_scoring_regions', 'reference_target_quality',
                               'terminal_correspondence_quality']
    result['target_record_sha256'] = canonical_sha(record)
    return result
