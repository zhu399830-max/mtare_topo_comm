"""Observed intervening-junction exclusion for terminal reference membership.

This is local reference incidence, NOT global reachability or a physical
separation certificate. The complete source construction is teacher-only.
Preserve unknowns unless a visible terminal, visible intervening junction,
source-continuation exclusion, and original directional rays agree.

EXPERIMENTAL: synthetic terminal-interior T-junction acceptance currently
fails upstream junction detection. Not qualified for a real label export.
"""
import numpy as np

from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .gse_joint_reference_targets_v6 import produce_joint_reference_targets as previous
from .gse_construction_continuations_v1 import construction_reference_continuations
from .gse_directed_interface_binding_v1 import interpret_bound_result
from .swept_superellipse_field import SweptSuperellipseProvenanceField


def produce_joint_reference_targets(bundle, raw_interfaces):
    return _produce_terminal_exclusion(bundle, raw_interfaces, previous)


def _produce_terminal_exclusion(bundle, raw_interfaces, base_producer, *, departure_field=None, field_spacing_m=0.01,
                                cap_source_settings=None):
    result = base_producer(bundle, raw_interfaces)
    document = bundle['construction_teacher_only']
    reference = construction_reference_continuations(document,
        expected_document_sha256=result['source_binding']['construction_sha256'])
    components = {s: c for c in reference['continuations'] for s in c.source_ids}
    directed = interpret_bound_result(bundle, raw_interfaces)['interfaces']
    if cap_source_settings is not None:
        from mtare_topo.evaluation.gse_bound_cap_precision import audit_bound_cap_precision
        precision_audits={}
        for kind in ('entering','leaving'):
            precision=audit_bound_cap_precision(bundle,raw_interfaces,direction_kind=kind,**cap_source_settings)
            for key,evidence in directed.items():
                stable=precision['interfaces'][key]['stable_'+kind+'_ray_indices']
                if not set(stable)<=set(evidence[kind+'_ray_indices']):
                    raise ValueError('cap precision cannot invent directional evidence')
                evidence[kind+'_ray_indices']=list(stable)
            precision_audits[kind]=precision
        result['terminal_exclusion_cap_precision']=precision_audits
    interfaces = raw_interfaces['interfaces_teacher_only']
    _, primitives = load_p1a_realized_construction(document)
    ids = [p.primitive_id for p in primitives]
    distances = SweptSuperellipseProvenanceField(primitives, spacing_m=field_spacing_m).operand_signed_distances_sparse(
        bundle['sensor_teacher_only']['sensor_xyz_m'])
    record = result['record']; provenance = result['teacher_provenance']
    audit = []
    for ti, terminal in enumerate(provenance['terminals']):
        ai = provenance['terminal_anchor_start'] + ti
        terminal_node = terminal['node_id_teacher_only']
        terminal_source = terminal['endpoint_key_teacher_only'][0]
        tc = components[terminal_source]
        boundaries = tc.structural_boundaries
        # A complete simple terminal corridor; unresolved or closed reference
        # chains cannot be certified by snapping/identity or a useful subset.
        if (tc.unresolved_degree_two_nodes or len(boundaries) != 2
                or sum(node == terminal_node for node, _, _ in boundaries) != 1):
            continue
        junction_node = next(node for node, _, _ in boundaries if node != terminal_node)
        junctions = [j for j in provenance['anchors'] if j['node_id_teacher_only'] == junction_node]
        if len(junctions) != 1:
            continue
        junction = junctions[0]
        cap_rays = {w['ray_index'] for w in terminal['witnesses']}
        terminal_interfaces = [i['interface_id_teacher_only'] for i in interfaces
            if i['node_id_teacher_only'] == junction_node
            and i['endpoint_key_teacher_only'][0] in tc.source_ids]
        if len(terminal_interfaces) != 1:
            continue
        key = terminal_interfaces[0]
        cap_entering = cap_rays & set(directed[key]['entering_ray_indices'])
        # A sensor already in the terminal operand need not cross its entrance
        # to see the cap. Require actual containment and unique source origin;
        # a signed tangent projection alone is not containment.
        inside = {f for f in range(5) if distances[f, ids.index(terminal_source)] < 0
                  and np.sum(distances[f] < 0) == 1}
        cap_inside = {r for r in cap_rays if r // 11520 in inside}
        outgoing = {r for r in directed[key]['leaving_ray_indices'] if r // 11520 in inside}
        if departure_field is not None:
            for interface_key, rays in zip(junction['interface_ids'], junction[departure_field], strict=True):
                if interface_key == key:
                    outgoing.update(r for r in rays if r // 11520 in inside)
        for oi, opening in enumerate(provenance['openings']):
            if record['membership'][oi][ai] is not None:
                continue  # never overwrite a positive or silently resolve conflict
            oc = components[opening['primitive_id_teacher_only']]
            if (oc == tc or oc.unresolved_degree_two_nodes
                    or not any(n == junction_node for n, _, _ in oc.structural_boundaries)
                    or any(n == terminal_node for n, _, _ in oc.structural_boundaries)):
                continue
            relation = provenance['relations'][oi]
            ji = relation.get('anchor_index')
            if (type(ji) is not int or ji >= provenance['terminal_anchor_start']
                    or provenance['anchors'][ji]['node_id_teacher_only'] != junction_node
                    or record['membership'][oi][ji] is not True):
                continue
            opening_rays = set(relation.get('ray_indices', []))
            transitions = outgoing & opening_rays
            if not opening_rays or not (cap_entering or (cap_inside and transitions)):
                continue
            record['membership'][oi][ai] = False
            audit.append(dict(opening_index=oi, anchor_index=ai,
                junction_node_teacher_only=junction_node,
                terminal_continuation_sources=list(tc.source_ids),
                opening_continuation_sources=list(oc.source_ids),
                cap_entering_ray_indices=sorted(cap_entering),
                cap_source_interior_ray_indices=sorted(cap_inside),
                opening_junction_ray_indices=sorted(opening_rays),
                terminal_outgoing_opening_ray_indices=sorted(transitions),
                status='PARTIAL_OBSERVED_REFERENCE_INTERVENING_JUNCTION_EXCLUSION'))
    provenance['terminal_nonmembership'] = audit
    result['producer_version'] = 'joint_partial_reference_v7_terminal_exclusion'
    result['target_record_sha256'] = canonical_sha(record)
    result['missing_tasks'] = sorted(set(result['missing_tasks']) | {'terminal_exclusion_quality'})
    return result
