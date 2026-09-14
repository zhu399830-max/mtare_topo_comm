"""Join construction entities to causal identity inventory, not label creation.

Construction identities are for sampling/provenance only. No geometry is sent
to a model and no observation is selected by prediction scores or visibility.
"""
from .gse_surface_selection_v1 import select_parent, VARIANTS


def inventory_parent(report, split, variant_groups):
    # Reuse full metadata validation; discard the example selection entirely.
    select_parent(report, split, edges_per_parent=1)
    if set(variant_groups) != set(VARIANTS):
        raise ValueError("all three construction variants required")
    parent = report['parent_id']
    signatures = []
    for variant in VARIANTS:
        nodes, seen = {}, set()
        for group in variant_groups[variant]:
            node = group['node_id_teacher_only']
            if node in nodes:
                raise ValueError('duplicate construction node')
            keys = []
            for path in group['paths']:
                primitive, side = path['endpoint_key']
                if (type(primitive) is not str or not primitive.startswith('primitive:edge_') or
                        type(side) is not int or side not in (0, 1) or (primitive, side) in seen):
                    raise ValueError('unique typed physical endpoint incidence required')
                seen.add((primitive, side)); keys.append((primitive, side))
            if not keys:
                raise ValueError('empty construction node')
            nodes[node] = tuple(sorted(keys))
        signatures.append(nodes)
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError('geometry variants disagree on topology identities')
    by_edge = {}
    for interval in report['intervals']:
        edge = interval['traversal_id'].rsplit(':', 1)[0]
        if interval['variants'][0]['sequence_rows']:
            by_edge.setdefault(edge, []).append(interval['traversal_id'])
    entities = []
    for node, keys in sorted(signatures[0].items()):
        if len(keys) == 2:
            continue  # Ordinary construction segmentation is not a structure label.
        incident = sorted({parent + ':' + primitive.removeprefix('primitive:') for primitive, _ in keys})
        available = sorted(t for edge in incident for t in by_edge.get(edge, []))
        entities.append(dict(parent_id=parent, split=split, node_id_teacher_only=node,
            kind='terminal' if len(keys) == 1 else 'junction', degree=len(keys),
            incident_physical_edges=incident, causal_traversals=available,
            candidate_only=True, observable_label=False))
    return dict(parent_id=parent, split=split, entities=entities,
        background_sampling_edges=sorted(by_edge),
        background_is_negative_label=False, observations_selected=0,
        limitation='Incident traversal availability does not prove visibility, proximity or physical connectivity.')
