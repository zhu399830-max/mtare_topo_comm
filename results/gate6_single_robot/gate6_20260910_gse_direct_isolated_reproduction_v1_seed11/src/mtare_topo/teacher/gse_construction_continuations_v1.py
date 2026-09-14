"""Pinned complete P1a adapter for teacher-only reference continuations.

Node ownership comes from the validated original construction. Continuation
coordinates come from realized operands, never prepended anchor connectors.
No document is loaded here; callers supply independently pinned bytes/data.
"""
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_reference_continuations_v1 import ReferenceSource,reference_continuations


def construction_reference_continuations(document, *, expected_document_sha256):
    if (type(expected_document_sha256) is not str or len(expected_document_sha256)!=64
            or canonical_sha(document)!=expected_document_sha256):
        raise ValueError('independently pinned construction document required')
    groups=construction_incident_paths(document)
    _,realized=load_p1a_realized_construction(document)
    owners={}
    for group in groups:
        for path in group['paths']:
            key=tuple(path['endpoint_key'])
            if key in owners:
                raise ValueError('multiply owned endpoint')
            owners[key]=group['node_id_teacher_only']
    required={(p.primitive_id,side) for p in realized for side in (0,1)}
    if set(owners)!=required:
        raise ValueError('every realized source needs both original endpoint owners')
    sources=tuple(ReferenceSource(p.primitive_id,
        (owners[p.primitive_id,0],owners[p.primitive_id,1]),
        (tuple(map(float,p.centerline_xyz_m[0])),tuple(map(float,p.centerline_xyz_m[-1]))))
        for p in realized)
    return dict(schema='gse_construction_reference_continuations_v1',
        construction_canonical_sha256=expected_document_sha256,
        coordinate_frame='cano_world',sources=sources,
        continuations=reference_continuations(sources),
        endpoint_incidence_count=len(owners),
        geometry_basis='realized_source_endpoints_not_anchor_connectors',
        physical_separation_certified=False,observation_support_verified=False)
