"""Teacher-only binding of degree-one source endpoints to observed cap returns.

The reader must validate construction and sealed return provenance first.
Unique saved operand provenance is required; ambiguous multi-source hits are
not assigned to a convenient endpoint. This produces partial reference evidence,
not complete background labels or proof that hidden branches do not exist.
"""
import numpy as np
from .source_witness_binding import validate_permission

from .csg_mesh_provenance import mesh_swept_superellipse
from .gse_cap_return_evidence_v1 import source_endpoint_cap_faces
from .gse_caster_cap_replay_v1 import replay_cap


def terminal_reference_evidence(groups, primitives, *, rays, membership_codes, source_sets,
                                local_center_m=None, source_directions=None, source_permission=None):
    rays.validate()
    allowed = validate_permission(source_permission, rays.valid.shape)
    raw = np.asarray(source_directions, dtype=np.float64)
    if raw.shape != rays.directions.shape or not np.isfinite(raw).all():
        raise ValueError('original pre-normalization source directions required')
    norms = np.linalg.norm(raw, axis=1)
    if np.any(norms <= 0) or not np.array_equal(raw/norms[:,None], rays.directions):
        raise ValueError('source ray normalization binding differs')
    if rays.range_error_bound_m != 0.:
        raise ValueError('exact backend correspondence accepts no range tolerance')
    groups = tuple(groups)
    primitives = tuple(primitives)
    ids = [p.primitive_id for p in primitives]
    if len(set(ids)) != len(ids):
        raise ValueError('unique source primitive inventory required')
    codes = np.asarray(membership_codes)
    if (codes.shape != rays.valid.shape or codes.dtype != np.uint16
            or type(source_sets) is not list or not source_sets or source_sets[0] != []
            or np.any(codes >= len(source_sets))):
        raise ValueError('bound source codebook and ray population required')
    for i, values in enumerate(source_sets):
        if (type(values) is not list or (i > 0 and not values)
                or any(type(v) is not int or not 0 <= v < len(ids) for v in values)
                or values != sorted(set(values))):
            raise ValueError('invalid source membership set')
    if not np.array_equal(codes != 0, rays.valid):
        raise ValueError('source code and return validity differ')
    lookup = {name: i for i, name in enumerate(ids)}
    if local_center_m is not None:
        local_center_m = np.asarray(local_center_m, dtype=float)
        if local_center_m.shape != (3,) or not np.isfinite(local_center_m).all():
            raise ValueError('finite current observation center required')
    nodes = set(); owned = set(); rows = []
    # Validate all ownership before producing any evidence, including junctions.
    for group in groups:
        node = group['node_id_teacher_only']
        if type(node) is not str or not node or node in nodes or not group['paths']:
            raise ValueError('unique nonempty construction nodes required')
        nodes.add(node)
        anchor = np.asarray(group['anchor_world_m'])
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError('finite source anchor required')
        for path in group['paths']:
            key = path['endpoint_key']
            if (len(key) != 2 or key[0] not in lookup or type(key[1]) is not int
                    or key[1] not in (0, 1) or key in owned):
                raise ValueError('ambiguous endpoint ownership')
            owned.add(key)
    for group in groups:
        if len(group['paths']) != 1:
            continue  # Degree-two construction cuts never become terminals.
        if (local_center_m is not None
                and np.linalg.norm(np.asarray(group['anchor_world_m'])-local_center_m) >= 10.):
            continue  # Original open 10m ROI; no cap or node created at its edge.
        identity, side = group['paths'][0]['endpoint_key']
        index = lookup[identity]
        mesh = mesh_swept_superellipse(primitives[index], axial_spacing_m=.05, angular_segments=64)
        caps = source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=side)
        result = replay_cap(mesh, cap_face_indices=caps, origins=rays.origins_m,
                            directions=raw, first_return=rays.first_return_m, valid=rays.valid)
        accepted = []; rejected = []
        for witness in result['exact_cap_witnesses']:
            witness = {**witness, 'source_frame_index': int(rays.source_frame_index[witness['ray_index']])}
            membership = source_sets[int(codes[witness['ray_index']])]
            (accepted if allowed[witness['ray_index']] and membership == [index] else rejected).append(witness)
        rows.append(dict(node_id_teacher_only=group['node_id_teacher_only'],
                         anchor_world_m=list(group['anchor_world_m']),
                         endpoint_key_teacher_only=[identity, side],
                         terminal_reference_observed=bool(accepted),
                         accepted_witnesses=accepted, nonunique_or_other_source_witnesses=rejected,
                         backend_version=result['backend_version'],
                         correspondence='EXACT_FLOAT32_BACKEND_PARAMETER',
                         semantic_label=None, complete_region=False,
                         reason=None if accepted else 'NO_UNIQUE_SOURCE_CAP_RETURN'))
    return dict(terminal_references=rows, model_input_fields=[], training_eligible=False,
                limitation='Observed construction reference is partial automatic evidence; '
                           'does not establish complete local topology or annotation coverage.')
