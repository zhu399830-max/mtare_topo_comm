"""Finite first-return correspondence to source mesh cap triangles.

Pure geometry evidence only. Source degree/endpoint ownership and union-surface
provenance must be checked separately before a terminal training target exists.
Triangle contacts at edges are retained as ambiguous, not silently snapped.
"""
import numpy as np


def source_endpoint_cap_faces(mesh, *, angular_segments, endpoint_index):
    """Bind the ORIGINAL mesh_swept_superellipse cap layout, never an ROI cut.

    Checks explicit triangle ownership rather than assuming the last faces are
    a cap. This function does not establish that an endpoint is degree one.
    """
    if (type(angular_segments) is not int or angular_segments < 12
            or angular_segments % 4 or type(endpoint_index) is not int
            or endpoint_index not in (0, 1)):
        raise ValueError('original mesh resolution and endpoint required')
    vertices = mesh.vertices_xyz_m
    faces = mesh.triangle_vertex_indices
    rings, remainder = divmod(len(vertices)-2, angular_segments)
    if remainder or rings < 2 or len(faces) != 2*rings*angular_segments:
        raise ValueError('source swept mesh layout mismatch')
    first_cap = 2*(rings-1)*angular_segments
    indices = first_cap + 2*np.arange(angular_segments) + endpoint_index
    center = len(vertices)-2+endpoint_index
    columns = np.arange(angular_segments)
    following = (columns+1) % angular_segments
    if endpoint_index == 0:
        expected = np.stack((np.full(angular_segments, center), following, columns), axis=1)
    else:
        base = (rings-1)*angular_segments
        expected = np.stack((np.full(angular_segments, center), base+columns, base+following), axis=1)
    if not np.array_equal(faces[indices], expected):
        raise ValueError('source endpoint cap triangle ownership mismatch')
    return indices


def cap_return_evidence(vertices_m, triangles, *, cap_face_indices, rays):
    rays.validate()
    vertices = np.asarray(vertices_m, dtype=np.float64)
    faces = np.asarray(triangles)
    cap = np.asarray(cap_face_indices)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
            or (faces < 0).any() or (faces >= len(vertices)).any()):
        raise ValueError('valid source mesh required')
    if (cap.ndim != 1 or cap.dtype.kind not in 'iu' or len(cap) == 0
            or len(np.unique(cap)) != len(cap) or (cap < 0).any()
            or (cap >= len(faces)).any()):
        raise ValueError('explicit unique source cap faces required')
    if len(rays.valid) > 57600 or len(cap) > 512:
        raise ValueError('local evidence capacity exceeded')
    valid_indices = np.flatnonzero(rays.valid)
    origins = rays.origins_m[valid_indices]
    directions = rays.directions[valid_indices]
    ranges = rays.first_return_m[valid_indices]
    eps = 128 * np.finfo(float).eps
    matches = {}; ambiguous = set()
    for face_index in cap:
        a, b, c = vertices[faces[face_index]]
        e1, e2 = b-a, c-a
        cross = np.cross(e1, e2)
        area = np.linalg.norm(cross)
        if not area > eps * max(1., np.linalg.norm(e1)*np.linalg.norm(e2)):
            raise ValueError('degenerate cap triangle')
        n = cross/area
        denom = directions@n
        eligible = np.abs(denom) > eps
        t = np.divide((a-origins)@n, denom, out=np.zeros(len(origins)), where=eligible)
        # Storage ULP plus supplied measurement bound; no fitted tolerance.
        error = (np.abs(np.spacing(ranges)).astype(float) + rays.range_error_bound_m
                 + eps*np.maximum(1., np.linalg.norm(a-origins, axis=1)))
        eligible &= (t > 0) & (np.abs(t-ranges) <= error)
        q = origins + t[:, None]*directions-a
        d00, d01, d11 = e1@e1, e1@e2, e2@e2
        determinant = d00*d11-d01*d01
        u = (d11*(q@e1)-d01*(q@e2))/determinant
        v = (d00*(q@e2)-d01*(q@e1))/determinant
        bary = np.stack((u, v, 1-u-v), axis=1)
        interior = (bary > eps).all(axis=1)
        contact = (bary >= -eps).all(axis=1) & ~interior
        ambiguous.update(map(int, valid_indices[eligible & contact]))
        for index in valid_indices[eligible & interior]:
            matches.setdefault(int(index), []).append(int(face_index))
    ambiguous.update(i for i, f in matches.items() if len(f) != 1)
    evidence = [dict(ray_index=i, triangle_index=f[0],
                     source_frame_index=int(rays.source_frame_index[i]))
                for i, f in sorted(matches.items()) if i not in ambiguous]
    return dict(cap_return_witnesses=evidence,
                ambiguous_ray_indices=sorted(ambiguous),
                cap_surface_observed=bool(evidence),
                terminal_label=None, training_eligible=False,
                limitation='Cap correspondence alone does not prove a terminal: '
                           'source endpoint ownership and union exterior are not checked here.')
