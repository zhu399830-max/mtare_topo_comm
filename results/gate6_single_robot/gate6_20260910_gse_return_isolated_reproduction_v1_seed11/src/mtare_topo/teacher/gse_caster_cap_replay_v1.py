"""Original caster float32 inputs and exact stored-parameter cap witnesses.

This is not a renderer replacement: it lists intersections of one supplied
original closed mesh. Original CSG union grouping may select another distance;
such cases remain unconfirmed. No distance tolerance or semantic labels.
"""
import numpy as np


def pack_caster_inputs(vertices, origins, directions):
    v=np.asarray(vertices,dtype=np.float64)
    o=np.asarray(origins,dtype=np.float64);d=np.asarray(directions,dtype=np.float64)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or o.ndim!=2 or o.shape[1:]!=(3,)
            or d.shape!=o.shape or not all(np.isfinite(x).all() for x in (v,o,d))):
        raise ValueError('finite source mesh and matching rays required')
    norms=np.linalg.norm(d,axis=1)
    if np.any(norms<=0):raise ValueError('nonzero ray directions required')
    # Exactly the source caster order: normalize float64, concatenate, cast.
    vertices32=v.astype(np.float32)
    rays32=np.concatenate((o,d/norms[:,None]),axis=1).astype(np.float32)
    if not np.isfinite(vertices32).all() or not np.isfinite(rays32).all():
        raise ValueError('source quantization overflow')
    return vertices32,rays32


def replay_cap(mesh, *, cap_face_indices, origins, directions, first_return, valid):
    vertices,rays=pack_caster_inputs(mesh.vertices_xyz_m,origins,directions)
    ranges=np.asarray(first_return);valid=np.asarray(valid);caps=np.asarray(cap_face_indices)
    if (ranges.shape!=(len(rays),) or ranges.dtype!=np.float32
            or valid.shape!=ranges.shape or valid.dtype!=bool
            or np.any(~np.isfinite(ranges[valid])) or np.any(ranges[valid]<=0)
            or caps.ndim!=1 or caps.dtype.kind not in 'iu' or len(caps)==0
            or len(np.unique(caps))!=len(caps) or (caps<0).any()
            or (caps>=len(mesh.triangle_vertex_indices)).any() or len(rays)>57600):
        raise ValueError('original bounded ranges, validity and cap faces required')
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices),
        o3d.core.Tensor(mesh.triangle_vertex_indices.astype(np.uint32))))
    raw={k:v.numpy() for k,v in scene.list_intersections(o3d.core.Tensor(rays)).items()}
    rayids=raw['ray_ids'].astype(np.int64);faceids=raw['primitive_ids'].astype(np.int64)
    hits=raw['t_hit']
    if hits.dtype!=np.float32:raise ValueError('backend hit precision differs')
    match=valid[rayids]&np.isin(faceids,caps)&(hits==ranges[rayids])
    selected=np.flatnonzero(match)
    return dict(exact_cap_witnesses=[dict(ray_index=int(rayids[i]),triangle_index=int(faceids[i]),
                                        stored_t=float(hits[i])) for i in selected],
                backend_version=o3d.__version__,total_intersections=len(hits),
                semantic_label=None,training_eligible=False,
                limitation='Exact parameter match only; caller must bind causal frames, '
                           'original mesh identity and saved CSG source provenance.')
