"""All-source fixed-window proposal diagnostic, never a training exporter."""
from collections import Counter
from copy import deepcopy
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .csg_mesh_provenance import mesh_swept_superellipse
from .gse_portal_ray_evidence import CausalRaySegments
from .gse_roi_crossings_v1 import roi_crossings
from .gse_window_opening_proposals_v1 import propose_window_openings
from .source_witness_binding import permitted_sources


def resolve_competition(proposals):
    """Count every geometric candidate, including unsupported/other sources."""
    rows=deepcopy(proposals)
    counts=Counter(i for r in rows for i in set(r['geometric_outward_crossing_ray_indices']))
    for row in rows:
        row['competing_crossing_ray_indices']=[i for i in row['outward_crossing_ray_indices'] if counts[i]>1]
        row['exclusive_outward_crossing_ray_indices']=[i for i in row['outward_crossing_ray_indices'] if counts[i]==1]
        row['proposal_supported_after_competition']=bool(row['exclusive_outward_crossing_ray_indices'] and row['surface_return_ray_indices'])
    return rows


def diagnose_observation(bundle):
    return _diagnose_observation(bundle,propose_window_openings)


def _diagnose_observation(bundle,proposer,competition_resolver=resolve_competition):
    s,student=bundle['sensor_teacher_only'],bundle['student']
    doc,book,source=bundle['construction_teacher_only'],bundle['codebook_teacher_only'],bundle['source']
    verify_alignment(s,student,doc,book,source)
    frames=np.asarray(source['frame_rows'])
    if frames.shape!=(5,) or frames.dtype.kind not in 'iu' or not np.all(np.diff(frames)>0):
        raise ValueError('five ordered source frames required')
    local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions=np.concatenate([world_directions(local,float(y)) for y in s['yaw_deg']])
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    rays=CausalRaySegments(np.repeat(s['sensor_xyz_m'],11520,axis=0),directions,
        student['ranges_m'].reshape(-1),student['valid_mask'].reshape(-1).astype(bool),
        np.repeat(frames,11520),int(frames[-1]),0.)
    rays.validate()
    _,primitives=load_p1a_realized_construction(doc)
    owners=np.array([v[0] if len(v)==1 else -1 for v in permitted_sources(bundle)],dtype=np.int64)
    center=s['sensor_xyz_m'][-1]; rows=[]
    for index,primitive in enumerate(primitives):
        crossings=roi_crossings(primitive.centerline_xyz_m,center_m=center)
        if crossings.ambiguous_segment_indices:
            raise ValueError('ambiguous ROI roots; no candidate deletion')
        if not len(crossings.positions_m):continue
        mesh=mesh_swept_superellipse(primitive,axial_spacing_m=.05,angular_segments=64)
        for position,direction,arc in zip(crossings.positions_m,crossings.outward_tangents,crossings.source_arc_m):
            result=proposer(mesh.vertices_xyz_m,mesh.triangle_vertex_indices,
                section_center_m=position,outward_direction=direction,roi_center_m=center,
                rays=rays,unique_return_source_index=owners,source_index=index)
            for proposal in result['proposals']:
                rows.append(dict(primitive_id_teacher_only=primitive.primitive_id,
                    reference_arc_m=float(arc),**proposal))
        del mesh
    rows=competition_resolver(rows)
    return dict(source=source,proposals=rows,candidate_count=len(rows),
        supported_count=sum(r['proposal_supported_after_competition'] for r in rows),
        qualified_labels=0,optimizer_steps=0,teacher_complete=False,scientific_gate_pass=False)
