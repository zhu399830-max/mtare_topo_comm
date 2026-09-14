"""Bind archived interface intersections to original axes and caster rays."""
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from .gse_caster_cap_replay_v1 import pack_caster_inputs
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_directed_interface_evidence_v1 import directed_evidence
from .source_witness_binding import named_sources


def bind_axes(groups,interfaces):
    paths={}
    for group in groups:
        for path in group['paths']:
            key=(group['node_id_teacher_only'],tuple(path['endpoint_key']))
            if key in paths:raise ValueError('duplicate node incidence')
            paths[key]=path
    output=[];seen=set()
    for interface in interfaces:
        identifier=interface['interface_id_teacher_only']
        key=(interface['node_id_teacher_only'],tuple(interface['endpoint_key_teacher_only']))
        if type(identifier) is not int or identifier<0 or identifier in seen or key not in paths:
            raise ValueError('raw interface must bind a unique actual incidence')
        seen.add(identifier);path=paths[key];points=np.asarray(path['points_world_m'],dtype=np.float64)
        start=path['axis_start_index']
        if (type(start) is not int or start not in (0,1) or points.ndim!=2 or points.shape[1:]!=(3,)
                or not np.isfinite(points).all() or len(points)<start+2):
            raise ValueError('actual source axis missing; cannot substitute connector')
        tangent=points[start+1]-points[start];length=float(np.linalg.norm(tangent))
        if length==0:raise ValueError('degenerate actual source segment')
        output.append(dict(interface_id_teacher_only=identifier,node_id_teacher_only=key[0],
            source_key_teacher_only=key[1][0],inward_direction=(tangent/length).tolist()))
    return output


def interpret_bound_result(bundle,raw_result):
    sensor=bundle['sensor_teacher_only'];student=bundle['student'];source=bundle['source']
    doc=bundle['construction_teacher_only'];book=bundle['codebook_teacher_only']
    verify_alignment(sensor,student,doc,book,source)
    if any(raw_result['source'][k]!=source[k] for k in ('task','source_sequence_id','frame_rows')):
        raise ValueError('raw intersection observation differs')
    interfaces=bind_axes(construction_incident_paths(doc),raw_result['interfaces_teacher_only'])
    lookup={i['interface_id_teacher_only']:i for i in interfaces}
    local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
    origins=np.repeat(sensor['sensor_xyz_m'],11520,axis=0)
    _,packed=pack_caster_inputs(np.empty((0,3)),origins,directions)
    records=raw_result['raw_interface_intersections']
    for r in records:
        ray=r['ray_index'];key=r['interface_id_teacher_only'];t=r['t']
        if (type(ray) is not int or not 0<=ray<len(packed) or key not in lookup
                or r['source_key_teacher_only']!=lookup[key]['source_key_teacher_only']
                or r['source_frame_index']!=ray//11520
                or r['source_frame_row']!=source['frame_rows'][ray//11520]
                or not np.isfinite(t) or float(np.float32(t))!=t):
            raise ValueError('raw interface source/frame/float32 parameter mismatch')
        xyz=packed[ray,:3].astype(np.float64)+t*packed[ray,3:].astype(np.float64)
        if (not np.array_equal(xyz,np.asarray(r['intersection_world_m']))
                or r['inside_roi']!=bool(np.linalg.norm(xyz-sensor['sensor_xyz_m'][-1])<10.)):
            raise ValueError('raw intersection point or original ROI differs')
    ids=book['primitive_ids']
    return_sources=named_sources(bundle)
    result=directed_evidence(records,interfaces,directions=packed[:,3:].astype(np.float64),
        first_return=student['ranges_m'].reshape(-1),valid=student['valid_mask'].reshape(-1).astype(bool),
        return_sources=return_sources)
    return dict(source=source,**result)
