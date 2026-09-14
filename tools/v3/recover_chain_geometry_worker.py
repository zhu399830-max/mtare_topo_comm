"""Isolated two-source, all-hit supplement; never generates teacher targets."""
import argparse,hashlib,importlib,io,json,resource
from pathlib import Path
from recover_lateral_witness_worker import load_archive, bounded_scene_threads, write_gzip_json, validate_indices

def run(root,scope,output):
    import numpy as np
    import open3d as o3d
    modules=load_archive(root,scope)
    entry=scope['entry'];ids=validate_indices(entry['original_ray_indices'])
    files={}
    for p,h in entry['files'].items():
        raw=(root/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('input drift '+p)
        if p.endswith('.npz'):
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:files[p]={k:z[k].copy() for k in z.files}
        else:files[p]=json.loads(raw)
    student=next(v for p,v in files.items() if '/student/' in p)
    sensor=next(v for p,v in files.items() if '/source_evidence/' in p and p.endswith('.npz'))
    doc=next(v for p,v in files.items() if p.endswith('_constructions.json'))
    _,all_primitives=modules['construction'].load_p1a_realized_construction(doc)
    byid={p.primitive_id:p for p in all_primitives}
    primitives=[byid[k] for k in scope['sources']]
    settings=scope['geometry_settings']
    meshes=[modules['mesh'].mesh_swept_superellipse(p,axial_spacing_m=settings['axial_spacing_m'],angular_segments=settings['angular_segments']) for p in primitives]
    local=modules['sensor'].lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions=np.concatenate([modules['sensor'].world_directions(local,float(y)) for y in sensor['yaw_deg']])
    _,packed=modules['pack'].pack_caster_inputs(np.empty((0,3)),np.repeat(sensor['sensor_xyz_m'],11520,axis=0),directions)
    query=packed[np.asarray(ids,dtype=np.int64)]
    with bounded_scene_threads(o3d,1) as threads:
        caster=modules['mesh'].CSGMeshProvenanceRaycaster(meshes)
        raw={k:v.numpy() for k,v in caster.scene.list_intersections(o3d.core.Tensor(query)).items()}
        mapping=dict(caster.geometry_to_operand)
    if threads!=dict(scene_builds=1,intersection_queries=1,nthreads=1):raise ValueError('unexpected scene calls')
    field_module=importlib.import_module('mtare_topo.teacher.swept_superellipse_field')
    field=field_module.SweptSuperellipseProvenanceField(primitives,spacing_m=settings['field_spacing_m'])
    origin_dist=field.operand_signed_distances(sensor['sensor_xyz_m'])
    if not np.isfinite(origin_dist).all():raise ValueError('nonfinite origin field')
    grouped=[[] for _ in ids]
    for j,t in enumerate(raw['t_hit']):
        q=int(raw['ray_ids'][j]);operand=mapping[int(raw['geometry_ids'][j])];triangle=int(raw['primitive_ids'][j])
        xyz=query[q,:3].astype(np.float64)+float(t)*query[q,3:].astype(np.float64)
        ray=ids[q];distance=float(student['ranges_m'].reshape(-1)[ray]);valid=bool(student['valid_mask'].reshape(-1)[ray])
        grouped[q].append(dict(source_id=scope['sources'][operand],triangle=triangle,t=float(t),
            world_xyz_m=xyz.tolist(),inside_original_10m=bool(np.linalg.norm(xyz-sensor['sensor_xyz_m'][-1])<10.),
            before_first_return=bool(0<=t<distance and valid)))
    rows=[]
    for q,ray in enumerate(ids):
        hits=sorted(grouped[q],key=lambda h:(h['t'],h['source_id'],h['triangle']))
        rows.append(dict(ray_index=ray,frame_slot=ray//11520,frame_row=entry['frame_rows'][ray//11520],
            packed_ray=query[q].tolist(),first_return_m=float(student['ranges_m'].reshape(-1)[ray]),
            valid=bool(student['valid_mask'].reshape(-1)[ray]),hits=hits))
    response=dict(status='GEOMETRY_SUPPLEMENT_NOT_CHAIN_OR_LABEL_QUALIFICATION',sources=scope['sources'],
        archive_sha256=scope['archive']['sha256'],rays=rows,origin_operand_signed_distance_m=origin_dist.tolist(),
        origin_containment_note='Archived .025 field values, not exact mesh parity or robot safety certificate',
        current_sensor_world_m=sensor['sensor_xyz_m'][-1].tolist(),current_yaw_deg=float(sensor['yaw_deg'][-1]),
        thread_execution=threads,mesh_counts=[dict(vertices=len(m.vertices_xyz_m),triangles=len(m.triangle_vertex_indices)) for m in meshes],
        teacher_target_calls=0,new_labels=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_gzip_json(output,response)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--card',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    run(a.root,json.loads(a.card.read_text())['scope'],a.output)
