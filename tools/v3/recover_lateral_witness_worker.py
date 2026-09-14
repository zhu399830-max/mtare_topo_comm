"""Isolated original-mesh intersection recovery; never calls target producer.

The caller supplies a frozen scope and one owned output path. Each worker is
one case, preserving the original3GiB address-space ceiling. All operand meshes
are retained; only the query-ray population is reduced to saved witnesses.
"""
import argparse,gzip,hashlib,importlib,json,os,resource,sys
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def bounded_scene_threads(open3d, threads):
    """Process-local API adapter; archived geometry/filter code stays untouched.

    Only for this fresh single-worker process. Restore the native constructor
    even when construction or intersection fails. Do not mutate ray tensors.
    """
    if type(threads) is not int or threads != 1:
        raise ValueError('explicit single-thread recovery only')
    geometry = open3d.t.geometry
    native = geometry.RaycastingScene
    calls = {'scene_builds': 0, 'intersection_queries': 0, 'nthreads': threads}

    class Scene:
        def __init__(self, *args, **kwargs):
            if args or kwargs:
                raise ValueError('archived zero-argument constructor expected')
            self._scene = native(nthreads=threads)
            calls['scene_builds'] += 1

        def list_intersections(self, rays):
            calls['intersection_queries'] += 1
            return self._scene.list_intersections(rays, nthreads=threads)

        def __getattr__(self, name):
            return getattr(self._scene, name)

    geometry.RaycastingScene = Scene
    try:
        yield calls
    finally:
        geometry.RaycastingScene = native


def validate_indices(indices):
    if not indices or any(type(i) is not int or not 0<=i<57600 for i in indices) or indices!=sorted(set(indices)):
        raise ValueError('sorted unique original five-frame ray indices required')
    return indices


def restore_indices(entries,indices):
    validate_indices(indices);out=[]
    for e in entries:
        i=e['ray_index']
        if type(i) is not int or not 0<=i<len(indices):raise ValueError('local ray index outside request')
        out.append(dict(e,query_row_index=i,ray_index=indices[i]))
    return out


def write_gzip_json(output,value):
    """Stream JSON chunks without retaining whole text and compressed copies."""
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            for chunk in json.JSONEncoder(allow_nan=False).iterencode(value):
                zipped.write(chunk.encode('utf-8'))


def load_archive(root,scope):
    if any(n=='mtare_topo' or n.startswith('mtare_topo.') for n in sys.modules):raise ValueError('fresh isolated process required')
    archive=root/scope['archive']['path']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=scope['archive']['sha256']:raise ValueError('archive drift')
    sys.path.insert(0,str(archive)+'/src')
    modules={k:importlib.import_module('mtare_topo.'+v) for k,v in dict(
        construction='data.primitive_relation_materialization',sensor='data.cano_sensor_smoke',
        pack='teacher.gse_caster_cap_replay_v1',mesh='teacher.csg_mesh_provenance',
        entries='teacher.gse_observed_operand_entries_v1').items()}
    for name,m in sys.modules.items():
        if name=='mtare_topo' or name.startswith('mtare_topo.'):
            if not str(getattr(m,'__file__','')).startswith(str(archive)+'/src/'):raise ValueError('mixed project source '+name)
    return modules


def run(root,scope,index,output,thread_execution=None):
    import io
    import numpy as np
    e=scope['entries'][index];ids=validate_indices(e['original_ray_indices'])
    if len(ids)!=e['ray_count']:raise ValueError('ray count drift')
    modules=load_archive(root,scope);files={}
    for p,h in e['files'].items():
        raw=(root/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift '+p)
        if p.endswith('.npz'):
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:files[p]={k:z[k].copy() for k in z.files}
        else:files[p]=json.loads(raw)
    student=next(v for p,v in files.items() if '/student/' in p)
    sensor=next(v for p,v in files.items() if '/source_evidence/' in p and p.endswith('.npz'))
    construction=next(v for p,v in files.items() if p.endswith('_constructions.json'))
    book=next(v for p,v in files.items() if p.endswith('_codebooks.json'))
    _,primitives=modules['construction'].load_p1a_realized_construction(construction)
    names=[p.primitive_id for p in primitives]
    if names!=book['primitive_ids']:raise ValueError('operand order drift')
    settings=scope['geometry_settings']
    meshes=[modules['mesh'].mesh_swept_superellipse(p,axial_spacing_m=settings['axial_spacing_m'],angular_segments=settings['angular_segments']) for p in primitives]
    caster=modules['mesh'].CSGMeshProvenanceRaycaster(meshes)
    local=modules['sensor'].lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions=np.concatenate([modules['sensor'].world_directions(local,float(y)) for y in sensor['yaw_deg']])
    _,packed=modules['pack'].pack_caster_inputs(np.empty((0,3)),np.repeat(sensor['sensor_xyz_m'],11520,axis=0),directions)
    owners=[[names[i] for i in book['source_sets'][int(code)]] for code in sensor['primitive_membership_code'].reshape(-1)]
    ix=np.array(ids,dtype=np.int64)
    result=modules['entries'].observed_operand_entries(caster,packed[ix],
        first_return=student['ranges_m'].reshape(-1)[ix],valid=student['valid_mask'].reshape(-1)[ix].astype(bool),
        return_sources=[owners[i] for i in ids],center_m=sensor['sensor_xyz_m'][-1])
    entries=restore_indices(result['entries'],ids)
    recovered={e['ray_index'] for e in entries}
    response=dict(status='RECOVERY_ONLY_NOT_LABEL_QUALIFICATION',case=index,archive_sha256=scope['archive']['sha256'],
        entries=entries,queried_rays=ids,missing_entry_ray_indices=sorted(set(ids)-recovered),
        rejection_detail='Original filter returns accepted entries only; missing may be filtered or no hit, not evidence of absence',
        target_producer_calls=0,new_labels=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if thread_execution is not None:
        response['thread_execution'] = dict(thread_execution)
        if thread_execution['scene_builds'] != 1 or thread_execution['intersection_queries'] != 1:
            raise ValueError('unexpected archived execution path')
    # Geometry is no longer needed; release the scene before serialization.
    del caster,meshes,primitives,files
    import gc
    gc.collect()
    write_gzip_json(output,response)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scope',type=Path,required=True)
    p.add_argument('--index',type=int);p.add_argument('--output',type=Path);p.add_argument('--check-imports',action='store_true');p.add_argument('--single-thread',action='store_true');a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    scope=json.loads(a.scope.read_text())
    if a.check_imports:
        load_archive(a.root,scope);print('ARCHIVE_IMPORT_ONLY_NO_GEOMETRY_OR_DATA')
    else:
        if a.index not in range(3) or a.output is None:raise ValueError('one of exact3 cases and owned output required')
        if a.single_thread:
            import open3d
            with bounded_scene_threads(open3d,1) as execution:
                run(a.root,scope,a.index,a.output,execution)
        else:
            run(a.root,scope,a.index,a.output)
