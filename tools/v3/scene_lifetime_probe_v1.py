"""Repeat frozen caster construction on synthetic closed tubes only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import weakref
from v8_teacher_snapshot_probe_v1 import load_teacher


def memory():
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:','VmRSS:','VmPeak:','VmHWM:','Threads:')):
            k,v=line.split(':',1);result[k]=int(v.strip().split()[0])
    no_access=0
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split();a,b=[int(x,16) for x in fields[0].split('-')]
        if fields[1].startswith('---'):no_access+=b-a
    result['no_access_mapping_bytes']=no_access
    return result


def main(cleanup):
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    _,_,_,manifest=load_teacher()
    import numpy as np
    import open3d as o3d
    from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster,mesh_swept_superellipse
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
    stages=[dict(stage='imports',memory=memory())];hashes=[]
    rays=np.tile(np.array([-20.,.123,.234,1.,0.,0.],np.float32),(57600,1))
    for iteration in range(3):
        stages.append(dict(stage='before_mesh',iteration=iteration,memory=memory()))
        meshes=[mesh_swept_superellipse(SweptSuperellipsePrimitive(str(i),
            np.array([[-15.,6.*i,0.],[15.,6.*i,0.]]),((2.,2.),(2.,2.)),(2.,2.)),
            axial_spacing_m=.05,angular_segments=64) for i in range(8)]
        triangles=sum(len(m.triangle_vertex_indices) for m in meshes)
        stages.append(dict(stage='meshes',iteration=iteration,memory=memory(),triangles=triangles))
        caster=CSGMeshProvenanceRaycaster(meshes);ref=weakref.ref(caster)
        stages.append(dict(stage='scene_added_before_query',iteration=iteration,memory=memory()))
        hits=caster.scene.list_intersections(o3d.core.Tensor(rays))
        hashes.append({k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in hits.items()})
        stages.append(dict(stage='after_query',iteration=iteration,memory=memory()))
        del hits,caster,meshes
        stages.append(dict(stage='after_del',iteration=iteration,memory=memory(),caster_alive=ref() is not None))
        if cleanup=='gc':
            collected=gc.collect()
            stages.append(dict(stage='after_gc',iteration=iteration,memory=memory(),collected=collected,caster_alive=ref() is not None))
    assert all(h==hashes[0] for h in hashes)
    print(json.dumps(dict(status='SYNTHETIC_SCENE_LIFETIME_COMPLETE',cleanup=cleanup,
        archive_sha256=manifest['archive_sha256'],stages=stages,output_sha256=hashes,
        repeat_output_equal=True,real_data_reads=0,teacher_calls=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cleanup',choices=['none','gc'],required=True)
    main(p.parse_args().cleanup)
