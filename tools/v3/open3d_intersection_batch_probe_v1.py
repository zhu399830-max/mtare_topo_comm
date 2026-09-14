"""Fixed synthetic all-intersection comparison, no real scan or teacher IO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main(batch_size):
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    import numpy as np
    import open3d as o3d
    if batch_size not in (57600,1152):raise ValueError('two fixed comparison schedules only')
    rng=np.random.default_rng(20260906)
    rays=np.zeros((57600,6),np.float32)
    rays[:,0]=-1.;rays[:,1:3]=rng.uniform(.05,.95,(57600,2));rays[:,3]=1.
    # Empty, reversed and inside rays are deliberately retained.
    rays[:16,1]=2.;rays[16:32,0]=34.;rays[16:32,3]=-1.;rays[32:48,0]=.5
    scene=o3d.t.geometry.RaycastingScene()
    for i in range(16):
        mesh=o3d.geometry.TriangleMesh.create_box()
        mesh.translate((2.*i,0.,0.))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    digests={};counts=np.zeros(len(rays),np.uint64);peak_arrays=0;total_hits=0
    start=time.monotonic()
    for offset in range(0,len(rays),batch_size):
        batch=rays[offset:offset+batch_size]
        raw={k:v.numpy() for k,v in scene.list_intersections(o3d.core.Tensor(batch)).items()}
        peak_arrays=max(peak_arrays,sum(v.nbytes for v in raw.values()))
        local=raw['ray_ids'].astype(np.int64)
        per_ray=np.bincount(local,minlength=len(batch)).astype(np.uint64)
        assert np.array_equal(np.diff(raw['ray_splits']).astype(np.uint64),per_ray)
        counts[offset:offset+len(batch)]=per_ray
        total_hits+=len(local)
        raw['ray_ids']=local+offset
        order=np.lexsort((raw['primitive_ids'],raw['geometry_ids'],raw['t_hit'],raw['ray_ids']))
        for k,v in sorted(raw.items()):
            if k=='ray_splits':continue
            digests.setdefault(k,hashlib.sha256()).update(v[order].tobytes())
        del raw,order,local
    memory={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmSize:','VmRSS:','VmPeak:','VmHWM:','Threads:')):
            key,value=line.split(':',1);memory[key]=int(value.strip().split()[0])
    print(json.dumps(dict(status='SYNTHETIC_ALL_INTERSECTIONS_COMPLETE',batch_size=batch_size,
        rays=len(rays),boxes=16,total_hits=total_hits,output_sha256={k:h.hexdigest() for k,h in digests.items()},
        per_ray_counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(rays.tobytes()).hexdigest(),peak_returned_array_bytes=peak_arrays,
        elapsed_s=time.monotonic()-start,memory_kib_except_threads=memory,
        allocator_arena_max=os.environ.get('MALLOC_ARENA_MAX'),address_space_cap_bytes=3*1024**3,
        real_data_reads=0,teacher_calls=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--batch-size',type=int,required=True)
    main(p.parse_args().batch_size)
