"""Synthetic box only: allocator arenas versus Open3D ray output, no dataset."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    rays=o3d.core.Tensor(np.tile(np.array([[-1.,.5,.5,1.,0.,0.]],dtype=np.float32),(57600,1)))
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    rows=[]
    for index in range(6):
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
        hits=scene.list_intersections(rays)
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in hits.items()}
        memory={}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmSize:','VmRSS:','VmPeak:','Threads:')):
                key,value=line.split(':',1);memory[key]=int(value.strip().split()[0])
        rows.append(dict(index=index,output_sha256=hashes,memory_kib_except_threads=memory))
        del scene,hits
    print(json.dumps(dict(allocator_arena_max=os.environ.get('MALLOC_ARENA_MAX'),rows=rows,
        dataset_reads=0,teacher_calls=0,model_calls=0)))


if __name__=='__main__':main()
