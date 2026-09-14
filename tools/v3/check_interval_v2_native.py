"""Native four-case software diagnostic; no dataset, label export or pytest."""
import json
import resource
import time
import numpy as np
import open3d as o3d
from mtare_topo.teacher.csg_mesh_provenance import ClosedPrimitiveMesh, CSGMeshProvenanceRaycaster
from mtare_topo.teacher.csg_interval_raycaster_v2 import CSGIntervalRaycasterV2


def box(identity,lo,hi):
    m=o3d.geometry.TriangleMesh.create_box(width=hi-lo,height=2.,depth=2.)
    m.translate([lo,-1.,-1.]);m.compute_triangle_normals()
    return ClosedPrimitiveMesh(identity,np.asarray(m.vertices),np.asarray(m.triangles),np.asarray(m.triangle_normals))


def main():
    start=time.monotonic();rows=[]
    for entry in [2.0001,2.006755640983918898,2.02,1.5]:
        meshes=[box('main',-1.,2.),box('branch',entry,30.)]
        args=([[0.,.13,.17]],[[1.,0.,0.]],[[True,False]])
        old=CSGMeshProvenanceRaycaster(meshes).ray_exit_hits(*args)[0]
        new=CSGIntervalRaycasterV2(meshes).ray_exit_hits(*args)[0]
        expected=30. if entry<2. else 2.
        assert new is not None and abs(new.distance_m-expected)<1e-5
        expected_old=30. if entry<2.01 else 2.
        assert old is not None and abs(old.distance_m-expected_old)<1e-5
        rows.append({'branch_entry_m':entry,'expected_m':expected,'old_m':old.distance_m,'v2_m':new.distance_m})
    print(json.dumps({'open3d':o3d.__version__,'numpy':np.__version__,'cases':rows,
        'elapsed_s':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'native_box_software_only_not_archived_double_ray','training_steps':0},indent=2))


if __name__=='__main__':main()
