"""Pinned declared double-junction ray, native meshes; no archive/export."""
import json
import resource
import time
import numpy as np
import open3d as o3d
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse,CSGMeshProvenanceRaycaster
from mtare_topo.teacher.csg_interval_raycaster_v2 import CSGIntervalRaycasterV2
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit
from mtare_topo.teacher.ordered_exit_candidate import ordered_exit_candidate


def main():
    started=time.monotonic()
    case=next(c for c in matrix() if c['case_id']=='double_junction__ellipse__view3')
    origin=np.array([[1.9000000000000001,0.,.04]])
    direction=np.array([[.05232798574151599,-.9984774386302507,.017452405410415962]])
    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    expected,supported,intervals=declared_union_exit(case,origin,direction)
    _,primitives=load_p1a_realized_construction(construction_document(case))
    field=SweptSuperellipseProvenanceField(primitives,spacing_m=.025)
    inside=field.operand_signed_distances_sparse(origin)<=1e-9
    assert supported[0]
    np.testing.assert_array_equal(inside,(intervals[:,:,0]<0)&(intervals[:,:,1]>0))
    meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
    built=time.monotonic()
    old_caster=CSGMeshProvenanceRaycaster(meshes)
    old=old_caster.ray_exit_hits(origin,direction,inside)[0]
    old_done=time.monotonic()
    new=CSGIntervalRaycasterV2(meshes).ray_exit_hits(origin,direction,inside)[0]
    new_done=time.monotonic()
    raw={k:v.numpy() for k,v in old_caster.scene.list_intersections(
        o3d.core.Tensor(np.concatenate((origin,direction),axis=1).astype(np.float32))).items()}
    ids=[old_caster.geometry_to_operand[int(g)] for g in raw['geometry_ids']]
    dots=[float(meshes[i].triangle_normals[int(t)]@direction[0])
          for i,t in zip(ids,raw['primitive_ids'])]
    candidate_started=time.monotonic()
    fast=ordered_exit_candidate(raw['t_hit'],ids,dots,inside[0])
    candidate_s=time.monotonic()-candidate_started
    result={'case_id':case['case_id'],'frame':3,'ray':6306,'open3d':o3d.__version__,
        'origin':origin.tolist(),'direction':direction.tolist(),'expected_m':float(expected[0]),
        'old_m':None if old is None else old.distance_m,'v2_m':None if new is None else new.distance_m,
        'triangles':sum(len(m.triangle_vertex_indices) for m in meshes),
        'build_s':built-started,'old_scene_and_ray_s':old_done-built,'v2_scene_and_ray_s':new_done-old_done,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'fast_status':fast.status,'fast_m':fast.distance_m,'fast_candidate_s':candidate_s,
        'native_crossings':len(raw['t_hit']),
        'archive_read':False,'labels_exported':0,'optimizer_steps':0}
    print(json.dumps(result,indent=2),flush=True)
    assert old is not None and old.distance_m>30.
    assert new is not None and abs(new.distance_m-expected[0])<1e-5
    assert fast.status=='candidate' and fast.distance_m==new.distance_m


if __name__=='__main__':main()
