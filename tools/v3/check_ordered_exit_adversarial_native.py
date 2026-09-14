"""Eight fixed native geometry fixtures; candidate status is not qualification."""
import json
import numpy as np
import open3d as o3d
from check_interval_v2_native import box
from mtare_topo.teacher.csg_mesh_provenance import ClosedPrimitiveMesh,CSGMeshProvenanceRaycaster
from mtare_topo.teacher.csg_interval_raycaster_v2 import verified_interval_hit
from mtare_topo.teacher.ordered_exit_candidate import ordered_exit_candidate


def shells(parts):
    vertices=[];indices=[];normals=[];offset=0
    for part in parts:
        vertices.append(part.vertices_xyz_m)
        indices.append(part.triangle_vertex_indices+offset)
        normals.append(part.triangle_normals);offset+=len(part.vertices_xyz_m)
    return ClosedPrimitiveMesh('shells',np.concatenate(vertices),np.concatenate(indices),np.concatenate(normals))


def main():
    base=box('main',-1.,2.)
    fixtures=[
        ('gap',[base,box('branch',2.00675564,30.)],[1,0],[0,.13,.17],2.),
        ('overlap',[base,box('branch',1.5,30.)],[1,0],[0,.13,.17],30.),
        ('separated',[base,box('branch',3.,30.)],[1,0],[0,.13,.17],2.),
        ('contact',[base,box('branch',2.,30.)],[1,0],[0,.13,.17],30.),
        ('coincident_sources',[base,box('other',-1.,2.)],[1,1],[0,.13,.17],2.),
        ('internal_shell',[shells([base,box('nested',.5,1.)])],[1],[0,.13,.17],2.),
        ('duplicate_shell',[shells([base,base])],[1],[0,.13,.17],2.),
        ('surface_origin',[base],[1],[0,1.,1.],None),
    ]
    rows=[]
    for name,meshes,inside,origin,expected in fixtures:
        direction=np.array([1.,0.,0.]);caster=CSGMeshProvenanceRaycaster(meshes)
        raw={k:v.numpy() for k,v in caster.scene.list_intersections(
            o3d.core.Tensor(np.array([list(origin)+direction.tolist()],dtype=np.float32))).items()}
        ids=[caster.geometry_to_operand[int(g)] for g in raw['geometry_ids']]
        dots=[float(meshes[i].triangle_normals[int(t)]@direction) for i,t in zip(ids,raw['primitive_ids'])]
        fast=ordered_exit_candidate(raw['t_hit'],ids,dots,inside)
        slow=verified_interval_hit(meshes,origin,direction,inside,raw['t_hit'],ids)
        rows.append(dict(case=name,crossings=len(ids),fast_status=fast.status,fast_m=fast.distance_m,
                         reason=fast.reason,reference_m=None if slow is None else slow.distance_m,
                         expected_m=expected))
    print(json.dumps({'open3d':o3d.__version__,'cases':rows,'scope':'software_not_dataset'},indent=2),flush=True)
    for row in rows:
        if row['expected_m'] is None:
            assert row['reference_m'] is None,row
        else:
            assert row['reference_m'] is not None and abs(row['reference_m']-row['expected_m'])<1e-5,row
            if row['fast_status']=='candidate':
                assert row['fast_m']==row['reference_m'],row
    for name in ('contact','internal_shell','duplicate_shell'):
        assert next(r for r in rows if r['case']==name)['fast_status']=='needs_reference'


if __name__=='__main__':main()
