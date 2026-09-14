"""Read archived invalid rays; inspect intersections without exporting labels."""
import _bootstrap
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from mtare_topo.data.gse_synthetic_sensor import SyntheticSensorScene
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'

def main(winding=False,rescue=False):
    pins=dict(line.split('  ',1)[::-1] for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())
    def checked(p):
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[str(p.relative_to(ROOT))]:raise ValueError('archived evidence drift')
        return b
    local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    count=0;rescued_count=0
    for p in sorted((RUN/'artifacts').glob('hidden_branch__*.npz')):
        checked(p)
        with np.load(p) as z:
            invalid=np.argwhere(z['valid_mask']==0)
            if not len(invalid):continue
            data=json.loads(gzip.decompress(checked(p.with_suffix('.json.gz'))))
            case=data['case'];scene=SyntheticSensorScene(case)
            rescue_caster=None
            if rescue:
                from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster
                rescue_caster=CSGMeshProvenanceRaycaster(scene.caster.meshes,rescue_missing_with_interval_winding=True)
            for frame,row,col in invalid:
                origin=z['sensor_xyz_m'][frame];direction=world_directions(local,float(z['yaw_deg'][frame]))[row*720+col]
                direction=direction/np.linalg.norm(direction)
                inside=scene.field.operand_signed_distances_sparse(origin[None])[0]<=1e-9
                ray=np.r_[origin,direction].astype(np.float32)[None]
                raw={k:v.numpy() for k,v in scene.caster.scene.list_intersections(o3d.core.Tensor(ray)).items()}
                hits=[]
                for i in np.argsort(raw['t_hit'],kind='stable'):
                    op=scene.caster.geometry_to_operand[int(raw['geometry_ids'][i])];tri=int(raw['primitive_ids'][i])
                    hits.append(dict(t=float(raw['t_hit'][i]),operand=scene.ids[op],triangle=tri,
                        normal_dot=float(scene.caster.meshes[op].triangle_normals[tri]@direction)))
                replay=scene.caster.ray_exit_hits(origin[None],direction[None],inside[None])[0]
                extra={}
                if rescue:
                    corrected=rescue_caster.ray_exit_hits(origin[None],direction[None],inside[None])[0]
                    expected=max(h['t'] for h in hits)
                    extra.update(rescued_exit=None if corrected is None else corrected.distance_m,
                        matches_previously_diagnosed_final_boundary=corrected is not None and corrected.distance_m==expected)
                    if corrected is not None and corrected.distance_m==expected:rescued_count+=1
                if winding:
                    from mtare_topo.evaluation.mesh_winding_diagnostic import oriented_winding
                    distances=np.unique([0.]+[h['t'] for h in hits]+[50.])
                    mids=(distances[:-1]+distances[1:])/2
                    # Both original and float32-exported meshes. Midpoints are
                    # defined by actual raw intersections, not new offsets.
                    values={}
                    for precision in ('source64','packed32'):
                        mesh=scene.caster.meshes[0]
                        vertices=mesh.vertices_xyz_m
                        qorigin=origin;qdir=direction
                        if precision=='packed32':
                            vertices=vertices.astype(np.float32).astype(np.float64)
                            qorigin=ray[0,:3].astype(np.float64);qdir=ray[0,3:].astype(np.float64)
                        points=qorigin+mids[:,None]*qdir
                        values[precision]=oriented_winding(vertices[mesh.triangle_vertex_indices],points).tolist()
                    extra.update(interval_midpoints_m=mids.tolist(),approach_winding=values)
                print(json.dumps(dict(case_id=case['case_id'],index=[int(frame),int(row),int(col)],
                    origin=origin.tolist(),direction=direction.tolist(),inside=inside.tolist(),
                    intersections=hits,recomputed_exit=None if replay is None else replay.distance_m,**extra)),flush=True)
                count+=1
    if count!=20:raise ValueError('fixed invalid population drift')
    print(json.dumps(dict(archived_invalid_rays=count,new_labels=0,full_frame_rerenders=0,
        rescue_enabled=rescue,rescued_final_boundaries=rescued_count)))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--winding',action='store_true')
    p.add_argument('--rescue',action='store_true');args=p.parse_args()
    main(args.winding,args.rescue)
