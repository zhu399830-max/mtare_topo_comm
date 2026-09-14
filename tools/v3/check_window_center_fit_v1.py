"""Read-only optimistic ellipse-center diagnostic, NOT a deployment baseline.

Uses fixed30 source-owned support points and the reference section plane.
Ellipse mismatch on superelliptic/variable sections is not teacher invalidity.
No files, labels, candidate selection or acceptance gates are changed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_surface_teacher_reader_v1 import SurfaceTeacherReader
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions


def fit_center(points_uv_m):
    xy=np.asarray(points_uv_m,dtype=np.float64)/10.
    x,y=xy.T
    design=np.column_stack((x*x,x*y,y*y,x,y))
    coefficients,_,rank,singular=np.linalg.lstsq(design,np.ones(len(x)),rcond=None)
    if rank<5:return dict(status='RANK_DEFICIENT',rank=int(rank))
    a,b,c,d,e=coefficients
    matrix=np.array([[a,b/2],[b/2,c]])
    eigen=np.linalg.eigvalsh(matrix)
    if eigen[0]*eigen[1]<=0:return dict(status='NOT_ELLIPSE',rank=int(rank))
    center=-np.linalg.solve(2*matrix,np.array([d,e]))
    radius_term=1.+center@matrix@center
    if np.any(radius_term/eigen<=0):return dict(status='EMPTY_ELLIPSE',rank=int(rank))
    return dict(status='ELLIPSE',center_uv_m=(10*center).tolist(),
                reference_center_offset_m=float(10*np.linalg.norm(center)),
                design_condition=float(singular[0]/singular[-1]),
                algebraic_rms=float(np.sqrt(np.mean((design@coefficients-1.)**2))))


def main():
    root=PROJECT_ROOT
    run=root/'results/gate3_semantics/gate3_20260907_gse_surface_window_openings_v1_seed20260906'
    seal=(run/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!='6c0af39c3054b2ad597609cdbac7b119f773e2d026e12d4741d2c33c0df9b88a':
        raise ValueError('seal drift')
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def load(path):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pins[str(path.relative_to(root))]:raise ValueError('result drift')
        return json.loads(raw)
    card=load(run/'config/data_card.json');reader=SurfaceTeacherReader(root,card['scope']);results=[]
    for row in card['scope']['selected']:
        saved=load(run/'artifacts'/(row['view_id']+'.json'));bundle=reader.read_task(row['task'])
        if saved['source']!=bundle['source']:raise ValueError('source mismatch')
        sensor=bundle['sensor_teacher_only'];student=bundle['student']
        local=lidar_local_directions().reshape(-1,3).astype(np.float64)
        directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        points=np.repeat(sensor['sensor_xyz_m'],11520,axis=0)+directions*student['ranges_m'].reshape(-1,1)
        for proposal in saved['proposals']:
            if not proposal['proposal_supported_after_competition']:continue
            normal=np.array(proposal['reference_direction']);vertical=np.array([0.,0.,1.])
            vertical-=normal*(vertical@normal)
            if np.linalg.norm(vertical)==0:raise ValueError('vertical section frame undefined')
            vertical/=np.linalg.norm(vertical);horizontal=np.cross(vertical,normal)
            relative=points[proposal['surface_return_ray_indices']]-proposal['reference_position_m']
            uv=np.column_stack((relative@horizontal,relative@vertical))
            result=dict(task=row['task'],primitive=proposal['primitive_id_teacher_only'],points=len(uv),
                across_horizontal=bool(uv[:,0].min()<0<uv[:,0].max()),
                across_vertical=bool(uv[:,1].min()<0<uv[:,1].max()),**fit_center(uv))
            results.append(result)
    for p,h in reader.opened.items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h:raise ValueError('source changed')
    errors=[r['reference_center_offset_m'] for r in results if r['status']=='ELLIPSE']
    print(json.dumps(dict(diagnostic_only=True,known_reference_plane=True,source_owned_support=True,
        results=results,ellipse_count=len(errors),total_candidates=len(results),
        error_quantiles_m=np.quantile(errors,[0,.5,.9,1]).tolist() if errors else [],
        source_files_reverified=len(reader.opened),labels=0),ensure_ascii=False))


if __name__=='__main__':main()
