"""Read-only source-parameter local sensitivity, not label qualification.

Uses reference plane, axes and exponent at the reference arc. Local section
approximation differs from the actual triangulated varying sweep; residual is
reported. No noise model or global uniqueness assertion. No output writes.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_surface_teacher_reader_v1 import SurfaceTeacherReader
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.gse_superellipse_constraints_v1 import local_constraints, finite_center_probe
from mtare_topo.teacher.gse_center_profile_v1 import profile_center


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
        _, primitives=load_p1a_realized_construction(bundle['construction_teacher_only'])
        primitive_by_id={p.primitive_id:p for p in primitives}
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
            primitive=primitive_by_id[proposal['primitive_id_teacher_only']]
            length=np.linalg.norm(np.diff(primitive.centerline_xyz_m,axis=0),axis=1).sum()
            axes, exponent=primitive.parameters_at_fraction(proposal['reference_arc_m']/length)
            sensitivity=local_constraints(uv,center_uv_m=[0.,0.],half_axes_m=axes,exponent=float(exponent))
            result=dict(task=row['task'],primitive=proposal['primitive_id_teacher_only'],points=len(uv),
                across_horizontal=bool(uv[:,0].min()<0<uv[:,0].max()),
                across_vertical=bool(uv[:,1].min()<0<uv[:,1].max()),exponent=float(exponent),half_axes_m=axes.tolist(),
                center_singular_values=sensitivity['center_profile_singular_values'].tolist(),
                nuisance_rank=sensitivity['nuisance_rank'],
                local_reference_residual_rms=float(np.sqrt(np.mean(sensitivity['residual']**2))),
                finite_center_probe=finite_center_probe(uv,center_uv_m=[0.,0.],half_axes_m=axes,exponent=float(exponent)))
            if '--profile-fixed-centers' in sys.argv[1:]:
                result['nonlinear_center_profile']=profile_center(uv,half_axes_m=axes,exponent=float(exponent))
            results.append(result)
    for p,h in reader.opened.items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h:raise ValueError('source changed')
    sensitivities=[r['center_singular_values'][-1] for r in results]
    print(json.dumps(dict(diagnostic_only=True,known_reference_plane=True,source_owned_support=True,
        results=results,total_candidates=len(results),
        min_center_sensitivity_quantiles=np.quantile(sensitivities,[0,.1,.5,1]).tolist(),
        source_files_reverified=len(reader.opened),labels=0),ensure_ascii=False))


if __name__=='__main__':main()
