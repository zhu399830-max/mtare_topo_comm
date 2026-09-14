"""Fixed original hidden circle view0; independent geometric exclusion only."""
import json
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_sensor import SyntheticSensorScene
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
from mtare_topo.evaluation.gse_recorded_segment_exclusion import recorded_segment_exclusion

case=next(c for c in matrix() if c['case_id']=='hidden_branch__circle__view0')
scene=SyntheticSensorScene(case);bundle=scene.render(case)
mesh=next(m for m in scene.caster.meshes if m.primitive_id=='hidden')
sensor=bundle['sensor_teacher_only'];local=lidar_local_directions().reshape(-1,3).astype(np.float64)
directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
directions/=np.linalg.norm(directions,axis=1)[:,None]
origins=np.repeat(sensor['sensor_xyz_m'],len(local),axis=0)
result=recorded_segment_exclusion(mesh.vertices_xyz_m,origins,directions,
    bundle['student']['ranges_m'].reshape(-1),bundle['student']['valid_mask'].reshape(-1).astype(bool))
print(json.dumps({**{k:v for k,v in result.items() if k not in ('separated','unproved')},
    'separated_segments':int(result['separated'].sum()),'unproved_segments':int(result['unproved'].sum())}),flush=True)
assert result['all_recorded_segments_excluded']
print('HIDDEN_RECORDED_SEGMENT_EXCLUSION_PASS_NOT_FULL_EQUIVALENCE')
