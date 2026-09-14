"""Fixed recorded observation with a source-checked, unseen operand removed."""
from copy import deepcopy
import numpy as np
from .gse_synthetic_matrix import construction_document
from .gse_structure_review_v1 import canonical_sha
from .gse_surface_teacher_reader_v1 import verify_alignment
from .cano_sensor_smoke import lidar_local_directions,world_directions
from .primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.evaluation.gse_recorded_segment_exclusion import recorded_segment_exclusion


def remove_unobserved_hidden_operand(case,bundle):
    if (bundle['source'].get('case_id')!=case['case_id']
        or not np.array_equal(bundle['sensor_teacher_only']['sensor_xyz_m'],np.asarray(case['poses_world_m']))
        or not np.array_equal(bundle['sensor_teacher_only']['yaw_deg'],np.asarray(case['yaw_deg']))):
        raise ValueError('exact declared counterfactual observation required')
    expected=construction_document(case)
    if canonical_sha(expected)!=canonical_sha(bundle['construction_teacher_only']):
        raise ValueError('primary construction mismatch')
    control=construction_document(case,hidden_control=True)
    verify_alignment(bundle['sensor_teacher_only'],bundle['student'],expected,bundle['codebook_teacher_only'],bundle['source'])
    _,primitives=load_p1a_realized_construction(expected)
    removed=[p for p in primitives if p.primitive_id=='hidden']
    if len(removed)!=1:raise ValueError('exact removed operand required')
    mesh=mesh_swept_superellipse(removed[0],axial_spacing_m=.05,angular_segments=64)
    sensor=bundle['sensor_teacher_only'];local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    proof=recorded_segment_exclusion(mesh.vertices_xyz_m,np.repeat(sensor['sensor_xyz_m'],len(local),axis=0),
        directions,bundle['student']['ranges_m'].reshape(-1),bundle['student']['valid_mask'].reshape(-1).astype(bool))
    if not proof['all_recorded_segments_excluded']:raise ValueError('removed operand not excluded from every recorded segment')
    old=bundle['codebook_teacher_only'];codes=sensor['primitive_membership_code'];active=np.unique(codes)
    ids=[p['primitive_id'] for p in control['realized_primitives']]
    new_sets=[[]];mapping={0:0}
    for code in active:
        if code==0:continue
        source_ids=[old['primitive_ids'][i] for i in old['source_sets'][int(code)]]
        if any(s not in ids for s in source_ids):raise ValueError('removed operand supplied a recorded return')
        values=sorted(ids.index(s) for s in source_ids)
        if values not in new_sets:new_sets.append(values)
        mapping[int(code)]=new_sets.index(values)
    output=deepcopy(bundle);output['construction_teacher_only']=control
    output['codebook_teacher_only']={**old,'primitive_ids':ids,'source_sets':new_sets}
    lookup=np.zeros(len(old['source_sets']),dtype=np.uint16)
    for code,value in mapping.items():lookup[code]=value
    output['sensor_teacher_only']['primitive_membership_code']=lookup[codes]
    output['source']['hidden_control']=True
    for k,v in bundle['student'].items():
        if not np.array_equal(v,output['student'][k]):raise ValueError('student observation must stay bitwise unchanged')
    verify_alignment(output['sensor_teacher_only'],output['student'],control,output['codebook_teacher_only'],output['source'])
    output['counterfactual_provenance']=dict(mode='fixed_recorded_observation_removed_segment_excluded_operand',
        original_construction_sha256=canonical_sha(expected),control_construction_sha256=canonical_sha(control),
        removed_primitive='hidden',segments_checked=len(proof['separated']),
        minimum_separating_axis_gap_m=proof['minimum_separating_axis_gap_m'],
        separate_renders_bitwise_equal_claimed=False,physical_sensor_noise_certified=False)
    return output
