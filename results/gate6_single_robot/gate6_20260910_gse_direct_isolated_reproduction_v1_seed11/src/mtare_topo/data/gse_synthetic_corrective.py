"""Versioned missing-ray-only derivation; caller owns exact export authority."""
from copy import deepcopy
import numpy as np
from .gse_surface_teacher_reader_v1 import verify_alignment
from .gse_synthetic_matrix import construction_document,matrix
from .gse_structure_review_v1 import canonical_sha

CASES={'hidden_branch__ellipse__view3':1,'hidden_branch__rounded_rectangle__view1':1,
       'hidden_branch__rounded_rectangle__view2':10,'hidden_branch__rounded_rectangle__view3':8}
REVISION='interval_winding_missing_returns_v1'


def restore_declared_case(saved_case):
    candidates=[c for c in matrix() if c['case_id']==saved_case.get('case_id')]
    if len(candidates)!=1 or canonical_sha(candidates[0])!=canonical_sha(saved_case):
        raise ValueError('archived case differs from fixed declaration')
    # JSON sort_keys preserves mapping meaning but not insertion order. The
    # constructor turns anchor insertion order into a list, so recover the
    # original registered order only after exact content equality.
    return candidates[0]


def derive_corrective(case,bundle,*,repair=None):
    case=restore_declared_case(case)
    key=case['case_id']
    if key not in CASES or bundle['source'].get('case_id')!=key:raise ValueError('exact corrective case required')
    if 'derived_observation_revision' in bundle['source']:raise ValueError('no repeat derivation')
    if canonical_sha(construction_document(case))!=canonical_sha(bundle['construction_teacher_only']):
        raise ValueError('case construction mismatch')
    student=bundle['student'];sensor=bundle['sensor_teacher_only']
    if not np.array_equal(sensor['sensor_xyz_m'],np.asarray(case['poses_world_m'])) or not np.array_equal(sensor['yaw_deg'],np.asarray(case['yaw_deg'])):
        raise ValueError('case pose mismatch')
    verify_alignment(sensor,student,bundle['construction_teacher_only'],bundle['codebook_teacher_only'],bundle['source'])
    if student['ranges_m'].shape!=(5,16,720) or student['ranges_m'].dtype!=np.float32 or student['valid_mask'].dtype!=np.uint8:
        raise ValueError('original scan shape/type drift')
    invalid=student['valid_mask']==0
    if int(invalid.sum())!=CASES[key]:raise ValueError('fixed invalid-ray population drift')
    if repair is None:
        from .gse_synthetic_sensor import SyntheticSensorScene
        from .cano_sensor_smoke import lidar_local_directions,world_directions
        from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster
        scene=SyntheticSensorScene(case)
        caster=CSGMeshProvenanceRaycaster(scene.caster.meshes,rescue_missing_with_interval_winding=True)
        local=lidar_local_directions().reshape(-1,3).astype(np.float64)
        def repair(index):
            f,r,c=index;origin=sensor['sensor_xyz_m'][f]
            direction=world_directions(local,float(sensor['yaw_deg'][f]))[r*720+c]
            inside=scene.field.operand_signed_distances_sparse(origin[None])[0]<=1e-9
            return caster.ray_exit_hits(origin[None],direction[None],inside[None])[0]
    output=deepcopy(bundle);out=output['student'];osensor=output['sensor_teacher_only'];book=output['codebook_teacher_only']
    rows=[]
    from .cano_sensor_smoke import NEAR_RANGE_M,MAX_RANGE_M
    for index in map(tuple,np.argwhere(invalid)):
        hit=repair(index)
        if hit is None or not np.isfinite(hit.distance_m) or not NEAR_RANGE_M<=hit.distance_m<=MAX_RANGE_M:
            raise ValueError('corrective return unqualified')
        if not hit.source_primitive_ids or any(x not in book['primitive_ids'] for x in hit.source_primitive_ids):
            raise ValueError('corrective source unqualified')
        sources=sorted(set(book['primitive_ids'].index(x) for x in hit.source_primitive_ids))
        if sources not in book['source_sets']:book['source_sets'].append(sources)
        code=book['source_sets'].index(sources)
        if code>np.iinfo(np.uint16).max:raise ValueError('codebook capacity')
        out['ranges_m'][index]=np.float32(hit.distance_m);out['valid_mask'][index]=1
        osensor['primitive_membership_code'][index]=code
        rows.append(dict(index=[int(x) for x in index],old_range_m=float(student['ranges_m'][index]),
            new_range_m=float(out['ranges_m'][index]),source_ids=list(hit.source_primitive_ids)))
    for k in ('ranges_m','valid_mask'):
        if not np.array_equal(out[k][~invalid],student[k][~invalid]):raise ValueError('valid ray changed')
    for k in student:
        if k not in ('ranges_m','valid_mask') and not np.array_equal(out[k],student[k]):raise ValueError('motion changed')
    if not np.array_equal(osensor['primitive_membership_code'][~invalid],sensor['primitive_membership_code'][~invalid]):
        raise ValueError('valid source changed')
    output['source']['derived_observation_revision']=REVISION
    output['derivation_provenance']=dict(revision=REVISION,changed_rays=rows,
        unchanged_ray_positions=int((~invalid).sum()),original_observation_unchanged_claimed=False)
    verify_alignment(osensor,out,output['construction_teacher_only'],book,output['source'])
    return output
