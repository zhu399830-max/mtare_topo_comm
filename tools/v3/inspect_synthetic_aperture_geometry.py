"""Geometry-only partition inspection of sealed unscored synthetic cases."""
import _bootstrap
from collections import Counter,defaultdict
import argparse,gzip,hashlib,io,json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_corrective import restore_declared_case
from mtare_topo.data.gse_synthetic_matrix import construction_document
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.aperture_polygon_boundary import straight_primitive_prism,classify_polygon_union
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere
from mtare_topo.teacher.aperture_component_contract import partition_boundary

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'


def main(with_support=False,summary_only=False,curved_only=False):
    pins={p:h for h,p in (l.split('  ',1) for l in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    def read(p):
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[str(p.relative_to(ROOT))]:raise ValueError('archived source drift')
        return b
    summary=json.loads(read(RUN/'metrics/summary.json'))
    keys=[x['case_id'] for x in summary['per_case'] if x['status']=='UNSCORED_NOT_PASS']
    if len(keys)!=99:raise ValueError('exact unscored population required')
    if curved_only:
        keys=[k for k in keys if k.split('__')[0] in ('ramp_connection','hidden_branch')]
        if len(keys)!=24:raise ValueError('exact curved subset required')
    v,f,a=octahedral_sphere(4);counts=defaultdict(Counter);skipped=Counter();readouts=defaultdict(Counter);same_component_references=[]
    cached_geometry=None;engine=None;mesh_seconds=0.;mesh_work=0
    for key in keys:
        data=json.loads(gzip.decompress(read(RUN/'artifacts'/(key+'.json.gz'))))
        case=restore_declared_case(data['case']);kind=case['program']['type']
        if canonical_sha(construction_document(case))!=canonical_sha(data['construction']):raise ValueError('source construction mismatch')
        _,primitives=load_p1a_realized_construction(data['construction'])
        unsupported=any(len(p.centerline_xyz_m)!=2 or p.endpoint_half_axes_m[0]!=p.endpoint_half_axes_m[1] or p.endpoint_shape_exponent[0]!=p.endpoint_shape_exponent[1] for p in primitives)
        if unsupported and not curved_only:skipped[kind]+=1;continue
        if curved_only:
            from mtare_topo.teacher.aperture_mesh_boundary import OriginalMeshBoundary
            from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
            geometry_hash=canonical_sha(data['construction'])
            if cached_geometry!=geometry_hash:
                engine=None
                engine=OriginalMeshBoundary([mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives])
                cached_geometry=geometry_hash
            result=engine.classify(v,f,center_m=case['poses_world_m'][-1]);state=result.state
            mesh_seconds+=result.elapsed_s;mesh_work+=result.triangle_point_evaluations
        else:
            prisms=[straight_primitive_prism(p) for p in primitives]
            state=classify_polygon_union(v,f,prisms,center_m=case['poses_world_m'][-1])
        partition=partition_boundary(a,state,np.full_like(state,-1))
        old=len(data['produced_targets']['record']['openings'])
        row=dict(case_id=key,geometry_free_components=len(partition.geometry_components),
            unresolved_pairs=len(partition.unresolved_component_pairs),unknown_cells=int((state==-1).sum()),
            old_reference_sections=old,observation_adapter_available=False,score_pass=None,
            mesh_distance_is_numerical_proxy=curved_only)
        if with_support:
            from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
            from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
            from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
            from mtare_topo.teacher.aperture_sparse_support import window_crossing_support
            with np.load(io.BytesIO(read(RUN/'artifacts'/(key+'.npz')))) as z:
                student={k:z[k] for k in ('ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg')}
                sensor={k:z[k] for k in ('sensor_xyz_m','yaw_deg','primitive_membership_code')}
                verify_alignment(sensor,student,data['construction'],data['codebook'],data['source'])
                local=lidar_local_directions().reshape(-1,3).astype(float)
                directions=np.concatenate([world_directions(local,float(y)) for y in sensor['yaw_deg']])
                directions/=np.linalg.norm(directions,axis=1)[:,None]
                indices=data['source']['frame_rows']
                rays=CausalRaySegments(np.repeat(sensor['sensor_xyz_m'],len(local),axis=0),directions,
                    student['ranges_m'].reshape(-1),student['valid_mask'].reshape(-1).astype(bool),
                    np.repeat(np.asarray(indices,dtype=int),len(local)),int(indices[-1]),0.)
                support=window_crossing_support(rays,v,f,state,partition,center_m=case['poses_world_m'][-1])
                row.update(observation_adapter_available=True,component_support_rays=support.component_ray_counts.tolist(),
                    components_with_sparse_support=int((support.component_ray_counts>0).sum()),
                    ambiguous_support_rays=support.ambiguous_crossings,
                    support_before_return_rays=int(support.crossing_before_return.sum()),
                    full_observation_qualification=False,extra_range_error_bound_m=0.,
                    error_scope='Stored synthetic render diagnostic with source dtype ULP; not physical sensor noise certification.')
                from mtare_topo.teacher.aperture_diagnostic_readout import diagnostic_readout,reference_seed_components
                refs=[o['position_m'] for o in data['produced_targets']['record']['openings']]
                mapping=reference_seed_components(refs,v,f,partition)
                row['diagnostic_readout']=diagnostic_readout(partition,support,mapping)
                from mtare_topo.teacher.aperture_field_eligibility import field_eligibility
                row['field_eligibility']=field_eligibility(row)
                out=row['diagnostic_readout'];readouts[kind][(out['supported_isolated_components'],out['supported_unresolved_sets'])]+=1
                if any(len(g['reference_seed_indices'])>1 for g in out['groups']):same_component_references.append(key)
        counts[kind][(row['geometry_free_components'],row['unresolved_pairs'],old)]+=1
        if not summary_only:print(json.dumps(row),flush=True)
    print(json.dumps(dict(inspected=len(keys),geometry_adapter_cases=sum(sum(x.values()) for x in counts.values()),
        unsupported_cases=dict(skipped),groups={k:{str(a):n for a,n in c.items()} for k,c in counts.items()},
        supported_readout_groups={k:{str(a):n for a,n in c.items()} for k,c in readouts.items()},
        multiple_reference_seeds_in_one_group=same_component_references,new_labels=0,full_frame_rerenders=0,
        mesh_classification_seconds=mesh_seconds,mesh_triangle_point_evaluations=mesh_work)))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--with-support',action='store_true');p.add_argument('--summary-only',action='store_true')
    p.add_argument('--curved-only',action='store_true')
    args=p.parse_args();main(args.with_support,args.summary_only,args.curved_only)
