"""One immutable CPU-only diagnostic on the sealed45 synthetic observations."""
import _bootstrap
import argparse, io, json, resource, shlex, signal, time, traceback, zipfile
from pathlib import Path
import numpy as np
from surface_features_v1 import PYTHON, environment, sha, write
from mtare_topo.governance import load_json, build_run_id
from mtare_topo.governance_axis_pair_diagnostic import scope, validate_card, SCHEMA, SLUG
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases
from mtare_topo.semantics.primitive_relation_nonlearning import RobustPrimitiveRelationBaseline
from mtare_topo.representation.gse_axis_pair_proposals import axis_pair_proposals
from mtare_topo.evaluation.gse_candidate_coverage import candidate_coverage
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry

ROOT = Path(__file__).resolve().parents[2]
CARD = 'configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate3/'+SLUG+'.json'
RUN = 'results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
ENTRY = 'tools/v3/axis_pair_diagnostic.py'
PATCHES = False
PATCH_CENTERS = False


def freeze():
    if any((ROOT/p).exists() for p in (CARD, SPEC)):
        raise FileExistsError('no refreeze')
    s = scope(ROOT, PATCHES, PATCH_CENTERS)
    a = dict(status='APPROVED', approved_by='user-standing-scope-authorization', approved_at='2026-09-08',
             scope='Exact45 synthetic observation-only axis-pair audit, zero training', scope_sha256=digest(s),
             authorized_operations=['audit'], authorized_gates=[3],
             confirmation_reference='User standing autonomous authority; PLAN same45 candidate diagnostic next step')
    card = dict(schema_version=SCHEMA, card_id=SLUG, operation='audit', scope=s, scope_sha256=digest(s), approval=a)
    assert validate_card(card).passed
    files = sorted(str(p.relative_to(ROOT)) for folder in ('src/mtare_topo', 'tools/v3') for p in (ROOT/folder).rglob('*.py'))
    spec = dict(schema_version='v3_run_spec_v1', gate=3, date='20260908', slug=SLUG, seed=0,
                operation='audit', data_card=CARD, config_path=CARD, user_authorization=a,
                command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',PYTHON,ENTRY,'--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)],
                question='Do observation-derived axis-pair hypotheses cover declared synthetic structures without teacher seeding?',
                method=s['method'], baseline='Unfiltered default nonlearning geometry; no comparison to learned detection scores',
                fallback='Report missing terminals, duplicate and extrapolated proposals; no retry, tuning or training',
                estimated_cost=dict(compute='CPU only,45 observations,zero learned inference',host_ram_gb=4,gpu_vram_gb=0,disk_gb=1,wall_time_hours=1),
                acceptance_criteria=['All45 retained; all hypotheses scored at1/2/4m; no teacher proposal selection',
                    'No research pass; eligible for consideration only if central9 junctions all have a1m candidate, with counts and false proposals exposed',
                    'No terminal exclusion, hidden filtering, model training or retry'],
                expected_evidence=['Per-case fitted axes,pair evidence,all scoring,raw log,environment,summary,source snapshot,SHA256 seal'],
                source_sha256={p:sha(ROOT/p) for p in files}, environment=environment())
    if PATCHES:
        spec['command'].insert(spec['command'].index('--spec'), '--patch-centers' if PATCH_CENTERS else '--patches')
        spec['acceptance_criteria'][1] = s['acceptance']
        spec['baseline'] = 'Axis-pair-only and combined axis/patch hypotheses scored separately, no learned detector comparison'
    for path, value in ((CARD,card),(SPEC,spec)):
        with (ROOT/path).open('x') as f: json.dump(value,f,indent=2)
    print(json.dumps(dict(spec=SPEC,scope_sha256=digest(s),observations=45)))


def execute(spec, run):
    run = run.resolve(strict=True)
    if run != ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED':
        raise ValueError('fresh exact run required')
    if load_json(run/'config/run_spec.json') != spec: raise ValueError('spec drift')
    started=time.monotonic(); error=None; rows=[]
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('resource cap')));signal.alarm(3600)
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card != load_json(run/'config/data_card.json'): raise ValueError('card drift')
        if environment()!=spec['environment']: raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n': raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        cases={c['case_id']:c for c in declared_cases()}
        baseline=RobustPrimitiveRelationBaseline()
        for r in card['scope']['observations']:
            with np.load(io.BytesIO(read_pinned(ROOT,r['input_path'],r['input_sha256'])),allow_pickle=False) as p:
                values=np.stack((p['ranges_m']/50.,p['valid_mask']),axis=1)
                translation=p['relative_translation_current_sensor_m'];yaw=p['relative_yaw_current_sensor_deg']
                prediction=baseline.predict(values,translation,yaw)
            proposals=axis_pair_proposals(prediction.axis_control_current_sensor_m,prediction.primitive_mask.astype(bool),coordinate_frame='current_sensor')
            xyz=np.asarray([v['position_m'] for v in proposals['pairs'] if v['position_m'] is not None],dtype=float).reshape(-1,3)
            pair_xyz=xyz.copy();patch_records=[]
            if 'patch_parameters' in card['scope']:
                from mtare_topo.semantics.primitive_relation_nonlearning import _register_points
                from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
                from mtare_topo.representation.gse_axis_patch_evidence import axis_patch_evidence
                points,frames=_register_points(values,translation,yaw)
                patches=extract_surface_patches(points,np.ones(len(points),dtype=bool),frames,
                    ray_origins_m=translation[frames],**card['scope']['patch_parameters'])
                e=axis_patch_evidence(prediction.axis_control_current_sensor_m,prediction.primitive_mask.astype(bool),patches,coordinate_frame='current_sensor')
                for i,j in np.argwhere(e.within_patch_bounds):
                    patch_records.append(dict(axis_index=int(i),patch_index=int(j),position_m=e.position_m[i,j].tolist(),
                        normal_axis_abs_dot=float(e.normal_axis_abs_dot[i,j]),extrapolation_m=float(e.extrapolation_m[i,j]),
                        point_count=int(patches.point_count[j]),frame_support=patches.frame_support[j].tolist(),
                        patch_bounds_m=[patches.bounds_min_m[j].tolist(),patches.bounds_max_m[j].tolist()],
                        closure_confirmed=False))
                xyz=np.concatenate((xyz,e.position_m[e.within_patch_bounds]),axis=0)
                if card['scope'].get('patch_candidate_selection') == 'all_observed_patch_centers':
                    patch_records=[dict(patch_index=j,position_m=c.tolist(),normal=patches.normals[j].tolist(),
                        normal_valid=bool(patches.normal_valid[j]),point_count=int(patches.point_count[j]),
                        frame_support=patches.frame_support[j].tolist(),closure_confirmed=False)
                        for j,c in enumerate(patches.centers_m)]
                    xyz=np.concatenate((pair_xyz,patches.centers_m),axis=0)
            # Only after predictions exist, construct independent scoring targets.
            truth=expected_geometry(cases[r['case_id']])
            score=candidate_coverage(np.asarray(truth['anchors'],dtype=float).reshape(-1,3),xyz,
                complete_region=truth['complete_for_declared_fixture'],expected_frame='current_sensor',candidate_frame='current_sensor')
            record=dict(case_id=r['case_id'],axes_m=prediction.axis_control_current_sensor_m[prediction.primitive_mask.astype(bool)].tolist(),proposals=proposals,score=score)
            if 'patch_parameters' in card['scope']:
                record.update(patch_candidates=patch_records,patch_count=len(patches.centers_m),
                    unique_axis_patch_intersections=int(e.unique_intersection.sum()),
                    axis_pair_only_score=candidate_coverage(np.asarray(truth['anchors'],dtype=float).reshape(-1,3),pair_xyz,
                        complete_region=truth['complete_for_declared_fixture'],expected_frame='current_sensor',candidate_frame='current_sensor'))
            write(run/('artifacts/'+r['case_id']+'.json'),record);rows.append(record)
            line=json.dumps(dict(case_id=r['case_id'],axes=proposals['valid_axis_count'],candidates=len(xyz),score=score['scores']['1.0']))
            with (run/'logs/raw.jsonl').open('a') as f:f.write(line+'\n')
            print(line,flush=True)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise MemoryError('host cap')
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>1024**3:raise RuntimeError('output cap')
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
    except Exception:error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        junction=[r for r in rows if r['case_id'].split('__')[0] in ('T','Y','four_way')]
        summary=dict(status='FAILED' if error else 'DIAGNOSTIC_COMPLETE',error=error,observations=len(rows),
            junctions_covered_1m=sum(r['score']['scores']['1.0']['matched']>0 for r in junction),
            total_candidates=sum(r['score']['candidate_count'] for r in rows),
            totals={t:{k:sum(r['score']['scores'][t][k] for r in rows) for k in ('matched','missed','false_positives')} for t in ('1.0','2.0','4.0')},
            optimizer_steps=0,learned_model_inference=0,scientific_gate_pass=False,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(run/'metrics/summary.json',summary);write(run/'config/environment.json',spec['environment'])
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--patches',action='store_true');p.add_argument('--patch-centers',action='store_true')
    a=p.parse_args()
    if a.patches or a.patch_centers:
        PATCHES=True;PATCH_CENTERS=a.patch_centers
        SLUG='gse_patch_center_diagnostic_v1' if PATCH_CENTERS else 'gse_axis_patch_diagnostic_v1'
        CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
        RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('freeze or spec/run-dir required')
