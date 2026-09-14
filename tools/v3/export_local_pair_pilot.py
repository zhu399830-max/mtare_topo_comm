"""Fixed representative raw inputs; source evidence is NOT a generated label."""
from _bootstrap import PROJECT_ROOT as ROOT
from check_local_pair_window_coverage import sha, write
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

SLUG='gse_local_pair_pilot_inputs_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
PARENT='results/gate3_semantics/gate3_20260910_gse_full_local_pair_coverage_v1_seed20260906'
SENSOR=('range_m','valid_mask','sensor_xyz_m','yaw_deg','primitive_membership_code')
SEQUENCE=('relative_translation_current_sensor_m','relative_yaw_current_sensor_deg','frame_row','source_global_sequence_index')


def nominate(rows):
    groups=defaultdict(list)
    for row in rows:
        for p in row['reference_pairs']:
            key=(row['parent'],row['task'],p['junction'],p['terminal'])
            groups[key].append(row)
    selected={}
    for key,choices in sorted(groups.items()):
        def rank(row):
            return hashlib.sha256(json.dumps([20260906,*key,row['source_global_sequence_index']],separators=(',',':')).encode()).hexdigest()
        row=min(choices,key=rank);identity=(row['task'],row['source_global_sequence_index'])
        item=selected.setdefault(identity,dict(row,reference_pairs=[]))
        item['reference_pairs'].append(dict(junction=key[2],terminal=key[3]))
    return [selected[k] for k in sorted(selected)],len(groups)


def freeze():
    from mtare_topo.governance_surface_input import SEALS,expected_tasks
    from mtare_topo.governance_identity_inventory import P1A
    if (ROOT/RUN).exists():raise FileExistsError('existing run')
    pins={p:h for h,p in (l.split('  ',1) for l in (ROOT/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    covered=PARENT+'/artifacts/covered_sequences.jsonl'
    if sha(ROOT/covered)!=pins[covered]:raise ValueError('coverage source drift')
    observations,strata=nominate([json.loads(l) for l in (ROOT/covered).read_text().splitlines()])
    if strata!=159:raise ValueError('fixed159 pair/variant strata required')
    tasks=expected_tasks();chosen_tasks={o['task'] for o in observations};files={covered:pins[covered]}
    prefixes={tasks[t]['sensor']+'/'+f+'/' for t in chosen_tasks for f in SENSOR}
    prefixes.update(tasks[t]['teacher']+'/'+f+'/' for t in chosen_tasks for f in SEQUENCE)
    json_paths={P1A+'/artifacts/'+kind+'/'+tasks[t]['partition']+'/'+t+'.json' for t in chosen_tasks for kind in ('constructions','codebooks')}
    source_pins={}
    for seal in SEALS.values():
        raw=(ROOT/seal['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal['sha256']:raise ValueError('archive seal drift')
        files[seal['path']]=seal['sha256']
        for line in raw.decode().splitlines():
            h,p=line.split(None,1)
            if p in json_paths or any(p.startswith(prefix) for prefix in prefixes):source_pins[p]=h
    plans={};task_files={};decoded=0
    for task in sorted(chosen_tasks):
        rows=[o for o in observations if o['task']==task]
        task_files[task]={}
        for kind in ('constructions','codebooks'):
            path=P1A+'/artifacts/'+kind+'/'+tasks[task]['partition']+'/'+task+'.json'
            files[path]=source_pins[path];task_files[task][kind]=path
        for role,fields in (('sensor',SENSOR),('teacher',SEQUENCE)):
            indices=sorted({f for o in rows for f in o['frame_rows']}) if role=='sensor' else sorted({o['sequence_row'] for o in rows})
            for field in fields:
                prefix=tasks[task][role]+'/'+field;path=prefix+'/.zarray';raw=(ROOT/path).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=source_pins[path]:raise ValueError('array header drift')
                h=json.loads(raw)
                if h['chunks'][1:]!=h['shape'][1:] or h['filters'] or h['order']!='C':raise ValueError('unsupported source layout')
                keys=['.'.join(map(str,[i]+[0]*(len(h['shape'])-1))) for i in sorted({j//h['chunks'][0] for j in indices})]
                files[path]=source_pins[path]
                for key in keys:files[prefix+'/'+key]=source_pins[prefix+'/'+key]
                plans[prefix]=dict(header=h,selected_rows=indices,chunk_keys=keys)
                decoded+=len(keys)*math.prod(h['chunks'])*int(h['dtype'][2:])
    counts={}
    for split in ['fit','calibration','development']:
        rows=[o for o in observations if o['split']==split]
        counts[split]=dict(observations=len(rows),parents=len({o['parent'] for o in rows}),
            tasks=len({o['task'] for o in rows}),unique_variant_frames=len({(o['task'],f) for o in rows for f in o['frame_rows']}))
    scope=dict(observations=observations,counts=counts,strata=strata,source_sha256=files,array_plans=plans,task_files=task_files,
        task_prefixes={t:tasks[t] for t in chosen_tasks},decoded_padded_bytes=decoded,
        selection='One minimum SHA256(seed20260906,parent,task,junction,terminal,global_sequence) per covered pair/variant; duplicate windows merged',
        student_fields=['ranges_m','valid_mask',*SEQUENCE[:2]],teacher_fields=list(SENSOR[2:]),
        labels_generated=0,training=False,strict_test=False,
        limitations=['geometry-biased diagnostic nomination, not random test population','development contains one parent only',
            'construction/source codes are teacher-only evidence; no membership labels asserted','no new sensor rendering or hidden/test worlds'])
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_gates=[3],authorized_operations=['data_export'],scope='Fixed representative existing-source input and separate evidence export; no labels/training',
        confirmation_reference='User goal authorizes scoped validation; fixed per-pair nomination announced before selection',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version='gse_local_pair_pilot_card_v1',scope=scope,approval=approval),'w')
    sources=[Path(__file__),ROOT/'tools/v3/check_local_pair_window_coverage.py',ROOT/'tools/v3/_bootstrap.py',ROOT/CARD,
        ROOT/'src/mtare_topo/data/gse_surface_teacher_reader_v1.py',ROOT/'src/mtare_topo/data/primitive_relation_sequences.py',
        ROOT/'src/mtare_topo/governance.py',ROOT/'src/mtare_topo/governance_branch_place_replay.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=approval,
        command=[sys.executable,'tools/v3/export_local_pair_pilot.py','--execute'],
        question='Can fixed representative windows be materialized with exact sensor/motion/source alignment and teacher isolation?',
        method='Reuse actual archived returns, motions and source codes; no inference or teacher generation',
        baseline='Frozen source arrays; identity/precision equality only',estimated_cost=dict(compute='CPU, sequential task decode under4GiB',disk_gb=1,wall_time_hours=.25),
        acceptance_criteria=['Exact nominated population, no replacement','Original coordinate-motion and codebook validity parity','Student NPZ excludes identities and absolute poses','No labels or training qualification claimed'],
        expected_evidence=['student NPZ, separate source evidence, source-bound manifest, environment, logs, seal'],
        input_sha256=files,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}),'w')
    print(json.dumps(dict(counts=counts,strata=strata,decoded_bytes=decoded,spec=SPEC)))


def execute(*, run_path=RUN, spec_path=SPEC, card_path=CARD, validator=None, reuse_student_cache=False):
    import numpy as np
    import numcodecs
    from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
    from mtare_topo.governance_branch_place_replay import validate_local_pair_pilot_card
    out=ROOT/run_path;spec=json.loads((ROOT/spec_path).read_text());card=json.loads((ROOT/card_path).read_text());s=card['scope']
    validator = validator or validate_local_pair_pilot_card
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec and validator(card).passed
    for p,h in dict(spec['input_sha256'],**spec['source_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;manifest=[]
    import resource, signal, zipfile
    limits=s.get('limits')
    if limits:
        def expire(*_):raise TimeoutError('fixed source export wall cap')
        signal.signal(signal.SIGALRM,expire);signal.alarm(limits['wall_seconds'])
    def read_rows(prefix):
        plan=s['array_plans'][prefix];h=plan['header'];codec=numcodecs.get_codec(h['compressor']);selected=set(plan['selected_rows']);result={}
        for key in plan['chunk_keys']:
            a=np.frombuffer(codec.decode((ROOT/(prefix+'/'+key)).read_bytes()),dtype=h['dtype']).reshape(h['chunks'])
            first=int(key.split('.')[0])*h['chunks'][0]
            for row in selected:
                if first<=row<first+len(a):result[row]=a[row-first].copy()
        if set(result)!=selected:raise ValueError('missing selected row')
        return result
    try:
        if limits:
            with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
                for path in spec['source_sha256']:z.write(ROOT/path,path)
        for name in (('source_evidence',) if reuse_student_cache else ('student','source_evidence')):(out/'artifacts'/name).mkdir()
        for task in sorted(s['task_files']):
            documents={kind:json.loads((ROOT/p).read_text()) for kind,p in s['task_files'][task].items()}
            for kind,d in documents.items():write(out/'artifacts/source_evidence'/(task+'_'+kind+'.json'),d)
            paths=s['task_prefixes'][task]
            fields=SENSOR[2:] if reuse_student_cache else (*SENSOR,*SEQUENCE)
            arrays={f:read_rows(paths['sensor' if f in SENSOR else 'teacher']+'/'+f) for f in fields}
            task_rows=[o for o in s['observations'] if o['task']==task]
            if reuse_student_cache:
                from mtare_topo.data.gse_membership_fit_reader import load_student_task_batch
                cached=load_student_task_batch(ROOT,task_rows)
            for j,source in enumerate(task_rows):
                row=source['sequence_row'];frames=source['frame_rows']
                if not reuse_student_cache and (arrays['frame_row'][row].tolist()!=frames or int(arrays['source_global_sequence_index'][row])!=source['source_global_sequence_index']):
                    raise ValueError('nomination/sequence binding mismatch')
                student=(vars(cached[j]) if reuse_student_cache else dict(ranges_m=np.stack([arrays['range_m'][i] for i in frames]),valid_mask=np.stack([arrays['valid_mask'][i] for i in frames]),
                    **{f:arrays[f][row] for f in SEQUENCE[:2]}))
                evidence={f:np.stack([arrays[f][i] for i in frames]) for f in SENSOR[2:]}
                verify_alignment(evidence,student,documents['constructions'],documents['codebooks'],source)
                if (student['ranges_m'].shape!=(5,16,720) or not np.isfinite(student['ranges_m']).all()
                        or np.any(student['ranges_m']<0) or np.any(student['ranges_m']>50)):
                    raise ValueError('invalid original ranges')
                name=task+'_'+str(source['source_global_sequence_index'])+'.npz'
                sp=out/'artifacts/student'/name;tp=out/'artifacts/source_evidence'/name
                if not reuse_student_cache:np.savez_compressed(sp,**student)
                np.savez_compressed(tp,**evidence)
                manifest.append(dict(source=source,student_path=(source['student_path'] if reuse_student_cache else str(sp.relative_to(out))),student_sha256=(source['student_sha256'] if reuse_student_cache else sha(sp)),
                    student_path_basis=('project_root_task_batch' if reuse_student_cache else 'run_single_window'),
                    source_evidence_path=str(tp.relative_to(out)),source_evidence_sha256=sha(tp),labels_generated=False))
                if limits:
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limits['host_bytes']:raise MemoryError('host cap')
                    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>limits['output_bytes']:raise MemoryError('evidence cap')
            print(json.dumps(dict(task=task,exported=len(manifest),elapsed_s=time.monotonic()-start)),flush=True)
        if len(manifest)!=len(s['observations']):raise ValueError('incomplete fixed export')
        write(out/'artifacts/manifest.json',manifest)
        write(out/'artifacts/environment.json',dict(python=sys.version,numpy=np.__version__,numcodecs=numcodecs.__version__))
    except Exception:error=traceback.format_exc()
    finally:
        if limits:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,exported_observations=len(manifest),
        declared_counts=s['counts'],labels_generated=0,training_steps=0,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        meaning='Input/evidence materialization only, not label or method qualification')
    write(out/'metrics/summary.json',summary);write(out/'logs/raw.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
