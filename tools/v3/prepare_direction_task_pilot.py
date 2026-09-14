"""Fixed independent-parent junction-neighborhood input pilot, no labels."""
from _bootstrap import PROJECT_ROOT as ROOT
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math,sys
from run_short_observation_chain import sha,write
CODE=Path(__file__).resolve().parents[2]
NAME='gse_direction_task_pilot_inputs_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260912_{NAME}_seed20260912'
COVER='results/gate3_semantics/gate3_20260910_gse_full_local_pair_coverage_v1_seed20260906'

def select(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='fit' and r['task'].endswith('__ellipse'):
            for p in r['reference_pairs']:groups[r['parent'],r['task'],r['traversal_index'],p['junction']].append(r)
    options=[]
    for k,rs in groups.items():
        rs=sorted(rs,key=lambda r:r['sequence_row'])
        for i in range(len(rs)-3):
            block=rs[i:i+4]
            if all(a['frame_rows'][1:]==b['frame_rows'][:-1] and b['sequence_row']==a['sequence_row']+1 for a,b in zip(block,block[1:])):
                rank=hashlib.sha256(json.dumps([20260912,k,block[0]['sequence_row']],separators=(',',':')).encode()).hexdigest()
                options.append((rank,k,block))
    chosen=[];parents=set()
    for rank,k,block in sorted(options):
        if k[0] in parents:continue
        chosen.append(dict(rank_sha256=rank,parent=k[0],task=k[1],traversal_index=k[2],reference_junction=k[3],observations=block));parents.add(k[0])
        if len(chosen)==3:break
    if len(chosen)!=3:raise ValueError('insufficient distinct parents; never duplicate to pad')
    return chosen

def freeze():
    from mtare_topo.governance_surface_input import SEALS,expected_tasks
    from mtare_topo.governance_identity_inventory import P1A
    p=COVER+'/artifacts/covered_sequences.jsonl';seal=ROOT/COVER/'artifacts/evidence_sha256.txt'
    known={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    if sha(ROOT/p)!=known[p]:raise ValueError('sealed source metadata drift')
    blocks=select([json.loads(l) for l in (ROOT/p).read_text().splitlines()])
    observations=[dict(r,fragment_id=i) for i,b in enumerate(blocks) for r in b['observations']]
    tasks=expected_tasks();selected={b['task'] for b in blocks};pins={p:known[p],str(seal.relative_to(ROOT)):sha(seal)};sourcepins={}
    prefixes=[tasks[t][role]+'/' for t in selected for role in ('sensor','teacher')]
    docs={P1A+'/artifacts/'+kind+'/'+tasks[t]['partition']+'/'+t+'.json' for t in selected for kind in ('constructions','codebooks')}
    for entry in SEALS.values():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('archive drift')
        pins[entry['path']]=entry['sha256']
        for line in (ROOT/entry['path']).read_text().splitlines():
            h,path=line.split(None,1)
            if path in docs or any(path.startswith(pre) for pre in prefixes):sourcepins[path]=h
    plans={};task_files={};decoded=0
    for t in sorted(selected):
        rows=[r for r in observations if r['task']==t];task_files[t]={}
        for kind in ('constructions','codebooks'):
            path=P1A+'/artifacts/'+kind+'/'+tasks[t]['partition']+'/'+t+'.json';pins[path]=sourcepins[path];task_files[t][kind]=path
        for role,fields in [('sensor',['range_m','valid_mask','sensor_xyz_m','yaw_deg','primitive_membership_code']),('teacher',['relative_translation_current_sensor_m','relative_yaw_current_sensor_deg','frame_row','source_global_sequence_index'])]:
            indices=sorted({f for r in rows for f in r['frame_rows']}) if role=='sensor' else sorted(r['sequence_row'] for r in rows)
            for field in fields:
                prefix=tasks[t][role]+'/'+field;path=prefix+'/.zarray'
                if sha(ROOT/path)!=sourcepins[path]:raise ValueError('header drift')
                h=json.loads((ROOT/path).read_text());assert h['chunks'][1:]==h['shape'][1:] and not h['filters'] and h['order']=='C'
                keys=['.'.join(map(str,[i]+[0]*(len(h['shape'])-1))) for i in sorted({r//h['chunks'][0] for r in indices})]
                for path in [prefix+'/.zarray',*[prefix+'/'+k for k in keys]]:pins[path]=sourcepins[path]
                plans[prefix]=dict(header=h,selected_rows=indices,chunk_keys=keys);decoded+=len(keys)*math.prod(h['chunks'])*int(h['dtype'][2:])
    scope=dict(blocks=blocks,observations=observations,counts=dict(fit=dict(parents=3,observations=12,tasks=3,unique_variant_frames=24)),
        array_plans=plans,task_files=task_files,task_prefixes={t:tasks[t] for t in selected},source_sha256=pins,decoded_padded_bytes=decoded,
        sampling='Seed20260912 stable hash of parent/task/traversal/junction/start; first3 distinct fit parents;4 contiguous windows each',
        independent_units=3,spatial_spacing='Original consecutive traversal windows; source pose distances verified at export; no temporal duration claim',
        teacher_source='Original separate construction/codebook/source rays only; NOT_EVALUATED visibility remains, no correspondence labels generated',
        leakage_audit='C01/C06 fit only; same parent/variant stays fit; no C07-C10, model score or evaluation-driven selection',
        labels_generated=0,training_steps=0,limits=dict(wall_seconds=900,host_bytes=4*1024**3,output_bytes=1024**3))
    approval=dict(status='APPROVED',approved_by='user-start-task-correspondence-validation',approved_at='2026-09-12',authorized_gates=[3],authorized_operations=['data_export'],
        confirmation_reference='User 所以开始做啊 following explicit local task correspondence implementation; standing scoped data export authorization',
        scope='Three fixed fit-parent fragments24frames/12windows; separate student/reference export only, no labels or training',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version=NAME,scope=scope,approval=approval))
    # Reused exporter verifies original-root sources, which remain unchanged.
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260912',slug=NAME,seed=20260912,operation='data_export',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__)),'--execute'],
        question='Can three actual contiguous junction-neighborhood fragments supply separated observation/reference evidence for task correspondence?',
        method='Reuse archived exact decoder and student/source alignment, fixed hash selection; no teacher label promotion',
        baseline='Original sealed scan/motion/source identity and reference visibility unknown',fallback='No substitute fragments or generated labels; preserve failures',
        estimated_cost=dict(compute='CPU3tasks; no GPU or model',host_ram_gb=4,disk_gb=1,wall_time_hours=.25),
        acceptance_criteria=['24unique frames/12windows/3fitparents','Four contiguous windows per fragment','Source and student fields separated','No true task identity in student inputs; no qualified correspondence claim'],
        expected_evidence=['Student windows, separate construction and ray sources, exact manifest, environment, logs, source snapshot and seal'],
        input_sha256=dict(pins),source_sha256=sources,
        execution_code_root=str(CODE),execution_source_sha256={str(p.relative_to(CODE)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (CODE/folder).rglob('*.py')}))
    print(json.dumps(dict(spec=SPEC,parents=[b['parent'] for b in blocks],frames=24,observations=12,decoded_bytes=decoded)))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        import zipfile
        spec=json.loads((ROOT/SPEC).read_text())
        for path,h in spec['execution_source_sha256'].items():
            if sha(CODE/path)!=h:raise ValueError('execution source drift')
        with zipfile.ZipFile(ROOT/RUN/'artifacts/execution_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in spec['execution_source_sha256']:z.write(CODE/path,path)
        from export_local_pair_pilot import execute
        from mtare_topo.governance_direction_task_pilot import validate_card
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card))
