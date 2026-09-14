"""One bounded derived-observation correction; never rewrite the original run."""
import _bootstrap
import argparse,gzip,io,json
from pathlib import Path
import numpy as np
import synthetic_matrix_qualification as guard
from mtare_topo.governance_synthetic_corrective import SCHEMA,SLUG,POLICY,scope,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.data.gse_synthetic_corrective import derive_corrective,restore_declared_case
from mtare_topo.data.gse_hidden_counterfactual import remove_unobserved_hidden_operand
from mtare_topo.data.gse_synthetic_matrix_execution import evaluate
from mtare_topo.data.gse_lossless_evidence_v1 import write_evidence
from mtare_topo.governance_surface_material import read_pinned

ROOT=guard.ROOT;CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'


def freeze(*,slug=SLUG,schema=SCHEMA,validator=validate_card,entry='tools/v3/synthetic_four_corrective.py'):
    card_path='configs/v3/gate3/data_cards/'+slug+'.json';spec_path='configs/v3/gate3/'+slug+'.json'
    run_path='results/gate3_semantics/gate3_20260908_'+slug+'_seed20260906'
    if (ROOT/card_path).exists() or (ROOT/spec_path).exists():raise FileExistsError('no refreeze')
    s=scope();a=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        authorized_operations=['data_export'],authorized_gates=[3],scope_sha256=digest(s),
        confirmation_reference='User standing autonomous scope authorization; PLAN entry and GSE_INTERVAL_EXIT_CORRECTIVE_20260908.md bind these four synthetic observations.',
        scope='Only four archived failed synthetic observations; 20 missing returns may change; no training or real-world reads.')
    card=dict(schema_version=schema,card_id=slug,operation='data_export',scope=s,scope_sha256=digest(s),policy=POLICY,approval=a)
    assert validator(card).passed
    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906',
        guard.PYTHON,entry,'--spec',str(ROOT/spec_path),'--run-dir',str(ROOT/run_path)]
    files=sorted(str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py'))
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=slug,seed=20260906,operation='data_export',data_card=card_path,config_path=card_path,
        user_authorization=a,command=command,question='Do versioned missing-ray corrections preserve other inputs and support hidden-map invariance?',
        method='Interval winding rescue of 20 missing returns; source-bound derived scans and four same-observation deletion controls.',
        baseline='Sealed four failed original controls retained; no original score rewriting.',
        fallback='Stop any binding/repair/control error; no retry or scope expansion.',
        estimated_cost=dict(compute='CPU only; 20 ray repairs, four observations and four controls',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=3),
        wall_time_cap_s=10800,acceptance_criteria=['20 corrected positions; 230380 valid positions and all motion unchanged.',
            'Exact source binding; all four deletion controls exclude recorded segments and preserve target records.',
            'No real-data reads, optimizer steps, original overwrites or full-label qualification.'],
        expected_evidence=['Derived NPZs, source codebooks, repair provenance, raw/target/control evidence, logs, summary and seal.'],
        source_sha256={p:guard.sha(ROOT/p) for p in files},environment=guard.environment())
    for path,value in ((card_path,card),(spec_path,spec)):
        with (ROOT/path).open('x') as f:json.dump(value,f,indent=2)
    print(spec_path)


def execute_four(run,*,progress,resource_check):
    s=json.loads((run/'config/data_card.json').read_text())['scope'];rows=[]
    for key,count in s['case_invalid_counts'].items():
        prefix=s['source_run']+'/artifacts/'+key
        raw=read_pinned(ROOT,prefix+'.json.gz',s['input_sha256'][prefix+'.json.gz'])
        saved=json.loads(gzip.decompress(raw))
        case=restore_declared_case(saved['case'])
        with np.load(io.BytesIO(read_pinned(ROOT,prefix+'.npz',s['input_sha256'][prefix+'.npz']))) as z:
            bundle=dict(source=saved['source'],construction_teacher_only=saved['construction'],codebook_teacher_only=saved['codebook'],
                student={k:z[k].copy() for k in ('ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg')},
                sensor_teacher_only={k:z[k].copy() for k in ('sensor_xyz_m','yaw_deg','primitive_membership_code')})
        derived=derive_corrective(case,bundle)
        with (run/'artifacts'/(key+'.npz')).open('xb') as f:np.savez_compressed(f,**derived['student'],**derived['sensor_teacher_only'])
        write_evidence(run/'artifacts'/(key+'_derived.json.gz'),{k:v for k,v in derived.items() if k not in ('student','sensor_teacher_only')})
        raw,target,_=evaluate(derived,case)
        write_evidence(run/'artifacts'/(key+'_primary_targets.json.gz'),dict(raw=raw,target=target))
        control=remove_unobserved_hidden_operand(case,derived)
        craw,ctarget,_=evaluate(control,case)
        same=target['record']==ctarget['record']
        write_evidence(run/'artifacts'/(key+'_targets.json.gz'),dict(raw=raw,target=target,control_raw=craw,control_target=ctarget,
            control_provenance=control['counterfactual_provenance'],control_source=control['source'],control_codebook=control['codebook_teacher_only'],
            control_construction=control['construction_teacher_only'],control_membership_code=control['sensor_teacher_only']['primitive_membership_code'].tolist()))
        row=dict(case_id=key,changed=count,unchanged=derived['derivation_provenance']['unchanged_ray_positions'],control_pass=same)
        rows.append(row)
        with (run/'logs/conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        progress(row);resource_check()
        if not same:raise ValueError('derived control target mismatch')
    assert sum(x['changed'] for x in rows)==20 and sum(x['unchanged'] for x in rows)==230380
    return dict(primary_observations=4,control_conditions=4,control_pass=4,changed_ray_positions=20,unchanged_ray_positions=230380,
        per_case=rows,formal_optimizer_steps=0,full_matrix_qualification=False,scientific_gate_pass=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(guard.execute(json.loads(a.spec.read_text()),a.run_dir,validator=validate_card,execute_fn=execute_four))
    else:p.error('--freeze or --spec/--run-dir required')
