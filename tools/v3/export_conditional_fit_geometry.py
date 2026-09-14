"""Same conditional reference algorithm on all frozen fit observations."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json
from ai_junction_pilot import sha,write
from export_conditional_development_inputs import PYTHON
from export_conditional_fit_evidence import RUN as INPUT
from export_conditional_fit_features import RUN as FEATURE
from mtare_topo.governance_conditional_fit_geometry import BINDING,LIMITS,validate_card
NAME='gse_conditional_fit_geometry_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
ORIGINAL='configs/v3/gate3/gse_original_surface_source_binding_v1.json'

def freeze():
    rows=json.loads((ROOT/BINDING).read_text())['entries']
    evidence=json.loads((ROOT/INPUT/'artifacts/manifest.json').read_text())
    lookup={(r['source']['task'],r['source']['source_sequence_id']):r for r in evidence}
    original=json.loads((ROOT/ORIGINAL).read_text())
    assert original==json.loads((ROOT/'configs/v3/gate3/gse_new12_surface_source_scope_v1.json').read_text())['original_surface_binding']
    pins={BINDING:sha(ROOT/BINDING),ORIGINAL:sha(ROOT/ORIGINAL),original['archive_path']:original['archive_sha256']}
    sealed={}
    for base in (INPUT,FEATURE):
        assert json.loads((ROOT/base/'RUN_STATE.json').read_text())['state']=='COMPLETED'
        seal=ROOT/base/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        sealed.update({p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())})
    entries=[]
    for i,row in enumerate(rows):
        r=lookup[(row['task'],row['source_sequence_id'])]
        assert r['source']['frame_rows']==row['frame_rows'] and r['student_path_basis']=='project_root_task_batch'
        paths=dict(student=row['student_path'],evidence=INPUT+'/'+r['source_evidence_path'],
                   construction=INPUT+'/artifacts/source_evidence/'+row['task']+'_constructions.json',
                   codebook=INPUT+'/artifacts/source_evidence/'+row['task']+'_codebooks.json',
                   feature=FEATURE+f'/artifacts/window_{i:02d}.npz')
        for role,path in paths.items():
            expected=row['student_sha256'] if role=='student' else sealed[path]
            if path not in pins:
                assert sha(ROOT/path)==expected; pins[path]=expected
        entries.append(dict(identity=dict(row,case=i),student_binding=row,paths=paths))
    s=dict(entries=entries,parents=60,observations=2880,frames=14400,patches=2366216,roi_returns=99928912,
           original_surface_binding=original,target_schema='construction_conditioned_geometry_targets_v1',observability_certified=False,connectivity_certified=False,training_steps=0,
           selection='Original seed20260906 fixed16edges/parent,one direction/window,three variants; no model selection',
           spacing='Five ordered frame IDs, decision arc in entries; acquisition time/history spacing unknown; independent traversals, not continuous exploration',
           split_audit='C01-C06 fit only; old encoder fit here; no C07-C10 or benchmark payload; no unseen claim',limits=LIMITS)
    a=dict(status='APPROVED',approved_by='user-standing-conditional-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['data_export'],
           scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='Same conditional references for frozen2880fit observations; no optimizer, graph or test',
           confirmation_reference='User confirmed construction-conditioned supervision and continuing development goal; bounded fit reference scope announced before execution')
    card=dict(schema_version='gse_conditional_fit_geometry_card_v1',scope=s,approval=a);assert validate_card(card).passed;write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_conditional_fit_geometry.py','--execute'],
        question='What same-definition conditional geometry targets exist across the complete frozen fit population?',
        method='Unchanged original surface matcher, singleton arc components, middle slab references and unknown compiler;4CPU workers',
        baseline='Original12fit/240development reference definition; no model scores consumed',fallback='Seal failure; no sample deletion, retry, changed labels or threshold',
        acceptance_criteria=['2880 identities/2366216patches/99928912ROIreturns','All ambiguous sources retained','Original conditional reference semantics unchanged','Zero GPU/optimizer/contour qualification'],
        expected_evidence=['Per-case source records,references,targets,masks,logs,summary,environment,source snapshot and SHA seal'],
        estimated_cost=dict(compute='4CPUworkers;linear development extrapolation14.83h,hard cap24h',host_ram_gb=32,gpu_vram_gb=0,disk_gb=8,wall_time_hours=24),input_sha256=pins,source_sha256=sources))
    print(SPEC)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');g.add_argument('--worker',type=int);a=p.parse_args()
    if a.freeze:freeze()
    elif a.worker is not None:
        from export_development_geometry import worker
        worker(a.worker,spec_path=SPEC,card_path=CARD,run_path=RUN)
    else:
        from export_development_geometry import execute
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card,worker_script=str(__file__)))
