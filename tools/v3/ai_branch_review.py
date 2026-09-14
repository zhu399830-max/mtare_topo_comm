"""Bounded observation-only review. No GT, prediction or training reads."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
import argparse,hashlib,json,sys
from pathlib import Path

NAME='gse_ai_branch_review_v1r1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
CACHE='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'
CODE=['tools/v3/ai_branch_review.py','src/mtare_topo/governance_branch_review.py']

def freeze():
    old=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_direction_task_model_cache_v1.json').read_text())
    s=dict(entries=old['scope']['entries'],observations=12,unique_frames=24,parents=3,
        split='fit',spacing='Original four successive windows per fragment, overlapping five-frame histories; exact frame_rows retained',
        independent_units='3 parent maps / 3 recorded fragments; not 12 independent places',
        selection='Entire existing pilot in fixed order, not selected by predictions',
        teacher_source='Current assistant observation interpretation; nonblind prior exposure disclosed; no construction loaded for review',
        blind=False,prior_teacher_exposure=True,human_gold_count=0,training_authorized=False,
        leakage='C01/C06 fit parents only; no C07-C10 or robot test reads; no score-selected cases',
        output='Per-observation judgments, optional explicit ray regions, reasons, unknowns; no automatic qualification')
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['ai_annotation'],authorized_gates=[3],
        scope='AI-assisted visible branch evidence review of the existing exact 12 observations, no training',
        confirmation_reference='你自己不能解决问题吗; 继续啊要做完整的工作别这么搞了; standing authorization for AI annotation',
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    annotation=dict(labeler_name='current assistant',labeler_version='current session; model revision not exposed',
        prompt_reference='Review all12 causal observations; identify visible distinct openings, do not infer hidden connectivity or crop ends',
        input_bundle_contract='SHA-bound student NPZ and registered observation arrays, current range plus five-frame views',
        output_schema='Observation ID, judgment, measured support regions, explicit reasons and unknowns',
        conflict_policy='Keep disagreements unknown; never overwrite old witness',abstain_policy='Missing boundaries or ray assignment means abstain',
        planned_ai_sample_count=12,planned_human_gold_count=0,strict_test_excluded=True,raw_responses_preserved=True)
    card=dict(schema_version='gse_ai_branch_review_v1',scope=s,approval=approval,annotation=annotation)
    from mtare_topo.governance import validate_data_card,validate_annotation_plan
    assert validate_data_card(card).passed and validate_annotation_plan(card).passed
    pins={}
    for i,e in enumerate(s['entries']):
        for p in (e['student_path'],f'{CACHE}/artifacts/common_{i:02d}.npz'):pins[p]=sha(ROOT/p)
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='ai_annotation',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',sys.executable,'tools/v3/ai_branch_review.py','--begin'],
        question='Can visible evidence support same/different branch assignments, not just different construction IDs?',
        method='Full12 observation-only in-session AI review of ranges and aligned five-frame surfaces',baseline='Existing weak witness preserved; no method ranking',
        fallback='Explicit abstention with image evidence and missing requirement, no automatic training',
        acceptance_criteria=['Every observation reviewed','No crop boundary or ID-derived negatives','Ray-addressable support required for labels','Unknown retained'],
        expected_evidence=['12 panels, raw review response, structured decisions, environment, summary, SHA seal'],
        estimated_cost=dict(compute='CPU rendering, in-session AI review,0training',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.25),
        input_sha256=pins,source_sha256={p:sha(ROOT/p) for p in CODE})
    write(ROOT/SPEC,spec);print(SPEC)

def execute(mode):
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text())
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():assert sha(ROOT/p)==h,p
    state=json.loads((out/'RUN_STATE.json').read_text())['state']
    if mode=='begin':
        assert state=='CREATED_NOT_EXECUTED'
        import numpy as np
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        write(out/'RUN_STATE.json',dict(state='RUNNING',activity='in-session review, no background training'),'w')
        write(out/'config/review_environment.json',dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__))
        for i,e in enumerate(card['scope']['entries']):
            with np.load(ROOT/e['student_path'],allow_pickle=False) as d:
                ranges=np.where(d['valid_mask'][-1],d['ranges_m'][-1],np.nan)
            with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:
                xyz=d['registered_returns_xyz_m'];valid=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz,axis=1)<=10)&(np.linalg.norm(xyz,axis=1)>0)
                xyz=xyz[valid]
            fig=plt.figure(figsize=(16,10))
            ax=fig.add_subplot(2,2,1);im=ax.imshow(ranges,origin='lower',aspect='auto',vmin=0,vmax=30,extent=(0,720,0,16),cmap='viridis');fig.colorbar(im,ax=ax,label='first return m (display capped30)');ax.set(xlabel='azimuth column (0.5deg)',ylabel='elevation row',title='CURRENT observation, invalid blank')
            for j,(a,b,title) in enumerate([(0,1,'XY'),(0,2,'XZ')],2):
                ax=fig.add_subplot(2,2,j);ax.scatter(xyz[::3,a],xyz[::3,b],s=.5,c=xyz[::3,2],cmap='viridis');ax.scatter(0,0,c='red',marker='+');ax.set(xlim=(-10,10),ylim=(-10,10),xlabel='X m',ylabel=('Y m' if b==1 else 'Z m'),title=f'Five causal frames {title}; sphere boundary NOT wall/end');ax.set_aspect('equal');ax.grid(alpha=.2)
            ax=fig.add_subplot(2,2,4,projection='3d');q=xyz[::5];ax.scatter(*q.T,s=.5,c=q[:,2],cmap='viridis');ax.view_init(25,45);ax.set(xlim=(-10,10),ylim=(-10,10),zlim=(-6,6),xlabel='X m',ylabel='Y m',zlabel='Z m');ax.set_box_aspect((20,20,12))
            fig.suptitle(f'Observation {i:02d} | fit pilot | current sensor coordinates | NO model/GT overlay');fig.tight_layout();fig.savefig(out/f'previews/observation_{i:02d}.png',dpi=110);plt.close(fig)
        print('12 observation panels ready; review then seal.');return
    assert state=='RUNNING'
    raw=out/'logs/raw_ai_response.md';assert raw.is_file()
    decisions=json.loads((out/'artifacts/review.json').read_text());assert len(decisions['observations'])==12
    assert [r['observation'] for r in decisions['observations']]==list(range(12))
    for r in decisions['observations']:
        assert r['reason'] and r['image']==f'previews/observation_{r["observation"]:02d}.png'
        assert r['training_qualified'] is False
    write(out/'metrics/summary.json',dict(status='GATE_MIXED',reviewed=12,human_gold_count=0,training_steps=0,
        train_ready=False,decisions=decisions,meaning='AI review is evidence, not automatic independent gold'))
    write(out/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print('Review sealed; no training run created.')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('freeze','begin','seal'):g.add_argument('--'+mode,action='store_true')
    a=p.parse_args();mode=next(m for m in ('freeze','begin','seal') if getattr(a,m))
    freeze() if mode=='freeze' else execute(mode)
