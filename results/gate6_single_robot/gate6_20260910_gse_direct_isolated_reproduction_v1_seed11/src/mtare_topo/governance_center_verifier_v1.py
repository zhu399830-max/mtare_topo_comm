"""One bounded position-validity task, with explicit changed supervision."""
import json
from pathlib import Path
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_center_verifier_card_v1';SLUG='gse_candidate_center_verifier_v1'
SCORE='results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
L2='results/gate3_semantics/gate3_20260910_gse_fixed_score_l2_v1_seed0'
POSITION='results/gate3_semantics/gate3_20260909_gse_primitive_position_dynamic_v1r_seed0'
SEALS={SCORE:'300b8ca66a275c40b40850f557b76ca709741f2ea324f0fe107d15f98a30af70',L2:'ecd39f940a8f0eff19c5bc7563b27dfb7b4154a75fa37826a92ca945556b9b27',POSITION:'32be39a02bc791d5386b4d081d873e2aa66d0a5a5c9f286a5d41c398beb6a63e'}
AUTH='20260910 user taskbook d5f8d4e5-5534-4e4e-91a2-8f0593231dd0: implement one candidate-center geometry verifier, frozen predicted centers and predecoder block memory, versioned legal position labels, fixed2mXYZ greedy selection, zero-training oldL2 same selection, one1000full16updates, final evaluation, no graph/new model search.'
POLICY=dict(memory='ObservedAnchorDetectorV1.adapter output variable memory[M,128], after message layers and context/XYZ/extent/degenerate fusion, before extra positional encoding and Transformer query decoder; robot coordinates already present',
    architecture='nonaffineLN128 eps1e-5;relativeXYZ/10+extent/10+degenerate;135->64GELU->64GELU;valid mean/max;128->64GELU->1;21185parameters',
    validity='all<=1m confirmed source-bound candidates positive without unknown conflict; original independent4m background negatives only; duplicate-only not negative; conflicts halt; zero positive or negative halt training',
    selection='sigmoid>=.5 then descending probability,XYZ,slot;greedy<=2m suppression,no transitive merge/GT filter/top1',
    training=dict(seed=0,updates=1000,observations_per_update=16,optimizer='AdamW',lr_start=.001,lr_end=.00001,lr='cosine across1000updates,first1e-3 last1e-5',weight_decay=.0001,decay='new weights only,no bias decay',clip_norm=1.),
    evaluation='old1m and frozen original4m scoreable/unknown masks at1m; all512 before/after same NMS;final1000 only; both P/R>=.90 plus frozen/fullFP32 checks;unknown visible',
    numeric='permutation atol/rtol1e-5;cache/full CUDA allclose atol/rtol1e-5;CPU/CUDA actual selection equal;known minimum absolute logit>max(1e-4,10*maxCPU/CUDAerror) for clear threshold margin',
    limits=dict(wall_s=43200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=3*1024**3),
    no_restarts=True,no_graph=True,old_updates=0)

def compile_scope(root):
    indexes={}
    for run,h in SEALS.items():indexes[run]={p:v for v,p in (line.split('  ',1) for line in read_pinned(root,run+'/artifacts/evidence_sha256.txt',h).decode().splitlines())}
    idx=indexes[SCORE]
    old=json.loads(read_pinned(root,SCORE+'/config/data_card.json',idx[SCORE+'/config/data_card.json']))
    allowed=json.loads(read_pinned(root,SCORE+'/artifacts/source_reads_sha256.json',idx[SCORE+'/artifacts/source_reads_sha256.json']))
    bound={p:h for p,h in idx.items() if any(p.startswith(SCORE+'/'+x) for x in ('artifacts/fixed_candidates_','artifacts/fixed_supervision_evidence_','artifacts/input_','metrics/evaluation_0000.json','metrics/candidate_scores_0000.json'))}
    bound.update({p:h for p,h in indexes[L2].items() if any(p.startswith(L2+'/'+x) for x in ('artifacts/head_','metrics/old_evaluation_','metrics/fixed_region_','metrics/summary.json'))})
    pi=indexes[POSITION];schedule_path=POSITION+'/config/schedule.json';schedule=json.loads(read_pinned(root,schedule_path,pi[schedule_path]))
    checkpoint=POSITION+'/checkpoints/step_'+format(len(schedule),'04d')+'.pt'
    if checkpoint not in pi:raise ValueError('declared final schedule checkpoint missing in sealed manifest')
    bound[checkpoint]=pi[checkpoint];bound[schedule_path]=pi[schedule_path]
    return dict(full_forward_scope=old['scope'],full_forward_allowed_reads=allowed,bound_sha256=bound,seals=SEALS,parent_checkpoint=checkpoint,
        parent_checkpoint_sha256=pi[checkpoint],counts=old['scope']['counts'],candidate_queries=512,observations=16,old_label_counts=dict(positive=12,negative=162,unknown=338),
        new_labels='counts determined once on all512 by fixed validity evidence before any training; stop if illegal or missing class',
        spacing=old['scope']['spacing'],split='same16 fit-only,5physical structures,not16independent examples;no calibration or C08-C10',
        teacher='only existing sealed position references and source-bound4m background evidence;old duplicate selection labels not copied',
        token_cache='new observation-side predecoder cache on same causal five frames; full frozen parent inference, no new map teacher')

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong card operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]) or digest(card['scope'])!=card['scope_sha256']:errors.append('scope drift')
        if card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if a['status']!='APPROVED' or a['scope_sha256']!=card['scope_sha256'] or a['confirmation_reference']!=AUTH:errors.append('authorization missing')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
