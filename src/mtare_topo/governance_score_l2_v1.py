import json
from pathlib import Path
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_score_l2_card_v1';SLUG='gse_fixed_score_l2_v1'
PARENT='results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
NUMERIC='results/gate3_semantics/gate3_20260910_gse_fixed_score_numerics_v1r_seed0'
SEALS={PARENT:'300b8ca66a275c40b40850f557b76ca709741f2ea324f0fe107d15f98a30af70',NUMERIC:'e027aa9f0500ea5b0b008794d9a883db63edde885cdd6c1eacad524e5afe47f0'}
AUTH='20260910 user: close section15; reuse fixed features/verified weight reduction; at most three prespecified L2 fullbatch same-head solves; actual float32 and complete frozen forward; preserve old scoring and add fixed scoreable region varying match radius only; unknown not background/no GT filtering; pass freeze dev baseline then independent structure holdout, fail stop rescuing fixed feature scoring, no model search.'
POLICY=dict(lambdas=[.01,.0001,.000001],loss='section15 verified weighted softplus plus lambda/2 * original128weight norm squared; bias unpenalized',
    initialization='same section14 final head for each, no warm-start between strengths',solver='one L-BFGS-B per strength;10000maxiter30000maxfun,ftol1e-15,gtol1e-10,maxls50,maxcor30; full invertible curvature preconditioning',
    fixed_region='section14 step0 original4m query_scoreable and unknown-competitor masks fixed for all .5/1/2/4m matching; diagnostic only, never inference filtering',
    acceptance='all174labels correct; old1m and fixed-region1m P/R>=.90; original/conditioned gradient_inf<=1e-8; all512 FP64/FP32CPU/FP32CUDA/full-forward selection identical; minimum absolute known logit>max(1e-4,10*max logit numerical error); full frozen positions/features exactly equal parent',
    selection='highest lambda that passes all criteria; only fit-based dev baseline, not independent success',
    failure='stop this fixed-feature implementation, no fourth strength/model search',limits=dict(wall_s=1800,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3))

def compile_scope(root):
    bound={}
    for run,h in SEALS.items():
        idx={p:v for v,p in (l.split('  ',1) for l in read_pinned(root,run+'/artifacts/evidence_sha256.txt',h).decode().splitlines())}
        prefixes=('artifacts/fixed_candidates_','artifacts/input_','artifacts/prediction_0000_','metrics/evaluation_0000.json',
            'metrics/candidate_scores_0000.json','config/data_card.json','checkpoints/step_1000.pt','artifacts/source_reads_sha256.json') if run==PARENT else ('artifacts/fixed_design.npz','artifacts/solved_affine.npz','metrics/loss_equivalence.json','metrics/summary.json')
        bound.update({p:v for p,v in idx.items() if any(p.startswith(run+'/'+x) for x in prefixes)})
    old=json.loads(read_pinned(root,PARENT+'/config/data_card.json',bound[PARENT+'/config/data_card.json']))
    reads=json.loads(read_pinned(root,PARENT+'/artifacts/source_reads_sha256.json',bound[PARENT+'/artifacts/source_reads_sha256.json']))
    return dict(parent_run=PARENT,numeric_run=NUMERIC,seals=SEALS,bound_sha256=bound,full_forward_scope=old['scope'],
        full_forward_allowed_reads=reads,counts=dict(observations=16,known=174,positive=12,negative=162,unknown=338,queries=512),
        spacing=old['scope']['spacing'],teacher='unchanged fixed labels only; original reader loss bundle for scoring verification, never in forward',
        split='same16 fit, no independent holdout data used for selection; no C08-C10',
        full_forward='original same16 reader and complete PRIMITIVE model, GPU no gradient; raw scope identical to section14, no new teacher')

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]) or digest(card['scope'])!=card['scope_sha256']:errors.append('scope drift')
        if card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if a['status']!='APPROVED' or a['scope_sha256']!=card['scope_sha256'] or a['confirmation_reference']!=AUTH:errors.append('authority mismatch')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
