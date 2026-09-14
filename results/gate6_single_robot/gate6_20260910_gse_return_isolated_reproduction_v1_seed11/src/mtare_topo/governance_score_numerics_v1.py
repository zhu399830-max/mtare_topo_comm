"""Exact sealed-cache-only authorization for one affine numerical solve."""
import json
from pathlib import Path
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_score_numerics_card_v1';SLUG='gse_fixed_score_numerics_v1r'
PARENT='results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
SEAL='300b8ca66a275c40b40850f557b76ca709741f2ea324f0fe107d15f98a30af70'
AUTH='20260910 user: reuse section14 frozen features and labels; linear separability of174known candidates, verify original loss and equivalent weights; one fullbatch numerical solve of same129parameter head; unknown excluded; all512 original scorer replay; distinguish numeric unresolved, nonseparable, insufficient optimization, labels pass but detection fails; no A/B,position,1000batch repeats,network expansion or mapping.'
POLICY=dict(question='Are fixed known labels linearly separable, and can a fullbatch solve of the unchanged loss yield correct detection?',
    known=174,positive=12,negative=162,unknown=338,queries=512,head_parameters=129,
    loss='schedule-frequency weighted per-observation equal positive/negative group means; original softplus; wd0; unknown weight0',
    initialization='section14 final head, frozen features; no restart',
    solver='one float64 L-BFGS-B maxiter20000 maxfun50000 ftol1e-15 gtol1e-10 maxls50 maxcor30; invertible full129D SVD preconditioning, no directions dropped',
    separability='HiGHS signed margin>=1; exact rational primal or dual verification; otherwise numeric undetermined',
    label_pass='all174 known signs correct at original threshold.5; float64 and original float32 separately',
    detection_pass='original partial-reference1m P/R>=.90; all512, original.5/1/2/4m coverage, unknown separate',
    optimization='success plus original/conditioned gradient_inf<=1e-8 and objective coordinate discrepancy<=1e-9; no finite minimizer claim for separable logistic',
    limits=dict(cpu_wall_s=600,host_ram_bytes=8*1024**3,output_bytes=1024**3,gpu_bytes=0),no_expansion=True)

def compile_scope(root):
    failed='results/gate3_semantics/gate3_20260910_gse_fixed_score_numerics_v1_seed0'
    failure_seal='481bb0732f2bd85ebfc7657e04e11657bcf1935c7ef63562c5a6fa819c2576b1'
    read_pinned(root,failed+'/artifacts/evidence_sha256.txt',failure_seal)
    idx={p:h for h,p in (l.split('  ',1) for l in read_pinned(root,PARENT+'/artifacts/evidence_sha256.txt',SEAL).decode().splitlines())}
    bound={p:h for p,h in idx.items() if any(p.startswith(PARENT+'/'+prefix) for prefix in (
        'artifacts/fixed_candidates_','artifacts/input_','artifacts/prediction_0000_','artifacts/prediction_1000_',
        'metrics/evaluation_','metrics/candidate_scores_','config/schedule.json','config/data_card.json','config/run_spec.json','checkpoints/step_1000.pt'))}
    old=json.loads(read_pinned(root,PARENT+'/config/data_card.json',idx[PARENT+'/config/data_card.json']))
    scope={k:old['scope'][k] for k in ('selection','selected_rows','counts','spacing','split')}
    scope.update(parent_run=PARENT,parent_seal_sha256=SEAL,bound_sha256=bound,
        zero_solve_software_failure=dict(run=failed,seal_sha256=failure_seal,lp_calls=0,minimization_calls=0,correction='tuple/list JSON value normalization only'),
        actual_read_scope='only sealed section14 cache/labels/schedule/predictions and evaluation masks; no original raw source or teacher access',
        candidate_counts=dict(all=512,known=174,positive=12,negative=162,unknown=338),
        teacher='unchanged fixed section14 geometry assignment and repaired masks; no new labels',
        leakage='same16 fit only; no calibration or C08-C10; identities for alignment only, not head input')
    return scope

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]) or digest(card['scope'])!=card['scope_sha256']:errors.append('scope drift')
        if card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if a['status']!='APPROVED' or a['scope_sha256']!=card['scope_sha256'] or a['confirmation_reference']!=AUTH:errors.append('authorization mismatch')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
