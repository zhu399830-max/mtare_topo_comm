"""Per-query frozen prediction audit; no optimizer or teacher regeneration."""
from types import SimpleNamespace
import json
import numpy as np
import torch
from scipy.special import expit
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.grouping_center_training_v1 import center_objective as old_objective
from mtare_topo.representation.grouping_supervision_v2 import center_objective as new_objective


def audit_observation(observation, data, metric):
    target=observation.loss_only['target'].position_m.cpu().numpy()
    xyz=np.asarray(data['position_m']);logit=np.asarray(data['presence_logits'])
    results={};gradients={}
    # Output-space autograd compares exactly the old/new masks and normalization.
    # Parameter gradients are separately recorded by the real training executor.
    with torch.enable_grad():
        for name,fn in [('old',old_objective),('new',new_objective)]:
            p=SimpleNamespace(position_m=torch.tensor(xyz,requires_grad=True),
                presence_logits=torch.tensor(logit,requires_grad=True))
            r=fn(SimpleNamespace(prediction=p),observation)
            pg,lg=torch.autograd.grad(r['total'],(p.position_m,p.presence_logits),allow_unused=True)
            results[name]=r
            gradients[name]=(np.zeros_like(xyz) if pg is None else pg.detach().numpy(),
                np.zeros_like(logit) if lg is None else lg.detach().numpy())
    old,new=results['old'],results['new']
    matched=set(old['assignment'].tolist())
    if matched!=set(new['assignment'].tolist()):raise AssertionError('assignment changed')
    coverage=new['evidence']['coverage']
    d=np.linalg.norm(xyz[:,None]-target[None],axis=2)
    keep=metric['query_indices'];pairs={keep[i]:j for i,j in metric['pairs']}
    scoreable=dict(zip(keep,metric['coverage']['query_scoreable_mask']))
    pre=np.asarray(data['query_positions_m'])+np.asarray(data['residual_m'])
    rows=[]
    for i in range(len(xyz)):
        state=('TP' if i in pairs else ('FP' if scoreable.get(i) else 'IGNORED')) if i in keep else 'NOT_SELECTED'
        rows.append(dict(slot=i,position_m=xyz[i].tolist(),query_position_m=data['query_positions_m'][i].tolist(),
            query_source_index=int(data['query_indices'][i]),residual_m=data['residual_m'][i].tolist(),
            projected=bool(np.linalg.norm(pre[i])>10.),preprojection_norm_m=float(np.linalg.norm(pre[i])),
            probability=float(expit(logit[i])),selected=i in keep,metric_state_1m=state,
            nearest_target_distance_m=float(d[i].min()) if len(target) else None,
            training_positive=i in matched,
            old_negative=bool(old['negative_mask'][i]),new_negative=bool(new['negative_mask'][i]),
            observed_inside=bool(coverage['observed_inside_score_region_mask'][i]),
            unconfirmed_conflict=bool(coverage['possible_unconfirmed_reference_mask'][i]),
            confirmed_structure_4m=bool(coverage['confirmed_structure_coverage_mask'][i]),
            duplicate_candidate=bool(new['evidence']['duplicate_candidate_mask'][i]),
            old_presence_gradient=float(gradients['old'][1][i]),new_presence_gradient=float(gradients['new'][1][i]),
            old_position_gradient=gradients['old'][0][i].tolist(),new_position_gradient=gradients['new'][0][i].tolist()))
    targets=[]
    for j,t in enumerate(target):
        nearest=int(d[:,j].argmin());assigned=int(old['assignment'][j])
        selected=np.asarray(keep,dtype=int)
        targets.append(dict(reference_index=j,position_m=t.tolist(),nearest_slot=nearest,
            source_query_nearest_distance_m=float(np.linalg.norm(data['query_positions_m']-t,axis=1).min()),
            raw_nearest_distance_m=float(d[nearest,j]),assigned_slot=assigned,
            assigned_distance_m=float(d[assigned,j]),assigned_probability=float(expit(logit[assigned])),
            selected_nearest_distance_m=float(d[selected,j].min()) if len(selected) else None,
            matched_1m=j in pairs.values(),
            failure_stage=('none' if j in pairs.values() else 'raw_localization' if d[:,j].min()>1.
                else 'presence_selection' if not len(selected) or d[selected,j].min()>1. else 'scoring_or_matching')))
    return dict(source=observation.source,targets=targets,queries=rows,
        losses={name:{k:float(r[k].detach()) for k in ('total','position','presence')} for name,r in results.items()},
        counts={name:{k:r[k] for k in ('positive_count','negative_count','unknown_count')} for name,r in results.items()},
        evidence=new['evidence'],score_1m={k:v for k,v in metric.items() if k!='coverage'})


def audit_step(examples, run, step):
    metrics=json.loads((run/'metrics'/f'evaluation_{step:04d}.json').read_text())
    by_source={digest(r['source']):r for r in metrics['observations']}
    rows=[]
    for o in examples:
        key=digest(o.source)
        with np.load(run/'artifacts'/f'prediction_{step:04d}_{key}.npz') as data:
            rows.append(audit_observation(o,data,by_source[key]['scores']['1.0']))
    return dict(step=step,summary=metrics['summary'],observations=rows)


def case_outcomes(old, new):
    current={digest(o['source']):o for o in new['observations']}
    cases=[]
    for obs in old['observations']:
        other=current[digest(obs['source'])]
        for target in obs['targets']:
            if target['matched_1m']:continue
            after=other['targets'][target['reference_index']]
            cases.append(dict(kind='old_missed_reference',source=obs['source'],before=target,after=after,
                resolved=after['matched_1m']))
        extras=[q for q in obs['queries'] if q['metric_state_1m']=='FP' and not q['training_positive'] and not q['old_negative']]
        for before in extras:
            after=other['queries'][before['slot']]
            # Query slots are computation slots, not persistent physical identities.
            # Report replacements/unknowns; moving a slot to unknown is not success.
            remaining=[q['slot'] for q in other['queries'] if q['metric_state_1m']=='FP']
            ignored=[q['slot'] for q in other['queries'] if q['metric_state_1m']=='IGNORED']
            cases.append(dict(kind='old_unpenalized_extra',source=obs['source'],before=before,after=after,
                current_known_fp_slots=remaining,current_ignored_slots=ignored,
                resolved_in_known_domain=after['metric_state_1m'] in ('TP','NOT_SELECTED') and not remaining,
                unknown_is_not_certified_absence=True,slot_is_not_persistent_identity=True))
    return cases
