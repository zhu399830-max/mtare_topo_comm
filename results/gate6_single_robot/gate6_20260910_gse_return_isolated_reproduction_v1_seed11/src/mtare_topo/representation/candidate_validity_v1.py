"""Versioned position-validity labels; duplicate assignment is not a negative."""
import numpy as np

def candidate_validity(positions,targets,evidence,coverage1,source,old_masks):
    q=np.asarray(positions,dtype=float);t=np.asarray(targets,dtype=float).reshape(-1,3);n=len(q)
    background=evidence['original_background'];coverage4=evidence['coverage']
    if evidence['background_radius_m']!=4. or background['binding']['source']!=source or coverage1['source_binding']['source']!=source or coverage4['source_binding']['source']!=source:
        raise ValueError('source or original4m evidence drift')
    if not background['observation_identity_binding_verified'] or background['inventory_completeness_supplied_not_verified'] or background['observed_states_supplied_not_verified']:
        raise ValueError('unbound reference exclusion evidence')
    if not np.array_equal(q,np.asarray(coverage1['query_xyz_m'])) or not np.array_equal(q,np.asarray(coverage4['query_xyz_m'])):raise ValueError('candidate evidence coordinates differ')
    near=(np.linalg.norm(q[:,None]-t[None],axis=-1)<=1.).any(axis=1)
    conflict=np.asarray(coverage1['possible_unconfirmed_reference_mask'],bool)
    negative=np.asarray(background['reference_negative_mask'],bool)
    positive=near&~conflict
    if positive.shape!=(n,) or negative.shape!=(n,) or conflict.shape!=(n,):raise ValueError('candidate count mismatch')
    if (positive&negative).any():raise ValueError('POSITIVE_NEGATIVE_EVIDENCE_CONFLICT slots '+str(np.flatnonzero(positive&negative).tolist()))
    if (negative&conflict).any():raise ValueError('NEGATIVE_UNKNOWN_COMPETITOR_CONFLICT')
    unknown=~(positive|negative);records=[];transition={}
    duplicate=np.asarray(evidence['used_duplicate_mask'],bool)
    for i in range(n):
        old=next(k for k in ('positive','negative','unknown') if old_masks[k][i])
        new='positive' if positive[i] else 'negative' if negative[i] else 'unknown'
        reason='WITHIN1M_CONFIRMED_NO_UNKNOWN_CONFLICT' if positive[i] else background['reason'][i] if negative[i] else 'UNKNOWN_REFERENCE_COMPETITOR' if near[i]&conflict[i] else 'DUPLICATE_NOT_POSITION_NEGATIVE' if duplicate[i] else 'NO_POSITION_VALIDITY_EVIDENCE'
        transition[old+'->'+new]=transition.get(old+'->'+new,0)+1
        records.append(dict(query_slot=i,position_m=q[i].tolist(),old_label=old,new_label=new,reason=reason,old_duplicate=bool(duplicate[i]),independent_negative=bool(negative[i]),unknown_competitor=bool(conflict[i])))
    return dict(schema='candidate_validity_v1',positive=positive.tolist(),negative=negative.tolist(),unknown=unknown.tolist(),
        counts=dict(positive=int(positive.sum()),negative=int(negative.sum()),unknown=int(unknown.sum()),old_used_duplicates=int(duplicate.sum())),
        transition=transition,candidates=records,source=source,source_binding=background['binding'],grid_content_sha256=background['grid_content_sha256'],background_radius_m=4.,positive_radius_m=1.)
