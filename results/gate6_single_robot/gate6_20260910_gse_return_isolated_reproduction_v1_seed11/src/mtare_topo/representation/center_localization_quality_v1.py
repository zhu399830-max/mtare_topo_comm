"""Prospective reference-relative localization labels, not physical background.

Consumes independently source-bound exclusion/support evidence, never FP masks,
matching results, model scores or query responsibility. Original 4m negatives
are retained as a separate evidence channel.
"""
import numpy as np


def localization_quality(positions, targets, evidence, coverage1, source):
    q=np.asarray(positions,dtype=np.float64);t=np.asarray(targets,dtype=np.float64).reshape(-1,3)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or not np.isfinite(t).all():raise ValueError('finite XYZ required')
    n=len(q);bg=evidence['original_background'];c4=evidence['coverage'];binding=bg['binding']
    if evidence['background_radius_m']!=4. or binding['source']!=source:raise ValueError('original4m source mismatch')
    if not bg['observation_identity_binding_verified'] or bg['inventory_completeness_supplied_not_verified'] or bg['observed_states_supplied_not_verified']:raise ValueError('unverified original source')
    for c in (coverage1,c4):
        if c['source_binding']!=binding or c['grid_content_sha256']!=bg['grid_content_sha256'] or c['reference_inventory_complete_supplied_not_verified']:
            raise ValueError('unbound coverage evidence')
        if not np.array_equal(q,np.asarray(c['query_xyz_m'])):raise ValueError('source coordinates mismatch')
    if coverage1['target_record_sha256']!=c4['target_record_sha256']:raise ValueError('target record mismatch')
    def mask(obj,key):
        a=np.asarray(obj[key]);
        if a.shape!=(n,) or a.dtype!=bool:raise ValueError('aligned boolean '+key)
        return a
    inside=np.linalg.norm(q,axis=1)<=10.
    distances=np.linalg.norm(q[:,None]-t[None],axis=-1)
    near1=(distances<=1.).any(axis=1);near4=(distances<=4.).any(axis=1)
    conflict1=mask(coverage1,'possible_unconfirmed_reference_mask');conflict4=mask(c4,'possible_unconfirmed_reference_mask')
    state=np.asarray(bg['query_observed_states'])
    if state.shape!=(n,) or not np.isin(state,[0,1,2]).all():raise ValueError('query observed states')
    observed=inside&(state!=0)
    # Derive scope from actual coordinates and primitive/reference evidence;
    # independently compare the source snapshot, never use query_scoreable_mask.
    if not np.array_equal(observed,mask(coverage1,'observed_inside_score_region_mask')) or not np.array_equal(observed,mask(c4,'observed_inside_score_region_mask')):raise ValueError('local10m scope mismatch')
    context=inside&near4&~conflict4
    if not np.array_equal(context,mask(c4,'confirmed_structure_coverage_mask')):raise ValueError('structure context source mismatch')
    positive=inside&near1&~conflict1
    background=mask(bg,'reference_negative_mask')
    measured_bad=observed&mask(coverage1,'reference_negative_mask')&~conflict1&~near1
    context_bad=context&~near1&~conflict1
    negative=background|measured_bad|context_bad
    if (positive&negative).any() or (background&conflict1).any():raise ValueError('conflicting localization evidence')
    unknown=~(positive|negative);records=[]
    for i in range(n):
        support='candidate_observed' if observed[i] else 'confirmed_structure_context' if context[i] else 'insufficient'
        reason='confirmed_center_within1m' if positive[i] else 'original4m_background_exclusion' if background[i] else 'observed_reference_exclusion1m' if measured_bad[i] else 'confirmed_structure_offcenter' if context_bad[i] else 'insufficient_evidence'
        records.append(dict(query_slot=i,position_m=q[i].tolist(),label='positive' if positive[i] else 'negative' if negative[i] else 'unknown',
            original_background_negative=bool(background[i]),original_background_reason=bg['reason'][i],observed_state=int(state[i]),support_type=support,
            localization_basis=reason,measured_exclusion1m=bool(measured_bad[i]),structure_context4m=bool(context[i]),
            unknown_competitor1m=bool(conflict1[i]),unknown_competitor4m=bool(conflict4[i]),set_redundancy='not_a_position_label',
            background_claim=bool(background[i]),source_binding=binding,grid_content_sha256=bg['grid_content_sha256'],target_record_sha256=coverage1['target_record_sha256']))
    return dict(schema='center_localization_quality_v1',source=source,positive=positive.tolist(),negative=negative.tolist(),unknown=unknown.tolist(),
        original_background_negative=background.tolist(),counts=dict(positive=int(positive.sum()),negative=int(negative.sum()),unknown=int(unknown.sum())),candidates=records,
        original_background_radius_m=4.,position_tolerance_m=1.,scope_radius_m=10.,scope_center_m=[0.,0.,0.],whole_region_complete=False)
