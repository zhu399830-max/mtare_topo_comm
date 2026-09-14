"""Exact one-scene live diagnostic card; frame counts are measured, not invented."""
IMAGE='sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c'
SCHEMA='v3_live_geometry_shadow_card_v1'
SLUG='gse_live_geometry_shadow_v1'
SCOPE=dict(worlds=['tunnel'],episodes=1,seed=11,observation_wall_seconds=60,
    total_wall_limit_seconds=240,raw_frames=None,effective_observations=None,
    sampling='All delivered organized scans; five causal consecutive received frames; count after capture',
    independent_units='One development episode, no independent test or training samples',
    split='historically used development world only; no held-out evaluation',
    teacher='none; no structural identities or map geometry in frontend',
    protected_worlds_read=False,training=False,control_published=False,image=IMAGE)

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    expected=dict(SCOPE)
    schema=SCHEMA
    if card.get('schema_version') in {'v3_live_geometry_shadow_card_v2','v3_live_geometry_execution_card_v1','v3_live_geometry_execution_card_v2','v3_live_geometry_execution_card_v3','v3_live_geometry_execution_card_v4'}:
        schema=card['schema_version']
        expected.update(pose_interface='aee-commanded-odometry',past_pose_max_age_s=.1,
                        actual_link_pose_measured=False)
    execution=schema in {'v3_live_geometry_execution_card_v1','v3_live_geometry_execution_card_v2','v3_live_geometry_execution_card_v3','v3_live_geometry_execution_card_v4'}
    if execution:expected.update(control_published=True,execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1))
    if schema in {'v3_live_geometry_execution_card_v2','v3_live_geometry_execution_card_v3','v3_live_geometry_execution_card_v4'}:expected['rigid_motion']=True
    if schema in {'v3_live_geometry_execution_card_v3','v3_live_geometry_execution_card_v4'}:expected['branch_policy']=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3)
    if schema=='v3_live_geometry_execution_card_v4':expected['remote_returns']=True
    if card.get('schema_version')!=schema or card.get('scope')!=expected:
        errors.append('live geometry single tunnel shadow scope drift')
    a=card.get('approval',{})
    if (a.get('status')!='APPROVED' or a.get('authorized_operations')!=['closed_loop_single' if execution else 'shadow']
            or a.get('authorized_gates')!=[6 if execution else 5] or not a.get('confirmation_reference')):
        errors.append('standing development scope binding required')
    return ValidationReport(not errors,tuple(errors),())
