"""Exact one-episode development capture scope; not training authorization."""
import hashlib
import json
from mtare_topo.governance import ValidationReport

SCHEMA = 'gse_native_structure_capture_card_v1'
IMAGE = 'sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c'


def scope_digest(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_card(card):
    errors = []
    try:
        s, a = card['scope'], card['approval']
        assert card['schema_version'] == SCHEMA
        assert (s['world'], s['seed'], s['robots'], s['episodes']) == ('tunnel', 11, 1, 1)
        assert s['runtime_sec'] == 120 and s['training_steps'] == 0
        assert s['image'] == IMAGE
        assert s['start_xyz_yaw'] == [0, 0, 0, 0]
        assert s['control'] == 'original_mtare_only' and s['bridge_mode'] == 'shadow'
        assert s['source'] == 'new_missing_signal_supplement_not_reconstructed_old_bag'
        assert s['protected_worlds_read'] == [] and s['teacher'] is None
        assert s['sampling'] == dict(max_windows=120, history_frames=5,
            period_sim_ns=1000000000, select='first_eligible_raw_per_second',
            score_dependent=False, history='immediately_previous_four_raw_frames_plus_current',
            actual_raw_frames=None, actual_effective_windows=None)
        assert s['pose_binding'] == 'original_aee_commanded_odometry_full_rotation'
        assert s['full_rotation_policy'] == 'reject_non_yaw_compatible_window_keep_evidence'
        assert s['claim'] == 'shadow_integration_only_not_method_advantage'
        assert a['status'] == 'APPROVED' and a['authorized_operations'] == ['closed_loop_single']
        assert a['authorized_gates'] == [6] and a['approved_by'] == 'user'
        assert '继续执行' in a['confirmation_reference']
        assert a['scope_sha256'] == scope_digest(s)
    except (AssertionError, KeyError, TypeError, ValueError):
        errors.append('native structural capture scope/authorization mismatch')
    return ValidationReport(not errors, tuple(errors))


def validate_finalize_card(card):
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_native_structure_finalize_card_v1'
        assert s['source_run']=='results/gate6_single_robot/gate6_20260913_gse_native_structure_capture_v1_seed11'
        assert (s['world'],s['seed'],s['raw_frames'],s['windows'])==('tunnel',11,610,120)
        assert s['new_simulations']==s['training_steps']==s['resampled_windows']==0
        assert s['repair']=='decode_ros_callerid_bytes_only_no_input_or_model_change'
        assert s['old_failure_preserved'] is True and s['protected_worlds_read']==[]
        assert a['status']=='APPROVED' and a['approved_by']=='user'
        assert a['authorized_operations']==['closed_loop_single'] and a['authorized_gates']==[6]
        assert '继续执行' in a['confirmation_reference'] and a['scope_sha256']==scope_digest(s)
    except (KeyError,TypeError,AssertionError,ValueError):
        errors.append('exact captured-record finalization scope mismatch')
    return ValidationReport(not errors,tuple(errors))


def validate_se3_card(card):
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_native_full_se3_card_v1'
        assert s['source_run']=='results/gate6_single_robot/gate6_20260913_gse_native_structure_capture_v1_seed11'
        assert (s['world'],s['seed'],s['windows'],s['trajectories'])==('tunnel',11,120,1)
        assert s['new_simulations']==s['training_steps']==s['resampled_windows']==0
        assert s['preprocessing_version']=='full_relative_se3_v1'
        assert s['old_results_preserved'] is True and s['protected_worlds_read']==[]
        assert s['model_architecture_changed'] is False and s['checkpoint_selection'] is False
        assert a['status']=='APPROVED' and a['approved_by']=='user'
        assert a['authorized_operations']==['closed_loop_single'] and a['authorized_gates']==[6]
        assert '现在解决' in a['confirmation_reference'] and a['scope_sha256']==scope_digest(s)
    except (KeyError,TypeError,AssertionError,ValueError):
        errors.append('same120 full-SE3 production scope mismatch')
    return ValidationReport(not errors,tuple(errors))


def validate_explicit_source_card(card):
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_explicit_source_capture_card_v1'
        assert (s['world'],s['seed'],s['robots'],s['episodes'],s['runtime_sec'])==('tunnel',11,1,1,120)
        assert s['training_steps']==0 and s['protected_worlds_read']==[] and s['teacher'] is None
        assert s['control']=='original_mtare_only' and s['preprocessing']=='full_relative_se3_v1'
        assert s['sampling']['select']=='all_recorded_native_epochs_exact_callback_bound_five_frames'
        assert s['sampling']['max_windows']==120 and s['sampling']['max_raw_frames']==700
        assert s['old_results_preserved'] is True and s['model_weights_changed'] is False
        assert s['source_pairing']=='explicit_same_callback_headers_no_time_or_index_inference'
        assert a['status']=='APPROVED' and a['approved_by']=='user'
        assert a['authorized_operations']==['closed_loop_single'] and a['authorized_gates']==[6]
        assert a['scope_sha256']==scope_digest(s) and '继续' in a['confirmation_reference']
    except (KeyError,TypeError,AssertionError,ValueError):errors.append('explicit source capture scope mismatch')
    return ValidationReport(not errors,tuple(errors))


def validate_retrieval_card(card):
    errors = []
    try:
        s, a = card['scope'], card['approval']
        assert card['schema_version'] == 'gse_native_token_retrieval_card_v1'
        assert s['source_run'] == 'results/gate6_single_robot/gate6_20260913_gse_explicit_source_capture_v1r1_seed11'
        assert (s['world'], s['seed'], s['trajectories'], s['raw_frames'], s['selected_raw_frames'], s['windows']) == ('tunnel', 11, 1, 617, 575, 115)
        assert s['training_steps'] == s['model_forwards'] == s['new_simulations'] == 0
        assert s['teacher'] is None and s['protected_worlds_read'] == []
        assert s['operation'] == 'saved_token_causal_retrieval_only'
        assert s['threshold_selection'] is False and s['control_changes'] is False
        assert a['status'] == 'APPROVED' and a['approved_by'] == 'user'
        assert a['authorized_operations'] == ['audit'] and a['authorized_gates'] == [6]
        assert a['scope_sha256'] == scope_digest(s) and '继续推进' in a['confirmation_reference']
    except (AssertionError, KeyError, TypeError, ValueError):
        errors.append('saved native token retrieval scope mismatch')
    return ValidationReport(not errors, tuple(errors))


def validate_registration_card(card):
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_native_registration_card_v1'
        assert s['source_run']=='results/gate6_single_robot/gate6_20260913_gse_explicit_source_capture_v1r1_seed11'
        assert s['retrieval_run']=='results/gate6_single_robot/gate6_20260913_gse_native_token_retrieval_v1_seed11'
        assert (s['world'],s['seed'],s['trajectories'],s['windows'],s['candidate_pairs'],s['selected_raw_frames'])==('tunnel',11,1,115,560,575)
        assert s['training_steps']==s['model_forwards']==s['new_simulations']==0
        assert s['teacher'] is None and s['protected_worlds_read']==[]
        assert s['points']=='original_current_raw_finite_returns_within_existing10m'
        assert s['pose_source']=='original_commanded_sensor_poses_not_physical_link_measurements'
        assert s['control_changes'] is False and s['threshold_selection'] is False
        assert a['status']=='APPROVED' and a['approved_by']=='user'
        assert a['authorized_operations']==['audit'] and a['authorized_gates']==[6]
        assert a['scope_sha256']==scope_digest(s) and '继续' in a['confirmation_reference']
    except (AssertionError,KeyError,TypeError,ValueError):errors.append('native source-bound registration scope mismatch')
    return ValidationReport(not errors,tuple(errors))


def validate_visit_wiring_card(card):
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_native_visit_wiring_card_v1'
        assert s['source_run']=='results/gate6_single_robot/gate6_20260913_gse_explicit_source_capture_v1r1_seed11'
        assert (s['world'],s['seed'],s['trajectories'],s['trace_records'],s['epochs'],s['registration_pairs'])==('tunnel',11,1,2024,115,560)
        assert s['registration_run']=='results/gate6_single_robot/gate6_20260914_gse_native_token_registration_v1_seed11'
        assert s['training_steps']==s['model_forwards']==s['registration_calls']==s['new_simulations']==0
        assert s['teacher'] is None and s['protected_worlds_read']==[]
        assert s['scope']=='native_metric_visit_provenance_not_direction_or_place_identity'
        assert a['status']=='APPROVED' and a['approved_by']=='user'
        assert a['authorized_operations']==['audit'] and a['authorized_gates']==[6]
        assert a['scope_sha256']==scope_digest(s) and '继续' in a['confirmation_reference']
    except (AssertionError,KeyError,TypeError,ValueError):errors.append('same-native visit evidence scope mismatch')
    return ValidationReport(not errors,tuple(errors))
