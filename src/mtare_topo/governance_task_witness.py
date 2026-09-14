import hashlib,json
from mtare_topo.governance import ValidationReport

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version'] in ('gse_direction_task_witness_replay_v1','gse_direction_task_witness_replay_v2','gse_direction_task_decomposition_v1')
        if card['schema_version']=='gse_direction_task_decomposition_v1':
            assert s['support_policy']=='four_neighbor_ray_components_reaching_existing10m_boundary_no_pruning_residuals_preserved'
            assert s['control_run']=='results/gate3_semantics/gate3_20260913_gse_direction_task_witness_replay_v2_seed0'
        if card['schema_version'].endswith('_v2'):assert s['support_policy']=='exact_detector_connected_columns_v2'
        assert (s['observations'],s['unique_frames'],s['parents'])==(12,24,3)
        assert s['training_steps']==s['model_forwards']==0 and s['split']=='fit'
        assert s['training_qualified'] is False and s['independent_structural_accuracy_qualified'] is False
        assert s['methods']==['POSE_DIRECTION','A','B','C'] and len(s['manifest'])==12
        entries=s['source_scope']['entries']
        assert all(e['source']['task']==x['task'] and e['source']['frame_rows']==x['frame_rows'] for e,x in zip(s['manifest'],entries))
        assert {e['source']['parent'] for e in s['manifest']}=={'S08_3d_loop_rich_C01','S04_3d_unicyclic_small_C06','S08_3d_loop_rich_C06'}
        assert all(e['source']['split']=='fit' for e in s['manifest'])
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('task witness exact scope mismatch')
    return ValidationReport(not errors,tuple(errors))
