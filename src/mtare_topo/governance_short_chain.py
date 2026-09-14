"""No-training, one real generated traversal fragment; no structural GT."""
import hashlib,json
from mtare_topo.governance import ValidationReport

def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval'];e=s['identity']
        assert card['schema_version']=='gse_short_observation_chain_v1'
        assert e['task']=='S03_flat_unicyclic_small_C07__ellipse'
        assert e['traversal_id']=='S03_flat_unicyclic_small_C07:edge_0005:d0'
        assert s['frame_rows']==list(range(120,132)) and s['sequence_rows']==list(range(80,88))
        assert s['frames']==12 and s['observations']==8 and s['independent_units']==1
        assert s['labels_generated']==s['training_steps']==0
        assert s['methods']==['range_sectors','observed_surface_fit']
        assert set(s['array_plans'])=={'range_m','valid_mask','sensor_xyz_m','yaw_deg','frame_row','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg'}
        for field,p in s['array_plans'].items():
            assert '/c07/S03_flat_unicyclic_small_C07__ellipse.zarr/'+field==p['prefix'][p['prefix'].index('/c07/'):]
            assert p['selected_rows']==(s['frame_rows'] if field in {'range_m','valid_mask','sensor_xyz_m','yaw_deg'} else s['sequence_rows'])
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError,ValueError):errors.append('short chain exact population/operation mismatch')
    return ValidationReport(not errors,tuple(errors))
