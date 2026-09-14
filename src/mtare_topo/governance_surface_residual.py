"""Fixed16 residual-only diagnostic; no assignment acceptance threshold."""
import hashlib
import json


def validate_new12_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];b=s['binding']
        assert card['schema_version']=='gse_new12_surface_residual_card_v1'
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert (b['observations'],b['unique_frames'],b['independent_parents'],b['effective_valid_returns'])==(12,60,12,684440)
        assert b['unique_source_chunks']==40 and b['decoded_unique_chunk_bytes']==7056896
        assert len(b['entries'])==12 and [e['observation'] for e in b['entries']]==list(range(12))
        assert len({e['identity']['parent'] for e in b['entries']})==12
        assert all(e['identity']['split']=='fit' and any('_C%02d__'%k in e['identity']['task'] for k in range(1,7)) for e in b['entries'])
        assert s['roi_radius_m']==10 and s['labels_generated']==0 and s['training_steps']==0
        assert s['acceptance_distance_m'] is None and s['mesh_coordinates']=='original_float32_scene'
        assert s['limits']==dict(wall_seconds=10800,host_bytes=4*1024**3,output_bytes=2*1024**3,max_candidates=200000)
    except (KeyError,TypeError,AssertionError):errors.append('new12 surface scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval'];b=s['binding']
        assert card['schema_version']=='gse_surface_residual_card_v1'
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert b['observations']==16 and b['unique_frames']==80 and len(b['parents'])==8
        assert b['decoded_unique_chunk_bytes']==8091520 and len(b['entries'])==16
        assert s['roi_radius_m']==10 and s['labels_generated']==0 and s['training_steps']==0
        assert s['acceptance_distance_m'] is None and s['mesh_coordinates']=='original_float32_scene'
        assert s['limits']==dict(wall_seconds=10800,host_bytes=4*1024**3,output_bytes=2*1024**3,max_candidates=200000)
    except (KeyError,TypeError,AssertionError): errors.append('fixed16 residual contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def validate_interval_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_surface_interval_card_v1'
        assert len(s['entries'])==16 and s['parents']==8 and s['unique_frames']==80
        assert a['status']=='APPROVED' and a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        assert s['labels_generated']==0 and s['residual_threshold'] is None
        assert s['limits']==dict(wall_seconds=600,host_bytes=2*1024**3,output_bytes=512*1024**2)
    except (KeyError,TypeError,AssertionError):errors.append('fixed16 conditional interval contract drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
