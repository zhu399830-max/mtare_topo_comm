"""Fixed fit-only conditional references, never an independent observable gold."""
import hashlib,json
from pathlib import Path
BINDING='docs/figures/gse_conditional_geometry_fit_v1/fit_cache_binding.json'
LIMITS=dict(workers=4,worker_address_bytes=4*1024**3,total_rss_bytes=32*1024**3,
            wall_seconds=86400,case_seconds=900,output_bytes=8*1024**3,max_candidates=200000,reference_capacity=256)

def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        b=json.loads((Path(__file__).resolve().parents[2]/BINDING).read_text())
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_fit_geometry_card_v1'
        assert (s['parents'],s['observations'],s['frames'],s['patches'],s['roi_returns'])==(60,2880,14400,2366216,99928912)
        assert len(s['entries'])==2880 and s['limits']==LIMITS
        assert s.get('roi_policy','original_world_distance_v1')=='original_world_distance_v1'
        for i,(entry,old) in enumerate(zip(s['entries'],b['entries'])):
            assert entry['identity']==dict(old,case=i)
            assert entry['student_binding']==old and entry['paths']['student']==old['student_path']
            assert old['split']=='fit'
        assert s['target_schema']=='construction_conditioned_geometry_targets_v1'
        assert not s['observability_certified'] and not s['connectivity_certified'] and s['training_steps']==0
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,ValueError,AssertionError,OSError):errors.append('fit conditional geometry scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
