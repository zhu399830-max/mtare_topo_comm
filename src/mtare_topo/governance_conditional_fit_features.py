"""Exact fit-only existing-cache feature export; not training or teacher authority."""
import hashlib
import json
from pathlib import Path

BINDING = 'docs/figures/gse_conditional_geometry_fit_v1/fit_cache_binding.json'
LIMITS = dict(wall_seconds=3600, host_bytes=4*1024**3, gpu_bytes=28*1024**3, output_bytes=20*1024**3)


def validate_card(card):
    from .governance import ValidationReport
    errors = []
    try:
        root = Path(__file__).resolve().parents[2]
        binding = json.loads((root/BINDING).read_text())
        s,a = card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_fit_features_card_v1'
        assert s['entries']==[dict(e,parent=e['parent_id']) for e in binding['entries']]
        assert (s['parents'],s['frames'],len(s['entries']))==(60,14400,2880)
        assert all(e['split']=='fit' and e['parent'].endswith(tuple(f'C{i:02}' for i in range(1,7))) for e in s['entries'])
        assert s['checkpoint']['sha256']=='8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb'
        assert s['training_steps']==s['teacher_payload_reads']==0 and s['valid_returns_per_observation'] is None
        assert s['limits']==LIMITS
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,ValueError,AssertionError,OSError):errors.append('exact fit feature-export scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
