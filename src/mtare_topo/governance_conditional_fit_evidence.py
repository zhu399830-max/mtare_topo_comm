"""Fit-only missing-reference transport, not a geometry label run."""
import hashlib,json
from pathlib import Path
SOURCE='configs/v3/gate3/gse_conditional_fit_reference_source_scope_v1.json'
LIMITS=dict(wall_seconds=900,host_bytes=4*1024**3,output_bytes=2*1024**3)

def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        original=json.loads((Path(__file__).resolve().parents[2]/SOURCE).read_text())
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_fit_evidence_card_v1'
        assert s['observations']==[dict(e,parent=e['parent_id'],source_global_sequence_index=e['source_sequence_id']) for e in original['entries']]
        for k in ('array_plans','task_files','input_sha256'):assert s[k]==original[k]
        assert s['limits']==LIMITS and len(s['observations'])==2880
        assert s['labels_generated']==s['training_steps']==0
        assert a['status']=='APPROVED' and a['authorized_gates']==[3] and a['authorized_operations']==['data_export']
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,ValueError,AssertionError,OSError):errors.append('fit evidence scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
