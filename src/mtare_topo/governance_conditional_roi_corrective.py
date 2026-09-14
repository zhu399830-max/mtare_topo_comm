import hashlib,json
from pathlib import Path
OLD='results/gate3_semantics/gate3_20260911_gse_conditional_fit_geometry_v1_seed0'
REUSE='docs/figures/gse_conditional_geometry_fit_v1/completed_reference_reuse.json'

def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        root=Path(__file__).resolve().parents[2]
        original=json.loads((root/OLD/'config/data_card.json').read_text())['scope']
        reuse=json.loads((root/REUSE).read_text())['completed_cases']
        expected=dict(original,roi_policy='validated_frozen_student_roi_v1',reuse_cases=reuse)
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_roi_corrective_card_v1' and s==expected
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,ValueError,AssertionError,OSError):errors.append('ROI corrective scope or approval missing')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])
