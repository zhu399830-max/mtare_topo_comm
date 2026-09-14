from pathlib import Path
from .data.gse_synthetic_fit_cache import compile_scope as cached_scope
from .governance_synthetic_fit import POLICY
from .governance_surface_selection import digest
SCHEMA='v3_gse_synthetic_window_fit_card_v1'
SLUG='gse_synthetic_window_fit_v1'


def compile_scope(root):
    s=cached_scope(root);s['training_contract']='window_surface_set_v1'
    s['corrective_changes']=dict(opening_domain='10m sphere surface',training_matching='distance/10-presence_probability',no_object_weight=.1,
        scoring='geometry-only1m; confidence0.5; all selected sphere openings count; no radial discard',
        attribution='joint finite interface correction, not proof of which component causes gain')
    return s


def validate_card(card):
    from .governance import ValidationReport
    try:
        s=compile_scope(Path(__file__).resolve().parents[2]);a=card['approval']
        assert card['schema_version']==SCHEMA and card['card_id']==SLUG and card['operation']=='training'
        assert card['scope']==s and card['scope_sha256']==digest(s) and card['policy']==POLICY
        assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['training'] and a['authorized_gates']==[3] and a['confirmation_reference']
    except (AssertionError,KeyError,ValueError,TypeError):return ValidationReport(False,('exact window corrective fit scope required',))
    return ValidationReport(True,())
