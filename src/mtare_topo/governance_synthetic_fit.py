from pathlib import Path
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_gse_synthetic_fit_card_v1'
SLUG='gse_synthetic_fit_v1'
POLICY=dict(host_ram_bytes=32*1024**3,gpu_vram_bytes=28*1024**3,output_bytes=30*1024**3,
            wall_time_s=43200,no_retry=True,real_worlds=0)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        s=compile_scope(Path(__file__).resolve().parents[2])
        assert card['schema_version']==SCHEMA and card['card_id']==SLUG and card['operation']=='training'
        assert card['scope']==s and card['scope_sha256']==digest(s) and card['policy']==POLICY
        a=card['approval']
        assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['training'] and a['authorized_gates']==[3] and a['confirmation_reference']
    except (KeyError,TypeError,ValueError,AssertionError):
        return ValidationReport(False,('exact synthetic-only fitting scope required',))
    return ValidationReport(True,())
