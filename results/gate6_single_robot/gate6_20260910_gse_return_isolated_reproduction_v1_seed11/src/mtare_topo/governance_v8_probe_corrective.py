"""One comparison-only implementation correction; teacher/population fixed."""
from copy import deepcopy
from mtare_topo.governance_v8_probe import SCOPE_SHA, POLICY
from mtare_topo import governance_v8_probe as original

SCHEMA='v3_gse_v8_probe_card_v1r'
SLUG='gse_v8_original_ten_probe_v1r'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    if card.get('schema_version')!=SCHEMA or card.get('card_id')!=SLUG:
        return ValidationReport(False,('only the registered comparison correction allowed',))
    adapted=deepcopy(card)
    adapted['schema_version']=original.SCHEMA;adapted['card_id']=original.SLUG
    return original.validate_card(adapted)
