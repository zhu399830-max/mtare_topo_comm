"""Original ten cases, explicit precision policy; no population expansion."""
from copy import deepcopy
from mtare_topo import governance_v8_probe as original
SCHEMA='v3_gse_ten_precision_card_v1'
SLUG='gse_original_ten_precision_v1'
POLICY=original.POLICY
SCOPE_SHA=original.SCOPE_SHA


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    if card.get('schema_version')!=SCHEMA or card.get('card_id')!=SLUG:
        return ValidationReport(False,('exact original-ten precision contract required',))
    adapted=deepcopy(card)
    adapted['schema_version']=original.SCHEMA;adapted['card_id']=original.SLUG
    return original.validate_card(adapted)
