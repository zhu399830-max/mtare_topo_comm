"""Frozen supplementary position selection, never scan or label authority."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_structural_supplement_card_v1'
SLUG = 'gse_structural_supplement_v1'
SCOPE_SHA = '6fc278abeae006ac8b4f8ffb825644ee3675a96c113dc4d5524e9aa34cb42773'
POLICY = dict(wall_time_s=1800, host_ram_bytes=4294967296, output_bytes=536870912,
              gpu_bytes=0, labels=0, optimizer_steps=0, no_retry=True)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or set(card) != {'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False, ('closed supplementary card required',))
    if (card['schema_version'], card['card_id'], card['operation']) != (SCHEMA,SLUG,'data_export'):
        errors.append('supplementary position selection only')
    if digest(card['scope']) != SCOPE_SHA or card['scope_sha256'] != SCOPE_SHA:
        errors.append('independently compiled exact scope drift')
    if digest(card['policy']) != digest(POLICY):
        errors.append('resource/operation policy drift')
    a = card['approval']
    if (type(a) is not dict or a.get('status') != 'APPROVED' or a.get('scope_sha256') != SCOPE_SHA
            or a.get('authorized_operations') != ['data_export'] or a.get('authorized_gates') != [3]
            or not a.get('confirmation_reference')):
        errors.append('standing supplementary authority must bind exact scope')
    return ValidationReport(not errors, tuple(errors))
