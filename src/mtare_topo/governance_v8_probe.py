"""Frozen original ten-observation V8 diagnostic, not a training card."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_gse_v8_probe_card_v1'
SLUG = 'gse_v8_original_ten_probe_v1'
SCOPE_SHA = '70a0083a60341ebb6a6a8ea6b82959ca8485fe511a5f73598323e553482f0a85'
POLICY = dict(host_ram_bytes=4294967296, output_bytes=2147483648,
              wall_time_s=10800, gpu_bytes=0, optimizer_steps=0,
              full_label_qualification=False, no_retry=True)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or set(card) != {'schema_version','card_id','operation',
                                              'scope','scope_sha256','policy','approval'}:
        return ValidationReport(False, ('closed original-ten probe card required',))
    if (card['schema_version'],card['card_id'],card['operation']) != (SCHEMA,SLUG,'data_export'):
        errors.append('only original-ten partial target diagnostic allowed')
    if digest(card['scope']) != SCOPE_SHA or card['scope_sha256'] != SCOPE_SHA:
        errors.append('exact original-ten scope mismatch')
    if card['policy'] != POLICY:
        errors.append('fixed resource and qualification policy mismatch')
    a = card['approval']
    if (a.get('status') != 'APPROVED' or a.get('scope_sha256') != SCOPE_SHA
            or a.get('authorized_operations') != ['data_export'] or a.get('authorized_gates') != [3]
            or not a.get('confirmation_reference')):
        errors.append('standing authority must bind this exact diagnostic')
    return ValidationReport(not errors, tuple(errors))
