"""Exact existing860 observation-reader probe, no model or semantic labels."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_bidirectional_reader_probe_card_v1'
SLUG = 'gse_bidirectional_reader_probe_v1'
SCOPE_SHA = '3486367dd04c3b6409e4db978fb0fcb9808946a3823fbfe004d049726f199d94'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or set(card) != {'schema_version','card_id','operation','scope','scope_sha256','approval'}:
        return ValidationReport(False, ('closed reader probe card required',))
    if (card['schema_version'], card['card_id'], card['operation']) != (SCHEMA, SLUG, 'data_export'):
        errors.append('observation reader probe only')
    if digest(card['scope']) != SCOPE_SHA or card['scope_sha256'] != SCOPE_SHA:
        errors.append('exact860 selection/read policy drift')
    a = card['approval']
    if (type(a) is not dict or a.get('status') != 'APPROVED' or a.get('scope_sha256') != SCOPE_SHA
            or a.get('authorized_operations') != ['data_export'] or a.get('authorized_gates') != [3]
            or not a.get('confirmation_reference')):
        errors.append('exact standing scope authorization required')
    return ValidationReport(not errors, tuple(errors))
