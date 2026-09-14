"""Fixed2676 observed-support diagnostics, never semantic label authority."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_query_coverage_probe_card_v2'
SLUG = 'gse_query_coverage_probe_v2'
SCOPE_SHA = '321c1947b813b623e44fa26c830b6f28951266f48eca12ec5a7700fdf0bf4d7f'
POLICY = dict(task='S01_flat_tree_small_C01__c1_mixed',sequence=225,extra_range_error_m=0.,radius_m=10.,wall_time_s=1800, host_ram_bytes=4294967296, output_bytes=536870912,
              gpu_bytes=0, labels=0, optimizer_steps=0, no_retry=True, query_offsets_m=[[0.,0.,0.],[0.1,0.,0.]], diagnostic_radii_m=[1.,2.,4.], predictions_are_model_outputs=False)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or set(card) != {'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False, ('closed supplementary card required',))
    if (card['schema_version'], card['card_id'], card['operation']) != (SCHEMA,SLUG,'data_export'):
        errors.append('supplementary observation support only')
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



