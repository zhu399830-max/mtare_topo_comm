"""Fixed-population spatial coverage authority, no scan or label generation."""
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_surface_coverage_card_v1'
SLUG='gse_surface_coverage_v1'
POLICY=dict(radius_m=10., wall_time_s=1800, host_ram_bytes=4294967296,
            output_bytes=536870912, labels=0, optimizer_steps=0, no_retry=True)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False,('closed coverage card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
        errors.append('coverage-only operation required')
    s=card['scope']; a=card['approval']
    if type(s) is not dict or s.get('schema')!='gse_surface_coverage_scope_v1':
        return ValidationReport(False,('exact coverage scope required',))
    counts=s.get('counts',{})
    for key,n in dict(parents=70,tasks=210,physical_edge_units=1120,observations=3360,
                      selected_current_xyz_rows=3360,construction_files=210,xyz_headers=210,
                      pose_chunk_files=264,decoded_xyz_rows_including_collateral=656520,
                      decoded_padded_xyz_bytes=19927728,actual_payload_reads=0,scans=0,labels=0).items():
        if type(counts.get(key)) is not int or counts[key]!=n:errors.append('fixed count '+key)
    if digest(card['policy'])!=digest(POLICY):errors.append('policy drift')
    if (card['scope_sha256']!=digest(s) or type(a) is not dict or a.get('scope_sha256')!=digest(s)
        or a.get('status')!='APPROVED' or a.get('authorized_operations')!=['data_export']
        or a.get('authorized_gates')!=[3] or not a.get('confirmation_reference')):
        errors.append('exact standing authorization required')
    return ValidationReport(not errors,tuple(errors))
