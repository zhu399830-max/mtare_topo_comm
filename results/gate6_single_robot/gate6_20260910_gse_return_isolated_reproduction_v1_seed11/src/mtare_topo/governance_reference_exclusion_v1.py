"""Authority for fixed archived-grid reference exclusion, never training."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = 'v3_reference_exclusion_card_v1'
SLUG = 'gse_reference_exclusion_v1'
POLICY = dict(wall_time_s=1800,host_ram_bytes=4294967296,output_bytes=536870912,
              labels=0,optimizer_steps=0,no_retry=True,matching_radius_m=4.)

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False,('closed exclusion diagnostic card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
        errors.append('exclusion diagnostic only')
    s=card['scope'];a=card['approval']
    if s.get('schema')!='gse_reference_exclusion_scope_v1' or len(s.get('observations',[]))!=30:
        errors.append('fixed30 archived observations required')
    if (len(s.get('parents',[]))!=10 or s.get('independent_traversals')!=10 or s.get('unique_variant_frames')!=150
        or s.get('real_training') is not False or s.get('semantic_label_export') is not False
        or any(not r['task'].split('__')[0].endswith('_C01') for r in s.get('observations',[]))):
        errors.append('C01 source population and no-training boundary required')
    if s.get('query_policy',{}).get('coordinates_per_axis_m')!=[-5.625,-1.875,1.875,5.625] or s.get('query_policy',{}).get('matching_radius_m')!=4.:
        errors.append('fixed lattice/radius required')
    if digest(card['policy'])!=digest(POLICY):errors.append('fixed resource policy required')
    if (card['scope_sha256']!=digest(s) or a.get('scope_sha256')!=digest(s) or a.get('status')!='APPROVED'
        or a.get('authorized_operations')!=['data_export'] or a.get('authorized_gates')!=[3] or not a.get('confirmation_reference')):
        errors.append('exact standing authorization required')
    return ValidationReport(not errors,tuple(errors))
