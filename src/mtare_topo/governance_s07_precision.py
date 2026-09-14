"""One original S07 defect correction diagnostic, never a population pass."""
from mtare_topo.governance_surface_selection import digest
from mtare_topo.data.gse_s07_precision_scope import TASK, SEQUENCE
SCHEMA='v3_gse_s07_precision_card_v1'
SLUG='gse_s07_precision_corrective_v1'
POLICY=dict(host_ram_bytes=4294967296,output_bytes=2147483648,wall_time_s=1800,
            gpu_bytes=0,optimizer_steps=0,full_label_qualification=False,no_retry=True)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        assert set(card)=={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}
        assert (card['schema_version'],card['card_id'],card['operation'])==(SCHEMA,SLUG,'data_export')
        s=card['scope'];assert len(s['entries'])==1
        row=s['entries'][0]['source']
        assert (row['task'],row['source_sequence_id'],row['frame_rows'])==(TASK,SEQUENCE,[3084,3085,3086,3087,3088])
        assert card['scope_sha256']==digest(s) and card['policy']==POLICY
        assert s['counts']['observations']==1 and s['counts']['parents']==1
        assert s['geometry_settings']==dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025)
        a=card['approval'];assert a['status']=='APPROVED' and a['scope_sha256']==card['scope_sha256']
        assert a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['confirmation_reference']
    except (AssertionError,KeyError,TypeError):
        return ValidationReport(False,('exact single S07 corrective contract required',))
    return ValidationReport(True,())
