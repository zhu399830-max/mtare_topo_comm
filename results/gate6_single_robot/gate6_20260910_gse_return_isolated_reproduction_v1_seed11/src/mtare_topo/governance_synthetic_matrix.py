from mtare_topo.governance_surface_selection import digest
from mtare_topo.data.gse_synthetic_matrix import matrix
SCHEMA='v3_gse_synthetic_matrix_card_v1'
SLUG='gse_synthetic_matrix_v1'
POLICY=dict(host_ram_bytes=4294967296,output_bytes=2147483648,wall_time_s=10800,gpu_bytes=0,optimizer_steps=0,no_retry=True)


def scope():
    return dict(cases=matrix(),primary_observations=144,control_conditions=12,rendered_frame_occurrences=720,
        synthetic_geometry_families=12,section_variants=3,independent_real_worlds=0,
        split='Synthetic software qualification only; no real world, training or test data.',
        spacing_m=.1,history_frames=5,source='Declared finite CSG programs and original sensor renderer; independent prototype controls.',
        covered_geometry_conditions=45,uncovered_primary_conditions=99,
        geometry_settings=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025))


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        assert card['schema_version']==SCHEMA and card['card_id']==SLUG and card['operation']=='data_export'
        assert card['scope']==scope() and card['scope_sha256']==digest(scope()) and card['policy']==POLICY
        a=card['approval'];assert a['status']=='APPROVED' and a['scope_sha256']==card['scope_sha256']
        assert a['authorized_operations']==['data_export'] and a['authorized_gates']==[3] and a['scope']
    except (KeyError,TypeError,AssertionError):return ValidationReport(False,('exact synthetic matrix scope required',))
    return ValidationReport(True,())
