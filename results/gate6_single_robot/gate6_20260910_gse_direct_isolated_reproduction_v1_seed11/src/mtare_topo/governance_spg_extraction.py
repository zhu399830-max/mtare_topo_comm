from pathlib import Path
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_axis_pair_diagnostic import scope as archived_scope

SCHEMA='v3_spg_extraction_card_v1'
SLUG='gse_spg_paired_extraction_v1'


def scope(root):
    s=archived_scope(root)
    for key in ('teacher','thresholds_m'):
        s.pop(key,None)
    s.update(method='R1 fixed0.5m voxel PCA; R2 official geof45/adj10/reg0.1/voxel0.03 plus induced connected adapter; common raw-return attributes',
        teacher='NONE; archived metadata authenticates selection only, no labels generated or scored',
        roi_radius_m=10.,max_groups=4096,max_points=57600,coordinate_dtype='float32 shared before ROI',
        split='Same45 synthetic representation extraction only; no held-out or detector claim',
        fallback='Preserve EMPTY/INSUFFICIENT as explicit unknown, never drop observation; fail on invalid input, native output or capacity; no retry',
        acceptance='All45 source mappings and counts retained; finite common attributes, native raw and connected partitions preserved; not semantic accuracy')
    return s


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        s=scope(Path(__file__).resolve().parents[2])
        assert card['schema_version']==SCHEMA and card['card_id']==SLUG
        assert card['operation']=='audit' and card['scope']==s
        assert card['scope_sha256']==digest(s)
        a=card['approval']
        assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['audit'] and a['authorized_gates']==[3]
        assert a['confirmation_reference']
    except (KeyError,ValueError,TypeError,AssertionError):
        return ValidationReport(False,('Exact same45 representation-only extraction scope required',))
    return ValidationReport(True,())
