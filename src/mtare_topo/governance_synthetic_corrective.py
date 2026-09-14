from pathlib import Path
import hashlib
from mtare_topo.data.gse_synthetic_corrective import CASES
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_synthetic_matrix import POLICY

SCHEMA='v3_gse_synthetic_corrective_card_v1'
SLUG='gse_synthetic_corrective_v1'
ROOT=Path(__file__).resolve().parents[2]
SOURCE='results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'


def scope():
    seal=ROOT/SOURCE/'artifacts/evidence_sha256.txt'
    if hashlib.sha256(seal.read_bytes()).hexdigest()!='5d491bd1c6c89caeb47ea416eb135fee8a1d8a3507f70d95256cb55a91d9a395':
        raise ValueError('original matrix seal drift')
    pins={p:h for h,p in (line.split('  ',1) for line in seal.read_text().splitlines())}
    inputs={SOURCE+'/artifacts/'+case+ext:pins[SOURCE+'/artifacts/'+case+ext] for case in CASES for ext in ('.npz','.json.gz')}
    return dict(source_run=SOURCE,input_sha256=inputs,case_invalid_counts=CASES,
        primary_observations=4,control_conditions=4,frame_occurrences=20,ray_positions=230400,
        modified_invalid_positions=20,unchanged_valid_positions=230380,
        independent_synthetic_programs=1,section_variants=2,real_worlds=0,
        split='Archived synthetic diagnostic only; no real data or training.',spacing_m=.1,history_frames=5,
        source='Original sealed mesh/scan provenance; opt-in interval winding repairs only missing returns.',
        geometry_settings=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025),
        original_failed_controls_remain_failed=True,full_label_qualification=False)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        s=scope();assert card['schema_version']==SCHEMA and card['card_id']==SLUG and card['operation']=='data_export'
        assert card['scope']==s and card['scope_sha256']==digest(s) and card['policy']==POLICY
        a=card['approval'];assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['data_export'] and a['authorized_gates']==[3] and a['scope']
    except (KeyError,TypeError,AssertionError,ValueError):return ValidationReport(False,('exact four-observation corrective scope required',))
    return ValidationReport(True,())


def validate_card_v2(card):
    from copy import deepcopy
    from mtare_topo.governance import ValidationReport
    if card.get('schema_version')!='v3_gse_synthetic_corrective_card_v2' or card.get('card_id')!='gse_synthetic_corrective_v2':
        return ValidationReport(False,('exact corrective v2 identity required',))
    normalized=deepcopy(card);normalized['schema_version']=SCHEMA;normalized['card_id']=SLUG
    return validate_card(normalized)
