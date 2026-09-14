import json
from pathlib import Path
from mtare_topo.data.gse_partition_cache_binding import RUN,SEAL_SHA,CACHE,CACHE_SHA
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_block_fit_card_v1'
SLUG='gse_block_representation_fit_v1'
POLICY='configs/v3/gate3/gse_block_representation_policy_v1.json'


def scope(root,coordinate_version=1):
    if coordinate_version not in (1,2):raise ValueError('registered coordinate version required')
    root=Path(root)
    seal=read_pinned(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    pins={p:h for h,p in (s.split('  ',1) for s in seal.decode().splitlines())}
    card=json.loads(read_pinned(root,RUN+'/config/data_card.json',pins[RUN+'/config/data_card.json']))
    policy=json.loads((root/POLICY).read_text())
    inputs={p:h for p,h in pins.items() if p.endswith('.npz') or p.endswith('/config/data_card.json')}
    inputs.update({RUN+'/artifacts/evidence_sha256.txt':SEAL_SHA,CACHE:CACHE_SHA})
    if coordinate_version==2:
        prior='results/gate3_semantics/gate3_20260908_gse_block_representation_fit_v1_seed0'
        seal_hash='9993ec4cbbc08521a099ceba8b81fdfafeef7b91477b2f5fd9a0d3bb383bdfdf'
        old_seal=read_pinned(root,prior+'/artifacts/evidence_sha256.txt',seal_hash)
        old_pins={p:h for h,p in (line.split('  ',1) for line in old_seal.decode().splitlines())}
        inputs[prior+'/artifacts/evidence_sha256.txt']=seal_hash
        for rel in ('artifacts/shared_initial.pt','config/schedule.json'):
            inputs[prior+'/'+rel]=old_pins[prior+'/'+rel]
        policy=dict(policy,coordinate_correction=dict(version=2,only_change='10u/sqrt(1+norm(u)^2) anchor output',
            prior_run=prior,compare_initial_and_schedule_exact=True,correction_runs=1,
            not_claimed='Fitting/grouping diagnostic only, not geometric composition contribution'))
    return dict(observations=card['scope']['observations'],input_sha256=inputs,policy=policy,
        real_worlds=0,primary_observations=45,frame_occurrences=225,independent_program_types=6,section_variants=3,
        history_spacing_m=.1,view_offsets_m=[-4.,-2.,0.,2.],population=policy['verified_population'],
        sampling='Same declared45, no new selection; all18 empty-anchor observations retained',
        split='Same synthetic fit/evaluation only; no generalization or independent-world claim',
        teacher='Existing expected_geometry limited prototype reference;27anchor/84opening occurrences, max1anchor per observation; unknown other fields',
        leakage='Authenticated student compact fields only; targets constructed separately; no node/tunnel identity in forward; no C08-C10 reads or checkpoint inference')


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        version=2 if card.get('schema_version')=='v3_block_fit_card_v2' else 1
        s=scope(Path(__file__).resolve().parents[2],version);a=card['approval']
        assert card['schema_version']==('v3_block_fit_card_v2' if version==2 else SCHEMA)
        assert card['card_id']==('gse_block_representation_fit_v2' if version==2 else SLUG) and card['operation']=='training'
        assert card['scope']==s and card['scope_sha256']==digest(s)
        assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['training'] and a['authorized_gates']==[3] and a['confirmation_reference']
    except (KeyError,ValueError,TypeError,AssertionError):return ValidationReport(False,('Exact same45 block representation fit scope required',))
    return ValidationReport(True,())
