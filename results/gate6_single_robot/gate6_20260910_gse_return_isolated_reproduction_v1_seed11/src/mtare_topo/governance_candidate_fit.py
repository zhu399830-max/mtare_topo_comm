from pathlib import Path
import json
from mtare_topo.data.gse_candidate_cache_binding import RUN,SEAL_SHA,authenticated_read,CACHE,CACHE_SHA
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_candidate_fit_card_v1'
SLUG='gse_candidate_fit_v1'


def scope(root,corrective=False):
    pins={p:h for h,p in (s.split('  ',1) for s in authenticated_read(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA).decode().splitlines())}
    card=json.loads(authenticated_read(root,RUN+'/config/data_card.json',pins[RUN+'/config/data_card.json']))
    inputs={p:h for p,h in pins.items() if p.endswith('.json') and ('/artifacts/' in p or p.endswith('/config/data_card.json'))}
    inputs[RUN+'/artifacts/evidence_sha256.txt']=SEAL_SHA;inputs[CACHE]=CACHE_SHA
    result=dict(observations=card['scope']['observations'],input_sha256=inputs,real_worlds=0,
        primary_observations=45,frame_occurrences=225,independent_program_types=6,section_variants=3,
        history_spacing_m=.1,view_offsets_m=[-4.,-2.,0.,2.],candidates=32469,anchors=27,
        positive_support=468,negative_background=32001,
        split='Same synthetic fit/evaluation; no generalization or real-world research claim',
        teacher='Independent declared fixture geometry; targets only in loss/scoring, no hidden surfaces',
        leakage='No real-world files/checkpoints read; encoder pretraining inherited from authenticated cache; no GT candidate filtering',
        methods=['A','B','C'],seed=0,updates_per_method=500,microbatch=1,accumulation=4,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        objective='1m linear support heatmap; positive/background BCE groups averaged separately',
        output='Fixed positions,confidence0.5,NMS1m; raw and suppressed1/2/4m scores; no best epoch',
        resource_caps=dict(host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3,wall_s=43200),
        acceptance='Final suppressed1m F1>=0.90 is interface fit only; report allABC and raw results; no research gate advance')
    if corrective:
        result.update(objective='One-to-one feasible-cardinality/distance-confidence matching, positive BCE, hardest max(target_count,1) known backgrounds',
            output='Fixed positions,confidence0.5,RAW selection primary; NMS1m diagnostic only;1/2/4m matching',
            acceptance='Final RAW1m F1>=0.90 is interface fit only; allABC; no threshold search or further loss retries',
            corrective_version='candidate_set_objective_v1',corrective_budget_exhausted_after_this_run=True)
    return result


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        corrective=card['card_id']=='gse_candidate_set_fit_v1'
        s=scope(Path(__file__).resolve().parents[2],corrective);a=card['approval']
        assert card['schema_version']==SCHEMA and card['card_id'] in (SLUG,'gse_candidate_set_fit_v1') and card['operation']=='training'
        assert card['scope']==s and card['scope_sha256']==digest(s)
        assert a['status']=='APPROVED' and a['scope_sha256']==digest(s)
        assert a['authorized_operations']==['training'] and a['authorized_gates']==[3] and a['confirmation_reference']
    except (KeyError,ValueError,TypeError,AssertionError):return ValidationReport(False,('Exact sealed candidate fit scope required',))
    return ValidationReport(True,())
