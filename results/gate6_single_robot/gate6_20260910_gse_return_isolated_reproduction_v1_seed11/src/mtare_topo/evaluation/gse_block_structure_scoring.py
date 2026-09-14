"""Independent raw-set scoring; no training assignments or NMS reused."""
import numpy as np
import torch
from .gse_synthetic_field_scoring import match_positions


def prediction_record(prediction):
    def array(t):
        if not torch.isfinite(t).all():raise ValueError('nonfinite prediction')
        return t.detach().cpu().numpy()
    r=dict(coordinate_frame='current_sensor_m',observation_supported=prediction.observation_supported,
        unknown_fields=list(prediction.unknown_fields))
    for kind in ('anchor','opening'):
        xyz=array(getattr(prediction,kind+'_position_m'))
        logits=getattr(prediction,kind+'_presence_logits')
        array(logits)  # sigmoid(inf) would hide nonfinite raw model output.
        score=array(logits.sigmoid())
        keep=(score>=.5)&prediction.observation_supported
        r[kind+'_all_positions_m']=xyz.tolist();r[kind+'_all_scores']=score.tolist()
        r[kind+'_selected_indices']=np.flatnonzero(keep).tolist()
        r[kind+'_positions_m']=xyz[keep].tolist()
    direction=array(prediction.opening_direction)
    r['opening_directions']=direction[r['opening_selected_indices']].tolist()
    return r


def score_record(record,reference):
    if record['coordinate_frame']!='current_sensor_m':raise ValueError('coordinate frame mismatch')
    if not reference['complete_for_declared_fixture']:
        return dict(status='UNSCORED_INCOMPLETE_REFERENCE',scientific_gate_pass=False)
    anchors=reference['anchors'];openings=reference['openings']
    a={str(radius):match_positions(anchors,record['anchor_positions_m'],radius) for radius in (1.,2.,4.)}
    o=match_positions([v['position_m'] for v in openings],record['opening_positions_m'],1.)
    errors=[]
    for i,j,_ in o['pairs']:
        expected=openings[i].get('direction')
        if expected is None:errors.append(None);continue
        d=np.asarray(record['opening_directions'][j]);truth=np.asarray(expected)
        if not np.isfinite(d).all() or np.linalg.norm(d)==0:raise ValueError('invalid predicted direction')
        errors.append(float(np.degrees(np.arccos(np.clip(d@truth/(np.linalg.norm(d)*np.linalg.norm(truth)),-1,1)))))
    return dict(status='SCORED_COMPLETE_SYNTHETIC_REFERENCE',anchors=a,openings=o,
        opening_direction_errors_deg=errors,scientific_gate_pass=False)
