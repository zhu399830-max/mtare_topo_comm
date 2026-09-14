"""Partial-reference fit objective: teacher used only after forward.

No confidence-dependent correspondence and no unmatched/background penalty.
Matched presence is positive-only, explicitly not calibrated detection.
Unknown relations contribute neither labels nor gradients. Full detection and
unknown output accounting must be performed separately by the runner.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def observed_membership_loss(prediction, record):
    if prediction is None:
        raise ValueError('no observation queries; cannot fit supported targets')
    device=prediction.anchor_position_m.device;dtype=prediction.anchor_position_m.dtype
    terms={};denominators={};matches={}
    for role in ('anchor','opening'):
        rows=record[role+'s'];positions=getattr(prediction,role+'_position_m')
        if len(rows)>len(positions):
            raise ValueError('reference exceeds actual observed query capacity; no truncation')
        truth=torch.tensor([r['position_m'] for r in rows],device=device,dtype=dtype).reshape(-1,3)
        if not torch.isfinite(truth).all():raise ValueError('nonfinite known reference')
        if len(rows):
            q,t=linear_sum_assignment(torch.cdist(positions.detach(),truth).cpu().numpy())
            mapping=np.full(len(rows),-1,np.int64);mapping[t]=q
            if (mapping<0).any():raise ValueError('incomplete unique assignment')
            indices=torch.tensor(mapping,device=device)
            terms[role+'_position']=F.smooth_l1_loss(positions[indices]/10.,truth/10.)
            logits=getattr(prediction,role+'_presence_logits')[indices]
            terms[role+'_matched_presence']=F.softplus(-logits).mean()
            denominators[role+'_position']=len(rows)
            denominators[role+'_matched_presence']=len(rows)
        else: mapping=np.empty(0,np.int64)
        matches[role]=mapping.tolist()
    membership=record['membership']
    if len(membership)!=len(record['openings']) or any(len(r)!=len(record['anchors']) for r in membership):
        raise ValueError('membership dimensions inconsistent')
    values=[];labels=[];unknown=0
    for opening,row in enumerate(membership):
        for anchor,label in enumerate(row):
            if label is None:unknown+=1;continue
            if type(label) is not bool:raise ValueError('explicit true/false/unknown required')
            values.append(prediction.membership_logits[matches['opening'][opening],matches['anchor'][anchor]])
            labels.append(float(label))
    if values:
        terms['membership']=F.binary_cross_entropy_with_logits(torch.stack(values),
            torch.tensor(labels,device=device,dtype=dtype))
        denominators['membership']=len(values)
    if not terms:raise ValueError('no supported supervision')
    total=sum(terms.values())
    if not torch.isfinite(total):raise ValueError('nonfinite loss')
    return dict(total=total,terms=terms,denominators=denominators,assignments=matches,
        known_positive=sum(labels),known_negative=len(labels)-sum(labels),unknown_relations=unknown,
        unmatched_background_penalized=False,presence_calibrated=False)
