"""Convert authenticated saved arrays; source authentication belongs to caller."""
import numpy as np
from .anchor_branch_observation_v1 import (
    AnchorBranchObservationV1, AnchorBranchCandidateV1, BranchDirectionCandidateV1)


def convert_saved_prediction(arrays, *, stream_key, decision_index, frame_orders, input_binding_sha256):
    shapes=dict(position_m=(32,3),presence_logits=(32,),directions=(32,64,3),branch_logits=(32,64))
    if set(arrays)!=set(shapes):raise ValueError('exact saved prediction fields required')
    for key,shape in shapes.items():
        a=arrays[key]
        if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all():
            raise ValueError('finite saved prediction arrays required')
    if np.any(np.abs(np.linalg.norm(arrays['directions'].astype(float),axis=-1)-1)>1e-5):
        raise ValueError('invalid direction; do not normalize saved output')
    if np.any(np.linalg.norm(arrays['position_m'].astype(float),axis=-1)>10+1e-5):
        raise ValueError('anchor outside trained domain')
    def confidence(logit):
        x=float(logit)
        return float(1/(1+np.exp(-x))) if x>=0 else float(np.exp(x)/(1+np.exp(x)))
    anchors=[]
    # Compare logits directly: this is the frozen >=0.5 sigmoid decision,
    # without introducing floating-point rounding of scores into selection.
    for i in np.flatnonzero(arrays['presence_logits']>=0):
        branches=tuple(BranchDirectionCandidateV1(int(j),tuple(map(float,arrays['directions'][i,j])),
            confidence(arrays['branch_logits'][i,j])) for j in np.flatnonzero(arrays['branch_logits'][i]>=0))
        anchors.append(AnchorBranchCandidateV1(int(i),tuple(map(float,arrays['position_m'][i])),
            confidence(arrays['presence_logits'][i]),branches))
    return AnchorBranchObservationV1(stream_key,decision_index,tuple(frame_orders),input_binding_sha256,tuple(anchors))
