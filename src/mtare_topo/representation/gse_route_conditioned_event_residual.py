"""Low-capacity route-conditioned correction for causal decision events."""
from __future__ import annotations
from typing import Mapping
import numpy as np
import torch
from torch import nn

ROUTE_FLOW_DIM=4
CAUSAL_ROUTE_FLOW_DIM=16

def route_flow_features(raw_tokens:np.ndarray,history_references:np.ndarray,history_mask:np.ndarray)->np.ndarray:
    """Return current/mean/std/change forward-backward-lateral flow features."""
    token=np.asarray(raw_tokens,dtype=np.float64);references=np.asarray(history_references,dtype=np.int64);mask=np.asarray(history_mask,dtype=bool)
    if token.ndim!=4 or token.shape[1:]!=(3,6,40) or references.shape!=(len(token),5) or mask.shape!=references.shape or not np.all(mask[:,-1]):
        raise ValueError("route-flow feature input contract drift")
    confidence=np.clip(token[...,0],0,1);sine=token[...,1];cosine=token[...,2]
    forward=np.sum(confidence*np.clip(cosine,0,None),axis=2);backward=np.sum(confidence*np.clip(-cosine,0,None),axis=2);lateral=np.sum(confidence*np.abs(sine),axis=2);score=(backward-forward)/np.maximum(forward+backward+lateral,1e-8)
    current=np.stack((forward.mean(1),backward.mean(1),lateral.mean(1),score.mean(1)),axis=1);result=np.empty((len(token),CAUSAL_ROUTE_FLOW_DIM),dtype=np.float32)
    for row in range(len(token)):
        rows=references[row,mask[row]]
        if len(rows)==0 or rows[-1]!=row:raise ValueError("route-flow history must be a causal suffix ending at current row")
        history=current[rows];result[row]=np.concatenate((current[row],history.mean(0),history.std(0),current[row]-history[0]))
    if not np.all(np.isfinite(result)):raise RuntimeError("route-flow features are nonfinite")
    return result

class RouteConditionedEventResidual(nn.Module):
    """Zero-init 16->32->3 residual: structural plus two conditional logits."""
    def __init__(self)->None:
        super().__init__();self.hidden=nn.Linear(CAUSAL_ROUTE_FLOW_DIM,32);self.output=nn.Linear(32,3);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,route_flow:torch.Tensor)->dict[str,torch.Tensor]:
        if route_flow.ndim!=2 or route_flow.shape[1]!=CAUSAL_ROUTE_FLOW_DIM or not bool(torch.isfinite(route_flow).all()):raise ValueError("route residual feature contract drift")
        value=self.output(torch.nn.functional.silu(self.hidden(route_flow)))
        return {"structural_residual":value[:,0],"conditional_residual":value[:,1:]}

def corrected_event_distribution(base:Mapping[str,torch.Tensor],residual:RouteConditionedEventResidual,route_flow:torch.Tensor)->dict[str,torch.Tensor]:
    structural=base["structural_logit"];conditional=base["conditional_decision_logits"];value=residual(route_flow)
    if structural.shape!=(len(route_flow),) or conditional.shape!=(len(route_flow),2):raise ValueError("route residual base contract drift")
    corrected_structural=structural+value["structural_residual"];corrected_conditional=conditional+value["conditional_residual"];mass=torch.sigmoid(corrected_structural);classes=torch.softmax(corrected_conditional,dim=1);probability=torch.cat(((1-mass).unsqueeze(1),mass.unsqueeze(1)*classes),dim=1)
    return {"structural_logit":corrected_structural,"conditional_decision_logits":corrected_conditional,"decision_probability":probability,"structural_residual":value["structural_residual"],"conditional_residual":value["conditional_residual"]}

__all__=["CAUSAL_ROUTE_FLOW_DIM","RouteConditionedEventResidual","corrected_event_distribution","route_flow_features"]
