"""Observation-ray-seeded structural readout over common three-path memory.

No retired detector/scoring module is instantiated or restored. Reuses only
the previously tested geometric ball projection and deterministic FPS helper.
Outputs are partial-reference hypotheses, not calibrated detections.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .gse_common_structure_memory import CommonStructureMemory
from .gse_observed_membership_v1 import _seeds
from .observed_anchor_detector_v1 import residual_positions
from .gse_structure_prediction_contract import PartialStructuralPrediction,validate_prediction


@dataclass(frozen=True)
class StructuralReadout:
    prediction: PartialStructuralPrediction
    structure_query_ray_indices: np.ndarray
    section_query_ray_indices: np.ndarray


def section_positions(seed_direction,raw_offset):
    vector=seed_direction+2.*torch.tanh(raw_offset)
    precise=vector.double();norm=torch.linalg.vector_norm(precise,dim=-1,keepdim=True)
    if bool((norm<=torch.finfo(vector.dtype).eps).any()):raise ValueError('undefined section direction')
    return (10.*precise/norm).to(vector.dtype)


class CommonStructureReadout(nn.Module):
    def __init__(self,variant):
        super().__init__()
        self.memory=CommonStructureMemory(variant)
        self.query_adapter=nn.Sequential(nn.Linear(262,128),nn.GELU(),nn.LayerNorm(128))
        self.role=nn.Parameter(torch.zeros(2,128))
        layer=nn.TransformerDecoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(128))
        self.structure_offset=nn.Linear(128,3);self.section_offset=nn.Linear(128,3)
        self.structure_relation=nn.Linear(128,128);self.section_relation=nn.Linear(128,128)
        for head in (self.structure_offset,self.section_offset):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,observation,*,patch_batch=None):
        rays=observation.local_rays
        if not len(rays.ray_indices):return None
        memory=self.memory(observation,patch_batch=patch_batch)
        if not bool(memory.valid.any()):raise ValueError('ray observation without valid memory')
        midpoint=(rays.start_xyz_m+rays.end_xyz_m)/2.
        selected=_seeds(midpoint)  # at most64, no GT or scores, no padding
        device=memory.features.device;dtype=memory.features.dtype
        p=torch.tensor(midpoint[selected],device=device,dtype=dtype)
        end=rays.end_xyz_m[selected];norm=np.linalg.norm(end,axis=1)
        direction=end.copy()
        nonzero=norm>0
        direction[nonzero]/=norm[nonzero,None]
        # A measured return at current origin has no radial direction. Its
        # observed ray direction remains available; this is a query seed only.
        delta=rays.end_xyz_m[selected]-rays.start_xyz_m[selected]
        direction[~nonzero]=delta[~nonzero]/np.linalg.norm(delta[~nonzero],axis=1)[:,None]
        d=torch.tensor(direction,device=device,dtype=dtype)
        token=torch.tensor(observation.ray_sensor_token_indices[selected],device=device,dtype=torch.long)
        q=self.query_adapter(torch.cat((memory.features[0,token],memory.features[0,900+token],p/10.,d),-1))
        na=min(32,len(q))
        decoded=self.decoder(torch.cat((q[:na]+self.role[0],q+self.role[1]))[None],memory.features,
                             memory_key_padding_mask=~memory.valid)[0]
        a,o=decoded[:na],decoded[na:]
        positions,_=residual_positions(p[:na],self.structure_offset(a))
        sections=section_positions(d,self.section_offset(o))
        logits=self.section_relation(o)@self.structure_relation(a).T/(128.**.5)
        pred=PartialStructuralPrediction(positions,sections,logits);validate_prediction(pred)
        ids=rays.ray_indices[selected].copy();ids.setflags(write=False)
        return StructuralReadout(pred,ids[:na],ids)
