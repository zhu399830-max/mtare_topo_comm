"""Adapt an existing independent reference into loss-only located targets."""
import numpy as np
import torch
from mtare_topo.representation.gse_block_structure_loss import LocatedStructureTargets


def located_targets(reference,*,device='cpu'):
    complete=reference['complete_for_declared_fixture']
    if type(complete)!=bool:raise ValueError('explicit reference completeness required')
    anchors=reference['anchors'];openings=reference['openings']
    if anchors is None or openings is None:
        if complete or anchors is not None or openings is not None:raise ValueError('inconsistent unresolved reference')
        anchors=[];openings=[]
    if len(anchors)>32 or len(openings)>64:raise OverflowError('reference exceeds query capacity')
    a=np.asarray(anchors,dtype=np.float32).reshape(-1,3)
    o=np.asarray([v['position_m'] for v in openings],dtype=np.float32).reshape(-1,3)
    d=np.full((len(openings),3),np.nan,np.float32);known=np.zeros(len(openings),bool)
    for i,v in enumerate(openings):
        if v.get('direction') is not None:
            d[i]=v['direction'];known[i]=True
    if not np.isfinite(a).all() or not np.isfinite(o).all() or not np.isfinite(d[known]).all():
        raise ValueError('nonfinite known geometry')
    def tensor(x):return torch.from_numpy(x.copy()).to(device)
    return LocatedStructureTargets(tensor(a),tensor(o),tensor(d),tensor(known),complete,complete)
