"""Explicit numerical projection backend; no tolerance and no device fallback."""
import torch
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


def project_input(range_valid,translation_m,yaw_deg,*,device):
    if device not in ('cpu','cuda:0'):
        raise ValueError('explicit cpu or cuda:0 projection contract required')
    if device=='cuda:0' and not torch.cuda.is_available():
        raise ValueError('frozen CUDA projection device unavailable')
    with torch.no_grad():
        points,valid=register_causal_lidar_points(*[
            torch.from_numpy(a.copy()[None]).to(device) for a in (range_valid,translation_m,yaw_deg)])
    return points.detach().cpu().reshape(1,-1,3),valid.detach().cpu().reshape(1,-1)
