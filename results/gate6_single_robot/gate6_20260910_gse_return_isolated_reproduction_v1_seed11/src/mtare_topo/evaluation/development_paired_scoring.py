"""Bind selected predictions to existing partial-reference coverage and scoring."""
import numpy as np
import torch
from .anchor_branch_scoring import score_anchor_branches
from mtare_topo.teacher.gse_reference_query_coverage_v2 import bound_anchor_query_coverage


def score_observation(prediction, observation, *, matching_radius_m, matching_angle_deg,
                      anchor_probability=.5, branch_probability=.5):
    if not 0<anchor_probability<1 or not 0<branch_probability<1:raise ValueError('explicit probability thresholds required')
    for value in (prediction.position_m,prediction.presence_logits,prediction.directions,prediction.branch_logits):
        if not torch.isfinite(value).all():raise FloatingPointError('nonfinite prediction before selection')
    loss=observation.loss_only;target=loss['target']
    keep=torch.nonzero(torch.sigmoid(prediction.presence_logits.detach())>=anchor_probability).flatten()
    positions=prediction.position_m[keep].detach().cpu().double().numpy()
    directions=[]
    for i in keep.tolist():
        mask=torch.sigmoid(prediction.branch_logits[i].detach())>=branch_probability
        directions.append(prediction.directions[i,mask].detach().cpu().double().numpy())
    manifest={k:loss['frozen_manifest'][k] for k in ('source_binding','target_record_sha256')}
    coverage=bound_anchor_query_coverage(loss['bundle'],loss['grid'],positions,
        produced_targets=loss['produced_targets'],manifest_row=manifest,matching_radius_m=matching_radius_m)
    scoreable=np.asarray(coverage['query_scoreable_mask'],dtype=bool)
    allowed=~np.asarray(coverage['possible_unconfirmed_reference_mask'],dtype=bool)
    result=score_anchor_branches(positions,directions,target.position_m.cpu().double().numpy(),
        tuple(d.cpu().double().numpy() for d in target.directions),anchor_scoreable=scoreable,
        association_allowed=allowed,anchor_reference_complete=False,
        branch_reference_complete=target.branches_complete,matching_radius_m=matching_radius_m,
        matching_angle_deg=matching_angle_deg)
    result.update(provenance_verified=True,source=observation.source,split=observation.split,
        selected_anchor_query_indices=keep.cpu().tolist(),coverage=coverage,
        matching_radius_m=matching_radius_m,matching_angle_deg=matching_angle_deg,
        physical_annotation_complete=False)
    return result
