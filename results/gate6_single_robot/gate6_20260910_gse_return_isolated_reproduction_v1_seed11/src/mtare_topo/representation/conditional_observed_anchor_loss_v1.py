"""Versioned observation-query loss; retains authenticated partial evidence.

Old experiment losses stay unchanged. Teacher objects appear only in this
loss-side API, never in ObservedAnchorDetectorV1.forward.
"""
import numpy as np
import torch
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion
from .anchor_branch_loss import PartialAnchorBranches, anchor_branch_loss
from .observed_anchor_objective_v1 import observed_anchor_objective
from .branch_selection_objective_v1 import branch_selection_objective


def conditional_observed_anchor_loss(prediction,target,*,bundle,grid,produced_targets,frozen_manifest):
    if type(target) is not PartialAnchorBranches:
        raise ValueError('typed partial target required')
    if (type(target.anchors_complete) is not bool or target.anchors_complete
            or len(target.directions)!=len(target.position_m)
            or len(target.branches_complete)!=len(target.position_m)):
        raise ValueError('partial per-anchor branch contract required')
    if set(frozen_manifest)!={'source_binding','target_record_sha256','target_reference_indices'}:
        raise ValueError('closed independent target manifest required')
    record=produced_targets['record']
    if (produced_targets['source_binding']!=frozen_manifest['source_binding']
            or produced_targets['target_record_sha256']!=frozen_manifest['target_record_sha256']
            or canonical_sha(record)!=frozen_manifest['target_record_sha256']
            or record['source_frame_indices']!=frozen_manifest['source_binding']['source']['frame_rows']):
        raise ValueError('frozen source or target content mismatch')
    indices=frozen_manifest['target_reference_indices']
    if (not isinstance(indices,tuple) or len(indices)!=len(set(indices))
            or any(type(i) is not int or not 0<=i<len(record['anchors']) for i in indices)):
        raise ValueError('unique explicit reference indices required')
    expected=torch.as_tensor(np.asarray([record['anchors'][i]['position_m'] for i in indices],dtype=float).reshape(-1,3),
                             dtype=target.position_m.dtype,device=target.position_m.device)
    if not torch.equal(expected,target.position_m):
        raise ValueError('positive positions differ from frozen reference')
    # Reuse prediction validation only: empty unknown targets trigger no legacy
    # matching or old presence/branch objective, and are never used for scoring.
    anchor_branch_loss(prediction,PartialAnchorBranches(prediction.position_m.new_empty((0,3)),(),False,()))
    queries=prediction.position_m.detach().cpu().double().numpy()
    evidence=bound_reference_exclusion(bundle,grid,queries,
        expected_binding=frozen_manifest['source_binding'],matching_radius_m=4.)
    mask=torch.tensor(evidence['reference_negative_mask'],device=prediction.position_m.device,dtype=torch.bool)
    center=observed_anchor_objective(prediction.position_m,prediction.presence_logits,target.position_m,
                                     confirmed_negative=mask)
    branches=[branch_selection_objective(prediction.directions[q],prediction.branch_logits[q],
                  target.directions[i],complete=target.branches_complete[i])
              for i,q in enumerate(center['assignment'].tolist())]
    zero=prediction.position_m.new_zeros((),requires_grad=True)
    active=[r for r in branches if r['has_supervision']]
    positives=sum(r['positive_count'] for r in branches)
    terms=dict(anchor_position=center['position'],anchor_presence=center['presence'],
               branch_presence=sum(r['presence'] for r in active)/len(active) if active else zero,
               branch_direction=sum(r['direction']*r['positive_count'] for r in branches)/positives if positives else zero)
    counts=dict(anchor_position=center['positive_count'],anchor_presence=center['positive_count']+center['negative_count'],
                branch_direction=positives,branch_presence=sum(r['positive_count']+r['negative_count'] for r in branches))
    evidence['used_negative_query_indices']=torch.nonzero(center['negative_mask']).flatten().cpu().tolist()
    evidence['query_xyz_m']=queries.tolist()
    return dict(total=sum(terms.values()),terms=terms,counts=counts,has_supervision=any(counts.values()),
                anchor_assignment=center['assignment'],branch_assignments=tuple(r['assignment'] for r in branches),
                anchor_group_counts=dict(positive=center['positive_count'],negative=center['negative_count'],unknown=center['unknown_count']),
                branch_group_counts=[dict(positive=r['positive_count'],negative=r['negative_count']) for r in branches],
                conditional_anchor_evidence=evidence,scoring_policy_changed=False,
                anchor_presence_normalization='mean_of_valid_positive_and_negative_group_means',
                branch_presence_normalization='mean_of_supervised_node_balanced_group_means')
