"""Conservative instance-duplicate supervision; NOT a smaller background radius.

The historical 4m background exclusion stays intact. A separately observed
candidate can be supervised as a duplicate only in the unique 4m neighborhood
of a confirmed, training-included instance, with no unknown competitor. The
one-to-one positive is always exempt. Unobserved structure coverage alone is
insufficient. This remains a partial-reference training convention, not a
claim that every physical structure has been annotated.
"""
import numpy as np
import torch
from .grouping_center_training_v1 import center_objective as previous_objective
from .observed_anchor_objective_v1 import observed_anchor_objective
from mtare_topo.teacher.gse_reference_query_coverage_v2 import bound_anchor_query_coverage

RADIUS_M = 4.0


def duplicate_candidates(positions, confirmed_positions, target_indices, coverage):
    q = np.asarray(positions, dtype=float).reshape(-1, 3)
    refs = np.asarray(confirmed_positions, dtype=float).reshape(-1, 3)
    indices = list(target_indices)
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(refs) for i in indices):
        raise ValueError('unique valid training reference indices required')
    if not np.isfinite(q).all() or not np.isfinite(refs).all():
        raise ValueError('finite coordinates required')
    observed = np.asarray(coverage['observed_inside_score_region_mask'], dtype=bool)
    conflict = np.asarray(coverage['possible_unconfirmed_reference_mask'], dtype=bool)
    if observed.shape != (len(q),) or conflict.shape != observed.shape:
        raise ValueError('aligned authenticated coverage required')
    near = np.linalg.norm(q[:, None] - refs[None], axis=2) <= RADIUS_M
    # A confirmed but non-training reference ALSO vetoes duplicate supervision.
    return observed & ~conflict & (near.sum(axis=1) == 1) & near[:, indices].any(axis=1)


def center_objective(output, observation):
    old = previous_objective(output, observation)  # includes all original binding checks
    p = output.prediction
    loss = observation.loss_only
    frozen = loss['frozen_manifest']
    xyz = p.position_m.detach().cpu().double().numpy()
    coverage = bound_anchor_query_coverage(loss['bundle'], loss['grid'], xyz,
        produced_targets=loss['produced_targets'],
        manifest_row={k: frozen[k] for k in ('source_binding', 'target_record_sha256')},
        matching_radius_m=RADIUS_M)
    confirmed = [a['position_m'] for a in loss['produced_targets']['record']['anchors']]
    duplicate = duplicate_candidates(xyz, confirmed, frozen['target_reference_indices'], coverage)
    background = np.asarray(old['evidence']['reference_negative_mask'], dtype=bool)
    result = observed_anchor_objective(p.position_m, p.presence_logits,
        loss['target'].position_m.to(p.position_m),
        confirmed_negative=torch.as_tensor(background | duplicate, device=p.position_m.device))
    if not torch.equal(result['assignment'], old['assignment']):
        raise AssertionError('supervision correction must not change assignment')
    used_duplicate = duplicate.copy()
    used_duplicate[result['assignment'].cpu().numpy()] = False
    result['evidence'] = dict(original_background=old['evidence'], coverage=coverage,
        duplicate_candidate_mask=duplicate.tolist(), used_duplicate_mask=used_duplicate.tolist(),
        background_radius_m=RADIUS_M, duplicate_instance_radius_m=RADIUS_M,
        physical_background_claim=False)
    return result
