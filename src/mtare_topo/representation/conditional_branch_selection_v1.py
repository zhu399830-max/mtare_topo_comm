"""Versioned branch correction, retaining original source-bound node loss."""
from dataclasses import replace
import torch
from .conditional_anchor_branch_loss import conditional_anchor_branch_loss
from .branch_selection_objective_v1 import branch_selection_objective


def conditional_branch_selection(prediction, target, **binding):
    if (len(target.directions) != len(target.position_m)
            or len(target.branches_complete) != len(target.position_m)):
        raise ValueError('per-anchor branch targets required')
    # Validate every original target before suppressing the old branch objective.
    for directions, complete in zip(target.directions, target.branches_complete):
        if (type(complete) is not bool or directions.ndim != 2 or directions.shape[1] != 3
                or directions.requires_grad or directions.dtype != prediction.position_m.dtype
                or directions.device != prediction.position_m.device or not torch.isfinite(directions).all()
                or len(directions) > prediction.directions.shape[1]):
            raise ValueError('finite detached bounded branch targets required')
        norms=torch.linalg.vector_norm(directions,dim=-1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=64*torch.finfo(directions.dtype).eps,rtol=0):
            raise ValueError('unit target directions required')
    empty = replace(target, directions=tuple(d.new_empty((0,3)) for d in target.directions),
                    branches_complete=tuple(False for _ in target.directions))
    result = conditional_anchor_branch_loss(prediction, empty, **binding)
    terms = dict(result['terms']); counts = dict(result['counts'])
    branch_results = [branch_selection_objective(prediction.directions[q], prediction.branch_logits[q],
                      target.directions[i], complete=target.branches_complete[i])
                      for i,q in enumerate(result['anchor_assignment'].tolist())]
    active = [r for r in branch_results if r['has_supervision']]
    count = sum(r['positive_count'] for r in branch_results)
    if active:
        terms['branch_presence'] = sum(r['presence'] for r in active)/len(active)
    if count:
        terms['branch_direction'] = sum(r['direction']*r['positive_count'] for r in branch_results)/count
    counts['branch_presence'] = sum(r['positive_count']+r['negative_count'] for r in branch_results)
    counts['branch_direction'] = count
    return dict(result, terms=terms, counts=counts, total=sum(terms.values()),
                branch_assignments=tuple(r['assignment'] for r in branch_results),
                branch_presence_normalization='mean_of_supervised_node_balanced_group_means',
                branch_group_counts=[dict(positive=r['positive_count'],negative=r['negative_count']) for r in branch_results],
                has_supervision=any(counts.values()))
