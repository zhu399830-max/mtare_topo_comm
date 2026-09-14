"""Synthetic loss payload projection, not real teacher-adapter qualification."""
import copy
from dataclasses import replace

import pytest
import torch

from test_conditional_anchor_branch_loss import fixture
from mtare_topo.representation.conditional_branch_selection_v1 import conditional_branch_selection

REQUIRED = ('record', 'source_binding', 'target_record_sha256')


def evaluate_payload(*, compact, branches_complete):
    p, target, binding = fixture()
    target = replace(target, directions=(torch.tensor([[1., 0., 0.]], dtype=torch.float64),),
                     branches_complete=(branches_complete,))
    full = copy.deepcopy(binding['produced_targets'])
    full.update(teacher_provenance={'diagnostic_only': [0] * 1000},
                unknown_candidates={'anchors': [{'diagnostic_only': True}]},
                full_training_gate_eligible=False)
    binding['produced_targets'] = {k: full[k] for k in REQUIRED} if compact else full
    result = conditional_branch_selection(p, target, **binding)
    result['total'].backward()
    gradients = tuple(None if x.grad is None else x.grad.clone()
                      for x in (p.position_m, p.presence_logits, p.directions, p.branch_logits))
    return result, gradients


@pytest.mark.parametrize('complete', [False, True])
def test_exact_loss_assignment_evidence_and_gradient_equivalence(complete):
    full, fg = evaluate_payload(compact=False, branches_complete=complete)
    compact, cg = evaluate_payload(compact=True, branches_complete=complete)
    assert torch.equal(full['total'], compact['total'])
    assert full['counts'] == compact['counts']
    assert full['conditional_anchor_evidence'] == compact['conditional_anchor_evidence']
    assert torch.equal(full['anchor_assignment'], compact['anchor_assignment'])
    for key in full['terms']:
        assert torch.equal(full['terms'][key], compact['terms'][key])
    for a, b in zip(full['branch_assignments'], compact['branch_assignments'], strict=True):
        assert torch.equal(a, b)
    for a, b in zip(fg, cg, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
    assert cg[1][2] == 0  # Unknown query stays unsupervised.


@pytest.mark.parametrize('damage', ['hash', 'record', 'source'])
def test_projected_payload_keeps_original_authentication(damage):
    p, target, binding = fixture()
    binding['produced_targets'] = copy.deepcopy({k: binding['produced_targets'][k] for k in REQUIRED})
    payload = binding['produced_targets']
    if damage == 'hash':
        payload['target_record_sha256'] = 'invalid'
    elif damage == 'record':
        payload['record']['anchors'][0]['position_m'][0] = 1.
    else:
        payload['source_binding']['source']['frame_rows'][-1] += 1
    with pytest.raises(ValueError):
        conditional_branch_selection(p, target, **binding)
