from dataclasses import replace
import pytest
import torch
from mtare_topo.teacher.gse_channel_relation_binding import (
    PatchChannelMembership as Membership, SupportedRelationValue as Value,
    SupportedChannelPair as Pair, bind_channel_relation_targets as bind)
from mtare_topo.representation.gse_observable_relation_losses import observable_relation_losses
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
from test_structural_representation import inputs, context


def fixture():
    x, p = inputs()
    r = GeometryStructureEncoder('C')(x, p, context()).relations
    memberships = ((Membership(('a',), 'patch-a', 'observed_channel_support'),
                    Membership(('b',), 'patch-b', 'observed_channel_support'),
                    Membership(())),)
    pairs = (Pair(0, 'a', 'b', axis=Value(.5, 'axis-support'), height=Value(2., 'layer-support')),
             Pair(0, 'b', 'a', axis=Value(.5, 'axis-support'), height=Value(-2., 'layer-support')))
    return r, memberships, pairs


def test_partial_attributes_and_loss_backward():
    r, m, p = fixture(); t, audit = bind(r, m, p)
    assert t.axis_known.sum() == 2 and t.height_known.sum() == 2
    assert not t.section_known.any() and not t.correspondence_known.any()
    assert t.height_difference_m[0, 0, 0] == 2 and t.height_difference_m[0, 1, 0] == -2
    assert audit[0]['evidence_refs'] == ('axis-support', 'layer-support', 'patch-a', 'patch-b')
    losses, counts = observable_relation_losses(r, t)
    sum(losses.values()).backward()
    assert counts['correspondence'] == 0


@pytest.mark.parametrize('replacement', [Membership(()), Membership(('a', 'c'), 'ambiguous', 'observed_channel_support'), Membership(('a',))])
def test_missing_ambiguous_unqualified_stay_unknown(replacement):
    r, m, p = fixture(); m = ((replacement, *m[0][1:]),)
    t, audit = bind(r, m, p)
    assert not t.axis_known.any() and not t.height_known.any()
    assert t.evidence_refs == ()


def test_sources_cannot_be_used_as_channel_membership():
    r, m, p = fixture()
    m = ((replace(m[0][0], evidence_kind='construction_operand'), *m[0][1:]),)
    with pytest.raises(ValueError, match='not channel support'): bind(r, m, p)


def test_same_region_does_not_make_correspondence_positive():
    r, m, p = fixture(); m = ((m[0][0], m[0][0], m[0][2]),)
    t, audit = bind(r, m, ())
    assert not t.correspondence_known.any() and not t.axis_known.any()
    assert audit[0]['reason'] == 'NO_ORDERED_RELATION_EVIDENCE'


def test_no_reverse_relation_invention():
    r, m, p = fixture(); t, _ = bind(r, m, p[:1])
    assert t.height_known[0, 0, 0] and not t.height_known[0, 1, 0]


def test_prediction_values_do_not_select_labels():
    r, m, p = fixture(); t, audit = bind(r, m, p)
    other = replace(r, axis_abs_dot=torch.rand_like(r.axis_abs_dot), correspondence_logits=torch.randn_like(r.correspondence_logits)*100)
    u, other_audit = bind(other, m, p)
    assert torch.equal(t.axis_abs_dot, u.axis_abs_dot) and torch.equal(t.axis_known, u.axis_known)
    assert audit == other_audit


def test_membership_patch_permutation_tracks_neighbor_indices():
    r, m, p = fixture(); t, _ = bind(r, m, p)
    order = torch.tensor([2, 0, 1]); inv = torch.argsort(order)
    old = r.neighbor_index[:, order]
    other = replace(r, neighbor_index=torch.where(old >= 0, inv[old.clamp_min(0)], -1),
                    computation_valid=r.computation_valid[:, order], axis_abs_dot=r.axis_abs_dot[:, order])
    u, _ = bind(other, (tuple(m[0][i] for i in order.tolist()),), p)
    assert torch.equal(t.height_difference_m[:, order], u.height_difference_m)
    assert torch.equal(t.height_known[:, order], u.height_known)


def test_no_duplicate_or_evidenceless_relation():
    r, m, p = fixture()
    with pytest.raises(ValueError, match='duplicate'): bind(r, m, p+p[:1])
    with pytest.raises(ValueError, match='independent evidence'):
        bind(r, m, (replace(p[0], axis=Value(.5, '')),))


def test_known_relation_on_absent_neighbor_rejected():
    r, m, p = fixture()
    idx = r.neighbor_index.clone(); idx[0, 0, 0] = 100
    with pytest.raises(ValueError, match='out-of-range'): bind(replace(r, neighbor_index=idx), m, p)
