import unittest
from dataclasses import replace

import torch
from scipy.optimize import linear_sum_assignment

from mtare_topo.representation.gse_region_target_bridge import (
    align_axis_directions, partial_region_targets)
from mtare_topo.representation.gse_region_queries import (
    RegionQueryHead, tokens_from_axes, region_set_losses)


def axes(count=3, batch=1):
    target = torch.zeros((batch, 32, 3, 3), dtype=torch.float64)
    pred = torch.empty_like(target)
    for slot in range(32):
        pred[:, slot] = torch.tensor([[1000 + 10*slot, 1., 0.],
                                     [1001 + 10*slot, 1., 0.],
                                     [1002 + 10*slot, 1., 0.]])
    for slot in range(count):
        target[:, slot] = torch.tensor([[10.*slot, 0., 0.],
                                       [10.*slot + 1, .5, 0.],
                                       [10.*slot + 2, 1., .5]])
    pred[:, :count] = target[:, :count]
    mask = torch.zeros((batch, 32), dtype=torch.bool)
    mask[:, :count] = True
    return pred, target, mask


def region(event="corridor"):
    labels, valid = [0]*64, [False]*64
    labels[0], valid[0], valid[2] = 1, True, True
    return dict(center_current_sensor_m=[1., 2., 3.], center_valid=True,
                event_target=event, event_valid=event is not None,
                directional_member_target=labels, directional_member_valid=valid)


def observation(*regions):
    return dict(regions=list(regions), observation_label_complete=False)


class AxisDirectionTests(unittest.TestCase):
    def test_exact_three(self):
        result = align_axis_directions(*axes())
        self.assertEqual(result.teacher_direction[0, :6].tolist(), list(range(6)))
        self.assertEqual(int(result.valid.sum()), 6)
        self.assertEqual(result.ledger[0]["unselected_prediction_slots"], 29)

    def test_prediction_permutation_and_reversal(self):
        p, t, m = axes()
        permutation = torch.arange(32).roll(7)
        p = p[:, permutation].flip(2)
        result = align_axis_directions(p, t, m)
        for slot in range(3):
            q = int(torch.where(permutation == slot)[0])
            self.assertEqual(result.teacher_direction[0, 2*q:2*q+2].tolist(), [2*slot+1, 2*slot])

    def test_teacher_permutation(self):
        p, t, m = axes()
        perm = torch.arange(32).roll(9)
        result = align_axis_directions(p, t[:, perm], m[:, perm])
        for slot in range(3):
            new = int(torch.where(perm == slot)[0])
            self.assertEqual(result.teacher_direction[0, 2*slot:2*slot+2].tolist(), [2*new, 2*new+1])

    def test_common_rotation_translation_exact_pairs(self):
        p, t, m = axes()
        q, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [3., 1., 4.], [2., 5., 1.]], dtype=p.dtype))
        if torch.linalg.det(q) < 0: q[:, 0] *= -1
        shift = torch.tensor([3., -2., 8.], dtype=p.dtype)
        first = align_axis_directions(p, t, m)
        second = align_axis_directions(p @ q.T + shift, t @ q.T + shift, m)
        self.assertTrue(torch.equal(first.teacher_direction, second.teacher_direction))

    def test_noisy_common_rotation_preserves_map_where_l1_changes(self):
        p, t, m = axes(2)
        base = torch.tensor([[0., 0., -100.], [0., 0., 0.], [0., 0., 100.]], dtype=p.dtype)
        t[0, 0], t[0, 1] = base, base + torch.tensor([2., 0., 0.])
        offsets = torch.tensor([
            [.9116986000078935, -.10055071721202144, -.9435829659645868],
            [.9366343005536819, -.21030235274694212, -3.062547828416398]], dtype=p.dtype)
        p[0, :2] = base[None] + offsets[:, None]
        rotation = torch.tensor([
            [-.24938899717279447, -.9684026892442189, .0011659990190707843],
            [.004354548251509041, .00008262217717199085, .9999905154965727],
            [-.9683936007629711, .24939170924098614, .004196351175295877]], dtype=p.dtype)
        self.assertTrue(torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=p.dtype)))
        self.assertAlmostEqual(float(torch.linalg.det(rotation)), 1.)
        rp, rt = p @ rotation.T, t @ rotation.T
        def old_l1_columns(pred, teacher):
            direct = (pred[0, :2, None] - teacher[0, None, :2]).abs().mean((-1, -2))
            reverse = (pred[0, :2, None] - teacher[0, None, :2].flip(-2)).abs().mean((-1, -2))
            cost = torch.minimum(direct, reverse)
            # Strict alternate objectives: this is not solver tie ordering.
            self.assertGreater(float((cost[0, 0] + cost[1, 1] - cost[0, 1] - cost[1, 0]).abs()), .01)
            return linear_sum_assignment(cost.numpy())[1].tolist()
        self.assertNotEqual(old_l1_columns(p, t), old_l1_columns(rp, rt))
        first, second = align_axis_directions(p, t, m), align_axis_directions(rp, rt, m)
        self.assertTrue(first.valid[0, :4].all())
        self.assertTrue(torch.equal(first.teacher_direction, second.teacher_direction))
        self.assertEqual(first.ledger, second.ledger)

    def test_overflowing_norm_rejected_not_unknown(self):
        p, t, m = axes(1)
        p[0, 31] = 1e200
        with self.assertRaisesRegex(ValueError, "norm rounding scale"):
            align_axis_directions(p, t, m)

    def test_duplicate_predictions_unknown_not_slot_tiebreak(self):
        p, t, m = axes(1)
        p[:, 9] = p[:, 0]
        result = align_axis_directions(p, t, m)
        self.assertFalse(result.valid.any())
        self.assertEqual(result.ledger[0]["ambiguous_assignment_pairs"], 1)

    def test_duplicate_teacher_unknown(self):
        p, t, m = axes(2)
        t[:, 1] = t[:, 0]
        p[:, 1] = p[:, 0]
        result = align_axis_directions(p, t, m)
        self.assertEqual(result.ledger[0]["ambiguous_assignment_pairs"], 2)
        self.assertFalse(result.valid.any())

    def test_global_cycle_tie_despite_different_local_costs(self):
        p, t, m = axes(2)
        # Costs [[1,3],[2,4]] after x shifts: both full assignments cost 5.
        t[:, 1] = t[:, 0] + torch.tensor([2., 0., 0.])
        p[:, 0] = t[:, 0] - torch.tensor([1., 0., 0.])
        p[:, 1] = t[:, 0] - torch.tensor([2., 0., 0.])
        result = align_axis_directions(p, t, m)
        self.assertEqual(result.ledger[0]["ambiguous_assignment_pairs"], 2)

    def test_ambiguity_does_not_remove_independent_unique_pair(self):
        p, t, m = axes(3)
        p[:, 8] = p[:, 0]
        result = align_axis_directions(p, t, m)
        self.assertFalse(result.valid[0, :2].any())
        self.assertTrue(result.valid[0, 2:6].all())

    def test_reverse_cost_tie_unknown(self):
        p, t, m = axes(1)
        p[:, 0] = t[:, 0, 1:2].expand(-1, 3, -1)
        result = align_axis_directions(p, t, m)
        self.assertFalse(result.valid.any())
        self.assertEqual(result.ledger[0]["ambiguous_orientation_pairs"], 1)

    def test_one_ulp_alternative_is_unknown(self):
        p, t, m = axes(1)
        p[:, 9] = p[:, 0]
        p[0, 9, 2, 0] = torch.nextafter(p[0, 9, 2, 0], torch.tensor(float("inf")))
        result = align_axis_directions(p, t, m)
        self.assertFalse(result.valid.any())

    def test_one_ulp_orientation_is_unknown(self):
        p, t, m = axes(1)
        p[:, 0] = t[:, 0, 1:2].expand(-1, 3, -1)
        p[0, 0, 0, 0] = torch.nextafter(p[0, 0, 0, 0], torch.tensor(float("inf")))
        result = align_axis_directions(p, t, m)
        self.assertEqual(result.ledger[0]["ambiguous_orientation_pairs"], 1)
        self.assertFalse(result.valid.any())

    def test_far_unique_pair_is_not_rejected_by_invented_threshold(self):
        p, t, m = axes(1)
        p[:, 0] += torch.tensor([0., 30., 0.])
        result = align_axis_directions(p, t, m)
        self.assertTrue(result.valid[0, :2].all())
        self.assertEqual(result.reasons[0][0], "UNIQUE_GEOMETRY_MATCH_NOT_CONFIDENCE")

    def test_ambiguous_permutation_preserves_only_unique_map(self):
        p, t, m = axes(3)
        p[:, 9] = p[:, 0]
        permutation = torch.arange(32).roll(5)
        first = align_axis_directions(p, t, m)
        second = align_axis_directions(p[:, permutation], t, m)
        permuted_tokens = (2*permutation[:, None] + torch.arange(2)).flatten()
        self.assertTrue(torch.equal(first.teacher_direction[:, permuted_tokens], second.teacher_direction))

    def test_no_targets_including_inactive_nan(self):
        p, t, m = axes(0)
        t[:] = float("nan")
        result = align_axis_directions(p, t, m)
        self.assertFalse(result.valid.any())
        self.assertEqual(result.ledger[0]["selected_pairs"], 0)

    def test_float32_and_detached(self):
        p, t, m = axes()
        result = align_axis_directions(p.float().requires_grad_(), t.float().requires_grad_(), m)
        self.assertFalse(result.teacher_direction.requires_grad)
        self.assertEqual(int(result.valid.sum()), 6)

    def test_all_32_targets(self):
        result = align_axis_directions(*axes(32))
        self.assertTrue(result.valid.all())
        self.assertEqual(result.teacher_direction[0].tolist(), list(range(64)))

    def test_repeat_deterministic(self):
        data = axes()
        first, second = align_axis_directions(*data), align_axis_directions(*data)
        self.assertTrue(torch.equal(first.teacher_direction, second.teacher_direction))
        self.assertEqual(first.ledger, second.ledger)

    def test_bad_axes(self):
        for failure in ("slots", "active_nan", "pred_nan", "dtype", "mask"):
            with self.subTest(failure=failure):
                p, t, m = axes()
                if failure == "slots": p = p[:, :31]
                if failure == "active_nan": t[0, 0, 0, 0] = float("nan")
                if failure == "pred_nan": p[0, 31, 0, 0] = float("nan")
                if failure == "dtype": t = t.float()
                if failure == "mask": m = m.int()
                with self.assertRaises(ValueError): align_axis_directions(p, t, m)


class PartialBridgeTests(unittest.TestCase):
    def test_partial_unknown_and_padding(self):
        aligned = align_axis_directions(*axes(batch=2))
        result = partial_region_targets([observation(region(), region(None)), observation()], aligned, dtype=torch.float64)
        self.assertEqual(result.targets.centers_m.shape, (2, 2, 3))
        self.assertEqual(result.targets.events.tolist(), [[0, -1], [-1, -1]])
        self.assertFalse(result.targets.label_complete.any())
        self.assertFalse(result.targets.center_valid[1].any())
        self.assertFalse(result.targets.member_valid[1].any())
        self.assertEqual(result.ledger[0]["transferred_member_positive"], 2)
        self.assertEqual(result.ledger[0]["transferred_member_negative"], 2)

    def test_reversal_transfers_direction_not_identity(self):
        p, t, m = axes()
        p[:, 0] = p[:, 0].flip(1)
        r = region()
        r["construction_node_id_teacher_only"] = "never-used-for-matching"
        result = partial_region_targets([observation(r)], align_axis_directions(p, t, m), dtype=p.dtype)
        self.assertEqual(float(result.targets.members[0, 0, 1]), 1.)
        self.assertTrue(result.targets.member_valid[0, 0, 1])
        self.assertFalse(result.targets.member_valid[0, 0, 0])

    def test_ambiguous_positive_unknown_but_center_retained(self):
        p, t, m = axes()
        p[:, 9] = p[:, 0]
        result = partial_region_targets([observation(region())], align_axis_directions(p, t, m), dtype=p.dtype)
        self.assertTrue(result.targets.center_valid[0, 0])
        self.assertEqual(result.ledger[0]["original_member_positive"], 1)
        self.assertEqual(result.ledger[0]["unknown_correspondence_member_positive"], 1)
        self.assertEqual(result.ledger[0]["transferred_member_positive"], 0)
        self.assertEqual(result.ledger[0]["transferred_member_negative"], 1)

    def test_no_regions(self):
        p, t, m = axes(0)
        result = partial_region_targets([observation()], align_axis_directions(p, t, m), dtype=p.dtype)
        self.assertEqual(result.targets.members.shape, (1, 0, 64))
        prediction = RegionQueryHead().double()(tokens_from_axes(p))
        losses = region_set_losses(prediction, result.targets)
        self.assertFalse(losses["has_supervision"])
        self.assertEqual(float(losses["total"].detach()), 0.)

    def test_bad_teacher(self):
        for failure in ("complete", "unknown_event", "terminal", "mask", "labels", "inactive", "center"):
            with self.subTest(failure=failure):
                p, t, m = axes()
                r, obs = region(), None
                if failure == "unknown_event": r["event_valid"] = False
                if failure == "terminal": r["event_target"] = "terminal"
                if failure == "mask": r["directional_member_valid"][0] = 1
                if failure == "labels": r["directional_member_target"][0] = .5
                if failure == "inactive": r["directional_member_valid"][63] = True
                if failure == "center": r["center_current_sensor_m"] = [float("nan")]*3
                obs = observation(r)
                if failure == "complete": obs["observation_label_complete"] = True
                with self.assertRaises(ValueError):
                    partial_region_targets([obs], align_axis_directions(p, t, m), dtype=p.dtype)

    def test_corrupted_mapping_rejected(self):
        aligned = align_axis_directions(*axes())
        mapping = aligned.teacher_direction.clone()
        mapping[0, 1] = mapping[0, 0]
        with self.assertRaises(ValueError):
            partial_region_targets([observation(region())], replace(aligned, teacher_direction=mapping), dtype=torch.float64)

    def test_actual_head_loss_backward_no_teacher_in_forward(self):
        p, t, m = axes()
        head = RegionQueryHead().double()
        prediction = head(tokens_from_axes(p))  # all 64; target not supplied
        self.assertEqual(prediction.query_supported.shape, (1, 64))
        self.assertTrue(prediction.query_supported.all())
        target = partial_region_targets([observation(region())], align_axis_directions(p, t, m), dtype=p.dtype)
        losses = region_set_losses(prediction, target.targets)
        losses["total"].backward()
        self.assertEqual(losses["counts"]["presence_negative"], 0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(any(v.grad is not None and bool(v.grad.abs().sum()) for v in head.parameters()))


if __name__ == "__main__":
    unittest.main()
