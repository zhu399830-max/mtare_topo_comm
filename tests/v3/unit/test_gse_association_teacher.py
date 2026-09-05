from __future__ import annotations

import unittest

from mtare_topo.semantics.geometric_semantics import StructuralEvent
from mtare_topo.teacher.gse_association_teacher import (
    AssociationTeacherExample,
    EdgeEventSample,
    cluster_edge_event_identities,
    deterministic_association_pairs,
    node_event_identity,
)


class GSEAssociationTeacherTest(unittest.TestCase):
    def test_opposite_directions_share_one_physical_event_identity(self) -> None:
        samples = [
            EdgeEventSample("edge:d0", 9, 9.0),
            EdgeEventSample("edge:d0", 10, 10.0),
            EdgeEventSample("edge:d1", 20, 9.4),
            EdgeEventSample("edge:d1", 19, 10.4),
            EdgeEventSample("edge:d0", 25, 25.0),
            EdgeEventSample("edge:d1", 5, 25.4),
        ]
        mapping, identities = cluster_edge_event_identities(
            parent_id="S01_flat_tree_small_C01",
            edge_id="e0",
            event=StructuralEvent.GEOMETRY_TRANSITION,
            samples=samples,
        )
        self.assertEqual(len(identities), 2)
        self.assertEqual(mapping[("edge:d0", 9)], mapping[("edge:d1", 20)])
        self.assertEqual(mapping[("edge:d0", 25)], mapping[("edge:d1", 5)])
        self.assertNotEqual(mapping[("edge:d0", 9)], mapping[("edge:d0", 25)])
        self.assertEqual(identities[0].traversal_count, 2)

    def test_order_does_not_change_identity_assignment(self) -> None:
        samples = [EdgeEventSample("d1", 2, 4.2), EdgeEventSample("d0", 4, 4.0)]
        first = cluster_edge_event_identities(
            parent_id="p",
            edge_id="e",
            event=StructuralEvent.TURN,
            samples=samples,
        )
        second = cluster_edge_event_identities(
            parent_id="p",
            edge_id="e",
            event=StructuralEvent.TURN,
            samples=reversed(samples),
        )
        self.assertEqual(first, second)

    def test_node_identity_is_parent_scoped_and_wrong_event_rejected(self) -> None:
        self.assertEqual(node_event_identity("p", "n"), "p:node:n")
        with self.assertRaises(ValueError):
            cluster_edge_event_identities(
                parent_id="p",
                edge_id="e",
                event=StructuralEvent.JUNCTION,
                samples=[],
            )

    def test_pairs_prefer_cross_traversal_positive_and_closest_hard_negative(self) -> None:
        examples = [
            AssociationTeacherExample("a0", "p", "d0", "node_a", "junction", (4.0, 5.0)),
            AssociationTeacherExample("a1", "p", "d1", "node_a", "junction", (4.1, 5.0)),
            AssociationTeacherExample("b0", "p", "d2", "node_b", "junction", (4.2, 5.0)),
            AssociationTeacherExample("c0", "p", "d3", "node_c", "junction", (9.0, 9.0)),
        ]
        pairs = deterministic_association_pairs(examples)
        anchor_pairs = [pair for pair in pairs if pair.anchor_observation_id == "a0"]
        self.assertEqual(len(anchor_pairs), 2)
        self.assertEqual(anchor_pairs[0].paired_observation_id, "a1")
        self.assertEqual(anchor_pairs[0].pair_kind, "cross_traversal_positive")
        self.assertEqual(anchor_pairs[1].paired_observation_id, "b0")
        self.assertFalse(anchor_pairs[1].same_identity)

    def test_pair_order_is_deterministic(self) -> None:
        examples = [
            AssociationTeacherExample("a0", "p", "d0", "a", "turn", (1.0,)),
            AssociationTeacherExample("a1", "p", "d1", "a", "turn", (1.1,)),
            AssociationTeacherExample("b0", "p", "d2", "b", "turn", (1.2,)),
        ]
        self.assertEqual(
            deterministic_association_pairs(examples),
            deterministic_association_pairs(reversed(examples)),
        )


if __name__ == "__main__":
    unittest.main()
