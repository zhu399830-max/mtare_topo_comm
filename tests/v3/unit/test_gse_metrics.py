from __future__ import annotations

import unittest

from mtare_topo.evaluation.gse_metrics import (
    association_metrics,
    continuous_geometry_mae,
    graph_invariants,
    instance_mapped_graph_f1,
    macro_f1,
    mapped_graph_f1,
    relative_geometry_improvement,
    relative_mae_improvement,
)


class GSEMetricsTest(unittest.TestCase):
    def test_event_macro_f1_and_geometry_mae(self) -> None:
        classification = macro_f1(
            ["corridor", "junction", "terminal", "turn", "geometry_transition"],
            ["corridor", "junction", "corridor", "turn", "geometry_transition"],
            ["corridor", "junction", "terminal", "turn", "geometry_transition"],
        )
        self.assertGreater(classification["macro_f1"], 0.7)
        geometry = continuous_geometry_mae({"width": [4.0, 6.0]}, {"width": [5.0, 5.0]})
        self.assertEqual(geometry["width_mae"], 1.0)
        self.assertAlmostEqual(relative_mae_improvement(1.0, 0.8), 0.2)

    def test_association_precision_counts_false_merge_and_rejection(self) -> None:
        values = association_metrics(
            [
                {"reason": "learned_association", "association_status": "persistent", "evaluator_gt_identity": "a", "evaluator_matched_node_gt_identity": "a", "evaluator_has_existing_true_node": True},
                {"reason": "learned_association", "association_status": "persistent", "evaluator_gt_identity": "b", "evaluator_matched_node_gt_identity": "c", "evaluator_has_existing_true_node": True},
                {"reason": "ambiguous_association", "association_status": "provisional", "evaluator_gt_identity": "d", "evaluator_has_existing_true_node": True},
                {"reason": "no_event", "association_status": None, "evaluator_gt_identity": "d", "evaluator_has_existing_true_node": True, "evaluator_association_opportunity": False},
            ]
        )
        self.assertEqual(values["merge_attempts"], 2)
        self.assertEqual(values["false_merges"], 1)
        self.assertEqual(values["provisional_rejections"], 1)
        self.assertEqual(values["association_precision"], 0.5)
        self.assertAlmostEqual(values["association_recall"], 1.0 / 3.0)

    def test_graph_f1_and_invariants(self) -> None:
        self.assertEqual(graph_invariants([0, 1, 2], [(0, 1), (1, 2), (2, 0)])["cycle_rank"], 1)
        result = mapped_graph_f1(
            predicted_node_ids=[0, 1, 2],
            predicted_edges=[(0, 1), (1, 2)],
            predicted_to_teacher={0: "a", 1: "b", 2: "c"},
            teacher_node_ids=["a", "b", "c"],
            teacher_edges=[("a", "b"), ("b", "c")],
        )
        self.assertEqual(result["node"]["f1"], 1.0)
        self.assertEqual(result["edge"]["f1"], 1.0)
        self.assertEqual(result["cycle_rank_error"], 0)

    def test_instance_graph_metric_penalizes_duplicate_nodes_and_edges(self) -> None:
        result = instance_mapped_graph_f1(
            predicted_node_ids=[0, 1, 2],
            predicted_edges=[(0, 2), (1, 2)],
            predicted_to_teacher={0: "a", 1: "a", 2: "b"},
            teacher_node_ids=["a", "b"],
            teacher_edges=[("a", "b")],
        )
        self.assertAlmostEqual(result["node"]["precision"], 2.0 / 3.0)
        self.assertEqual(result["node"]["recall"], 1.0)
        self.assertEqual(result["duplicate_or_unmapped_predicted_nodes"], 1)
        self.assertEqual(result["edge"]["precision"], 0.5)
        self.assertEqual(result["duplicate_or_unmapped_predicted_edges"], 1)

    def test_relative_geometry_gate_uses_unitless_macro_and_regression_guard(self) -> None:
        result = relative_geometry_improvement(
            {"width": 0.8, "height": 0.9, "slope": 0.85, "curvature": 1.02},
            {"width": 1.0, "height": 1.0, "slope": 1.0, "curvature": 1.0},
        )
        self.assertAlmostEqual(result["macro_relative_improvement"], 0.1075)
        self.assertAlmostEqual(result["worst_field_relative_improvement"], -0.02)
        self.assertTrue(result["passed"])

    def test_relative_geometry_gate_rejects_material_single_field_regression(self) -> None:
        result = relative_geometry_improvement(
            {"width": 0.5, "height": 0.5, "slope": 0.5, "curvature": 1.10},
            {"width": 1.0, "height": 1.0, "slope": 1.0, "curvature": 1.0},
        )
        self.assertTrue(result["macro_gate_passed"])
        self.assertFalse(result["no_material_field_regression"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
