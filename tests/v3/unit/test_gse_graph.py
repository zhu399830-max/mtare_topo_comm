from __future__ import annotations

import unittest

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES, ExitGeometryToken, GeometricSemanticObservation
from mtare_topo.topology.gse_graph import GSEGraphConfig, GeometrySemanticEventGraph


def observation(
    event: str,
    descriptor: tuple[float, ...],
    *,
    uncertainty: float = 0.1,
    exit_descriptor: tuple[float, ...] | None = None,
) -> GeometricSemanticObservation:
    probabilities = {name: 0.01 for name in EVENT_NAMES}
    probabilities[event] = 0.96
    return GeometricSemanticObservation(
        event_probabilities=probabilities,
        local_axis=(1.0, 0.0, 0.0),
        width_m=5.0,
        height_m=4.0,
        slope_deg=2.0,
        curvature_per_m=0.02,
        place_descriptor=descriptor,
        exit_tokens=(
            ExitGeometryToken(
                heading_robot_deg=0.0,
                opening_width_m=4.0,
                vertical_profile=(0.0, 0.0, 0.0, 0.0),
                descriptor=exit_descriptor,
                confidence=0.95,
            ),
        ) if exit_descriptor is not None else (),
        uncertainty=uncertainty,
    )


def config(**changes: float | int) -> GSEGraphConfig:
    values = {
        "stable_event_frames": 1,
        "minimum_event_travel_m": 4.0,
        "metric_anchor_interval_m": 100.0,
        "event_probability_threshold": 0.8,
        "maximum_uncertainty": 0.3,
        "association_radius_m": 20.0,
        "descriptor_minimum_similarity": 0.5,
        "exit_heading_tolerance_deg": 20.0,
        "exit_descriptor_minimum_similarity": 0.5,
        "exit_width_log_tolerance": 0.3,
        "exit_vertical_profile_tolerance": 1.0,
        "maximum_exit_count_difference": 1,
        "ambiguity_similarity_margin": 0.1,
    }
    values.update(changes)
    return GSEGraphConfig(**values)


class GeometrySemanticEventGraphTest(unittest.TestCase):
    class _FrozenBackend:
        maximum_candidate_distance_m = 16.0

        def __init__(
            self,
            scores: dict[tuple[int, int], float],
            node_scores: dict[int, float] | None = None,
        ) -> None:
            self.scores = scores
            self.node_scores = node_scores or {}

        def evaluate_node(self, key: int) -> dict[str, object]:
            score = self.node_scores.get(key, 1.0)
            return {
                "seed_scores": [score, score, score],
                "ensemble_score": score,
                "threshold": 0.982292910416921,
                "accepted": score >= 0.982292910416921,
                "rejection_reason": None if score >= 0.982292910416921 else "below_frozen_node_matchability_threshold",
            }

        def evaluate_pair(self, left: int, right: int, distance_m: float) -> dict[str, object]:
            if distance_m > self.maximum_candidate_distance_m:
                return {
                    "seed_scores": [0.0, 0.0, 0.0],
                    "ensemble_score": 0.0,
                    "distance_m": distance_m,
                    "accepted": False,
                    "rejection_reason": "outside_16m_candidate_domain",
                }
            score = self.scores.get(tuple(sorted((left, right))), 0.0)
            return {
                "seed_scores": [score, score, score],
                "ensemble_score": score,
                "distance_m": distance_m,
                "accepted": score >= 0.9431912302970886,
                "rejection_reason": None if score >= 0.9431912302970886 else "below_frozen_ensemble_threshold",
            }

    def test_confident_structural_route_start_is_a_structural_node(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("terminal", (1, 0)))
        self.assertEqual(graph.nodes[0]["node_kind"], "structural")
        self.assertEqual(graph.nodes[0]["event"], "terminal")

    def test_learned_event_creates_node_and_only_trace_creates_edge(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.edges), 0)
        result = graph.update(frame_index=1, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0, observation=observation("junction", (0, 1)))
        self.assertEqual(result["reason"], "new_junction")
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0]["kind"], "trace_verified")
        self.assertEqual(graph.edges[0]["length_m"], 5.0)
        self.assertEqual(graph.edges[0]["traversal_count"], 1)
        self.assertEqual(graph.edges[0]["geometry"]["minimum_width_m"], 5.0)

    def test_repeated_verified_traversal_updates_one_topological_edge(self) -> None:
        graph = GeometrySemanticEventGraph(config(metric_anchor_interval_m=4.0))
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=1, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0, observation=observation("junction", (0, 1)))
        graph.update(frame_index=2, route_arc_m=10.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0]["traversal_count"], 2)
        self.assertEqual(len(graph.edges[0]["traversals"]), 2)

    def test_high_uncertainty_event_does_not_create_structural_node(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        result = graph.update(frame_index=1, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0, observation=observation("junction", (0, 1), uncertainty=0.9))
        self.assertEqual(result["reason"], "no_event")
        self.assertEqual(len(graph.nodes), 1)

    def test_ambiguous_learned_association_creates_provisional_node(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=1, route_arc_m=5.0, xyz_m=(10, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        graph.update(frame_index=2, route_arc_m=10.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=3, route_arc_m=15.0, xyz_m=(20, 0, 0), yaw_deg=0, observation=observation("junction", (0, 1)))
        graph.update(frame_index=4, route_arc_m=20.0, xyz_m=(18, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        result = graph.update(frame_index=5, route_arc_m=25.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=observation("junction", (1, 1)))
        self.assertEqual(result["reason"], "ambiguous_association")
        self.assertEqual(result["association_status"], "provisional")
        self.assertEqual(result["association_candidate_count"], 2)
        self.assertEqual(graph.nodes[-1]["association_status"], "provisional")

    def test_monotonic_causal_contract(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        obs = observation("corridor", (1, 0))
        graph.update(frame_index=2, route_arc_m=1.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=obs)
        with self.assertRaises(ValueError):
            graph.update(frame_index=1, route_arc_m=2.0, xyz_m=(1, 0, 0), yaw_deg=0, observation=obs)

    def test_exit_descriptor_rejects_same_heading_wrong_branch_identity(self) -> None:
        graph = GeometrySemanticEventGraph(config())
        graph.update(
            frame_index=0,
            route_arc_m=0.0,
            xyz_m=(0, 0, 0),
            yaw_deg=0,
            observation=observation("corridor", (1, 0)),
        )
        graph.update(
            frame_index=1,
            route_arc_m=10.0,
            xyz_m=(10, 0, 0),
            yaw_deg=0,
            observation=observation("junction", (1, 0), exit_descriptor=(1, 0)),
        )
        graph.update(
            frame_index=2,
            route_arc_m=30.0,
            xyz_m=(30, 0, 0),
            yaw_deg=0,
            observation=observation("corridor", (1, 0)),
        )
        graph.update(
            frame_index=3,
            route_arc_m=50.0,
            xyz_m=(50, 0, 0),
            yaw_deg=0,
            observation=observation("junction", (1, 0), exit_descriptor=(0, 1)),
        )
        result = graph.update(
            frame_index=4,
            route_arc_m=60.0,
            xyz_m=(40, 0, 0),
            yaw_deg=0,
            observation=observation("corridor", (1, 0)),
        )
        result = graph.update(
            frame_index=5,
            route_arc_m=70.0,
            xyz_m=(30, 0, 0),
            yaw_deg=0,
            observation=observation("junction", (1, 0), exit_descriptor=(1, 0)),
        )
        self.assertEqual(result["reason"], "learned_association")
        self.assertEqual(result["node_id"], 1)
        self.assertEqual(result["association_candidate_count"], 1)

    def test_frozen_ensemble_directly_controls_merge_and_freezes_representative_key(self) -> None:
        backend = self._FrozenBackend({(10, 30): 0.97})
        graph = GeometrySemanticEventGraph(
            config(association_radius_m=16.0, ambiguity_similarity_margin=0.01),
            association_reason="frozen_exit_token_ensemble",
            association_backend=backend,
        )
        graph.update(
            frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
            observation=observation("junction", (1, 0)), association_key=10,
        )
        graph.update(
            frame_index=1, route_arc_m=10.0, xyz_m=(10, 0, 0), yaw_deg=0,
            observation=observation("corridor", (1, 0)), association_key=11,
        )
        graph.update(
            frame_index=2, route_arc_m=20.0, xyz_m=(20, 0, 0), yaw_deg=0,
            observation=observation("terminal", (0, 1)), association_key=20,
        )
        graph.update(
            frame_index=3, route_arc_m=30.0, xyz_m=(10, 0, 0), yaw_deg=0,
            observation=observation("corridor", (1, 0)), association_key=21,
        )
        result = graph.update(
            frame_index=4, route_arc_m=40.0, xyz_m=(1, 0, 0), yaw_deg=0,
            observation=observation("junction", (0, 1)), association_key=30,
        )
        self.assertEqual(result["reason"], "frozen_exit_token_ensemble")
        self.assertEqual(result["node_id"], 0)
        self.assertEqual(result["association_candidate_count"], 1)
        self.assertEqual(result["association_accepted_candidate_count"], 1)
        self.assertEqual(graph.nodes[0]["association_key"], 10)

    def test_frozen_ensemble_rejection_creates_new_node_without_old_descriptor_gate(self) -> None:
        backend = self._FrozenBackend({(1, 2): 0.90})
        graph = GeometrySemanticEventGraph(
            config(association_radius_m=16.0, descriptor_minimum_similarity=0.99),
            association_reason="frozen_exit_token_ensemble",
            association_backend=backend,
        )
        graph.update(
            frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
            observation=observation("junction", (1, 0)), association_key=1,
        )
        graph.update(
            frame_index=1, route_arc_m=4.0, xyz_m=(4, 0, 0), yaw_deg=0,
            observation=observation("corridor", (1, 0)), association_key=3,
        )
        result = graph.update(
            frame_index=2, route_arc_m=8.0, xyz_m=(1, 0, 0), yaw_deg=0,
            observation=observation("junction", (1, 0)), association_key=2,
        )
        self.assertEqual(result["reason"], "new_junction")
        self.assertEqual(result["association_accepted_candidate_count"], 0)
        self.assertEqual(result["association_candidates"][0]["rejection_reason"], "below_frozen_ensemble_threshold")

    def test_open_set_node_gate_rejects_structural_trigger_before_association(self) -> None:
        backend = self._FrozenBackend({}, node_scores={1: 0.50, 2: 0.99})
        graph = GeometrySemanticEventGraph(
            config(association_radius_m=16.0),
            association_reason="frozen_exit_token_ensemble",
            association_backend=backend,
            node_generation_backend=backend,
        )
        start = graph.update(
            frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
            observation=observation("junction", (1, 0)), association_key=1,
        )
        self.assertEqual(graph.nodes[0]["node_kind"], "anchor")
        self.assertFalse(start["node_generation_decision"]["accepted"])
        result = graph.update(
            frame_index=1, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0,
            observation=observation("terminal", (0, 1)), association_key=2,
        )
        self.assertEqual(result["reason"], "new_terminal")
        self.assertTrue(result["node_generation_decision"]["accepted"])

    def test_one_structural_episode_emits_once_and_resets_only_after_stable_corridor(self) -> None:
        graph = GeometrySemanticEventGraph(config(stable_event_frames=2, minimum_event_travel_m=1.0))
        graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=1, route_arc_m=1.0, xyz_m=(1, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        first = graph.update(frame_index=2, route_arc_m=2.0, xyz_m=(2, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        suppressed = graph.update(frame_index=3, route_arc_m=8.0, xyz_m=(8, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        self.assertEqual(first["reason"], "new_junction")
        self.assertEqual(suppressed["reason"], "no_event")
        graph.update(frame_index=4, route_arc_m=9.0, xyz_m=(9, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=5, route_arc_m=10.0, xyz_m=(10, 0, 0), yaw_deg=0, observation=observation("corridor", (1, 0)))
        graph.update(frame_index=6, route_arc_m=11.0, xyz_m=(11, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        second = graph.update(frame_index=7, route_arc_m=12.0, xyz_m=(12, 0, 0), yaw_deg=0, observation=observation("junction", (1, 0)))
        self.assertNotEqual(second["reason"], "no_event")


if __name__ == "__main__":
    unittest.main()
