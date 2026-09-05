from __future__ import annotations

import numpy as np

from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES, ExitGeometryToken, GeometricSemanticObservation,
)
from mtare_topo.topology.factorized_gse_graph import (
    FactorizedCausalDecisionBackend, FactorizedConsensusMetricBackend,
    FactorizedDecisionGraph,
)
from mtare_topo.topology.gse_graph import GSEGraphConfig


def _observation(event: str, *, width: float = 5.0) -> GeometricSemanticObservation:
    probability = {name: .01 for name in EVENT_NAMES}
    probability[event] = .96
    return GeometricSemanticObservation(
        event_probabilities=probability, local_axis=(1.0, 0.0, 0.0),
        width_m=width, height_m=4.0, slope_deg=2.0,
        curvature_per_m=.02, place_descriptor=(1.0, 0.0),
        exit_tokens=(ExitGeometryToken(
            heading_robot_deg=0.0, opening_width_m=4.0,
            vertical_profile=(0.0, 0.0, 0.0, 0.0),
            descriptor=(1.0, 0.0), confidence=.95,
        ),), uncertainty=.1,
    )


def _config() -> GSEGraphConfig:
    return GSEGraphConfig(
        stable_event_frames=1, minimum_event_travel_m=2.0,
        metric_anchor_interval_m=100.0, event_probability_threshold=.8,
        maximum_uncertainty=.3, association_radius_m=4.0,
        descriptor_minimum_similarity=.5, exit_heading_tolerance_deg=20.0,
        exit_descriptor_minimum_similarity=.5, exit_width_log_tolerance=.3,
        exit_vertical_profile_tolerance=1.0, maximum_exit_count_difference=1,
        ambiguity_similarity_margin=.1,
    )


def _backend(scores=None) -> FactorizedConsensusMetricBackend:
    return FactorizedConsensusMetricBackend(
        scores or {}, seed_thresholds=(.8, .8, .8), votes_required=2,
        maximum_candidate_distance_m=4.0,
    )


def test_consensus_metric_backend_requires_two_votes_and_metric_cap() -> None:
    backend = _backend({(10, 20): (.9, .81, .1)})
    accepted = backend.evaluate_pair(20, 10, 3.0)
    assert accepted["accepted"] is True
    assert accepted["votes"] == 2
    assert accepted["seed_acceptance"] == [True, True, False]
    assert backend.evaluate_pair(10, 20, 4.01)["rejection_reason"] == "outside_frozen_metric_cap"
    assert backend.evaluate_pair(10, 30, 2.0)["rejection_reason"] == "pair_score_unavailable"


def test_turn_and_geometry_transition_never_create_decision_nodes() -> None:
    graph = FactorizedDecisionGraph(_config(), association_backend=_backend())
    first = graph.update(
        frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
        observation=_observation("turn"), association_key=1,
    )
    second = graph.update(
        frame_index=1, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0,
        observation=_observation("geometry_transition"), association_key=2,
    )
    assert graph.nodes[0]["node_kind"] == "anchor"
    assert first["reason"] == "route_start"
    assert second["reason"] == "no_event"
    assert len(graph.nodes) == 1


def test_verified_edge_retains_ordered_learned_geometry_profile() -> None:
    graph = FactorizedDecisionGraph(_config(), association_backend=_backend())
    graph.update(
        frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
        observation=_observation("junction", width=5.0), association_key=1,
    )
    graph.update(
        frame_index=1, route_arc_m=2.0, xyz_m=(2, 0, 0), yaw_deg=0,
        observation=_observation("turn", width=4.5), association_key=2,
    )
    result = graph.update(
        frame_index=2, route_arc_m=5.0, xyz_m=(5, 0, 0), yaw_deg=0,
        observation=_observation("terminal", width=4.0), association_key=3,
    )
    assert result["reason"] == "new_terminal"
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge["execution_state"] == "verified"
    assert [sample["arc_m"] for sample in edge["geometry_profiles"][0]] == [0.0, 2.0, 5.0]
    assert [sample["width_m"] for sample in edge["geometry_profiles"][0]] == [5.0, 4.5, 4.0]
    assert edge["traversability_summary"]["minimum_width_m"] == 4.0


def test_multiple_accepted_candidates_create_provisional_node_without_score_ranking() -> None:
    backend = _backend({(1, 3): (.9, .9, .1), (2, 3): (.95, .95, .1)})
    graph = FactorizedDecisionGraph(_config(), association_backend=backend)
    graph.update(
        frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0,
        observation=_observation("junction"), association_key=1,
    )
    graph.update(
        frame_index=1, route_arc_m=3.0, xyz_m=(3, 0, 0), yaw_deg=0,
        observation=_observation("turn"), association_key=10,
    )
    graph.update(
        frame_index=2, route_arc_m=6.0, xyz_m=(6, 0, 0), yaw_deg=0,
        observation=_observation("terminal"), association_key=2,
    )
    graph.update(
        frame_index=3, route_arc_m=8.0, xyz_m=(5, 0, 0), yaw_deg=0,
        observation=_observation("turn"), association_key=11,
    )
    result = graph.update(
        frame_index=4, route_arc_m=11.0, xyz_m=(3, 0, 0), yaw_deg=0,
        observation=_observation("junction"), association_key=3,
    )
    assert result["reason"] == "ambiguous_association"
    assert result["association_status"] == "provisional"
    assert result["association_accepted_candidate_count"] == 2
    assert graph.nodes[-1]["association_status"] == "provisional"


def test_causal_backend_accepts_only_collapsed_junction_terminal_trigger() -> None:
    probability = np.asarray([
        [.995, .003, .001, .001, 0.0],
        [.005, .990, .002, .002, .001],
        [.004, .991, .002, .002, .001],
        [.005, .001, .001, .992, .001],
    ])
    backend = FactorizedCausalDecisionBackend(
        global_sequence_index=[10, 11, 12, 13],
        event_probability=probability,
        traversal_id=["t"] * 4,
        sequence_index=[0, 1, 2, 3],
        boundary_offset_m=[0.0] * 4,
        uncertainty=[0.0] * 4,
        structural_threshold=.986,
    )
    # Accepted rows 1--3 are one episode. Row 2 has its maximum structural
    # score and is a junction; the adjacent turn cannot create another node.
    assert not backend.evaluate_node(10)["accepted"]
    assert not backend.evaluate_node(11)["accepted"]
    assert backend.evaluate_node(12)["accepted"]
    assert backend.evaluate_node(12)["trigger_event"] == "junction"
    assert not backend.evaluate_node(13)["accepted"]
    assert backend.decision_trigger_count == 1
