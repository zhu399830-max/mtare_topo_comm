"""Deterministic route and structural-graph contracts for offline GSE replay."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.evaluation.gse_metrics import association_metrics, instance_mapped_graph_f1
from mtare_topo.semantics.gse_observation_adapter import geometric_semantic_observation_from_arrays
from mtare_topo.semantics.geometric_semantics import GeometricSemanticObservation
from mtare_topo.topology.gse_graph import GSEGraphConfig, GeometrySemanticEventGraph
from mtare_topo.topology.factorized_gse_graph import FactorizedDecisionGraph
from mtare_topo.topology.gse_rule_graph import RuleAssociatedGeometryEventGraph


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


@dataclass(frozen=True)
class DirectedTraversal:
    traversal_id: str
    edge_id: str
    from_node_id: str
    to_node_id: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DirectedTraversal":
        values = cls(
            traversal_id=str(record["traversal_id"]),
            edge_id=str(record["edge_id"]),
            from_node_id=str(record["from_node_id"]),
            to_node_id=str(record["to_node_id"]),
        )
        if values.from_node_id == values.to_node_id:
            raise ValueError("directed traversal cannot be a self-loop")
        return values


def directed_euler_order(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return a lexicographic Euler circuit over both directions of every edge."""

    traversals = [DirectedTraversal.from_record(record) for record in records]
    if not traversals or len({item.traversal_id for item in traversals}) != len(traversals):
        raise ValueError("directed traversal manifest is empty or has duplicate IDs")
    by_edge: dict[str, list[DirectedTraversal]] = defaultdict(list)
    outgoing: dict[str, list[DirectedTraversal]] = defaultdict(list)
    indegree: Counter[str] = Counter()
    outdegree: Counter[str] = Counter()
    for traversal in traversals:
        by_edge[traversal.edge_id].append(traversal)
        outgoing[traversal.from_node_id].append(traversal)
        outdegree[traversal.from_node_id] += 1
        indegree[traversal.to_node_id] += 1
    for edge_id, pair in by_edge.items():
        if (
            len(pair) != 2
            or pair[0].from_node_id != pair[1].to_node_id
            or pair[0].to_node_id != pair[1].from_node_id
        ):
            raise ValueError(f"physical edge {edge_id} does not have one reverse traversal pair")
    nodes = sorted(set(indegree) | set(outdegree))
    if any(indegree[node] != outdegree[node] for node in nodes):
        raise ValueError("directed traversal graph is not balanced")
    for values in outgoing.values():
        values.sort(key=lambda item: item.traversal_id, reverse=True)

    start = nodes[0]
    node_stack = [start]
    edge_stack: list[str] = []
    reverse_circuit: list[str] = []
    while node_stack:
        node = node_stack[-1]
        if outgoing[node]:
            traversal = outgoing[node].pop()
            node_stack.append(traversal.to_node_id)
            edge_stack.append(traversal.traversal_id)
        else:
            node_stack.pop()
            if edge_stack:
                reverse_circuit.append(edge_stack.pop())
    order = list(reversed(reverse_circuit))
    if len(order) != len(traversals) or len(set(order)) != len(order):
        raise RuntimeError("directed graph is disconnected or Euler replay lost traversals")
    lookup = {item.traversal_id: item for item in traversals}
    for first, second in zip(order, order[1:] + order[:1], strict=True):
        if lookup[first].to_node_id != lookup[second].from_node_id:
            raise RuntimeError("Euler traversal order is not physically continuous")
    return order


def teacher_structural_graph(
    observations: Sequence[Mapping[str, Any]],
    traversal_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collapse corridor samples into edges between objective structural identities."""

    if not observations:
        raise ValueError("teacher structural graph requires observations")
    by_edge: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    event_by_identity: dict[str, str] = {}
    for record in observations:
        identity = record.get("identity")
        event = str(record["event"])
        if identity is None or event == "corridor":
            continue
        identity = str(identity)
        old = event_by_identity.setdefault(identity, event)
        if old != event:
            raise ValueError("one Teacher structural identity has multiple event types")
        by_edge[str(record["edge_id"])][identity].append(float(record["canonical_edge_arc_m"]))
    nodes = sorted(event_by_identity)
    edges: set[tuple[str, str]] = set()
    if traversal_records is None:
        for identities in by_edge.values():
            ordered = sorted(
                ((sum(values) / len(values), identity) for identity, values in identities.items()),
                key=lambda item: (item[0], item[1]),
            )
            for (_, first), (_, second) in zip(ordered[:-1], ordered[1:]):
                if first != second:
                    edges.add(tuple(sorted((first, second))))
    else:
        node_identity = {}
        for identity in nodes:
            if ":node:" in identity:
                node_identity[identity.rsplit(":node:", 1)[1]] = identity
        edge_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in traversal_records:
            edge_records[str(record["edge_id"])].append(record)
        adjacency: dict[str, set[str]] = defaultdict(set)
        placeholder_prefix = "__tng__:"

        def endpoint(node_id: str) -> str:
            return node_identity.get(node_id, placeholder_prefix + node_id)

        for edge_id, pair in edge_records.items():
            if len(pair) != 2:
                raise ValueError("Teacher traversal manifest lacks a reverse edge pair")
            canonical = sorted(pair, key=lambda item: str(item["traversal_id"]))[0]
            ordered_identities = [
                identity
                for _, identity in sorted(
                    (
                        (sum(values) / len(values), identity)
                        for identity, values in by_edge.get(edge_id, {}).items()
                    ),
                    key=lambda item: (item[0], item[1]),
                )
            ]
            chain = [endpoint(str(canonical["from_node_id"])), *ordered_identities, endpoint(str(canonical["to_node_id"]))]
            compact = [chain[0]]
            for value in chain[1:]:
                if value != compact[-1]:
                    compact.append(value)
            for first, second in zip(compact[:-1], compact[1:]):
                if first != second:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        placeholders = {node for node in adjacency if node.startswith(placeholder_prefix)}
        seen: set[str] = set()
        for start in sorted(placeholders):
            if start in seen:
                continue
            stack = [start]
            component: set[str] = set()
            external: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                seen.add(current)
                for neighbour in adjacency[current]:
                    if neighbour in placeholders:
                        stack.append(neighbour)
                    else:
                        external.add(neighbour)
            if len(external) > 2:
                raise RuntimeError("an unobservable Teacher branch node would make topology semantics ambiguous")
            if len(external) == 2:
                first, second = sorted(external)
                edges.add((first, second))
        for first, neighbours in adjacency.items():
            if first in placeholders:
                continue
            for second in neighbours:
                if second not in placeholders and first != second:
                    edges.add(tuple(sorted((first, second))))
    return {
        "node_ids": nodes,
        "edges": [list(edge) for edge in sorted(edges)],
        "event_by_identity": event_by_identity,
    }


def collapse_to_structural_graph(
    nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Suppress only degree-two metric anchors; retain missed branch/end anchors."""

    node_by_id = {int(node["id"]): node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("predicted graph node IDs are not unique")
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_by_id}
    for edge in edges:
        first, second = int(edge["from"]), int(edge["to"])
        if first not in adjacency or second not in adjacency or first == second:
            raise ValueError("predicted graph edge is invalid")
        adjacency[first].add(second)
        adjacency[second].add(first)
    retained = {
        node_id
        for node_id, node in node_by_id.items()
        if str(node["node_kind"]) == "structural" or len(adjacency[node_id]) != 2
    }
    collapsed_edges: set[tuple[int, int]] = set()
    for start in sorted(retained):
        for neighbour in sorted(adjacency[start]):
            previous, current = start, neighbour
            visited = {start}
            while current not in retained:
                if current in visited or len(adjacency[current]) != 2:
                    raise RuntimeError("metric-anchor suppression encountered an invalid chain")
                visited.add(current)
                next_values = sorted(adjacency[current] - {previous})
                if len(next_values) != 1:
                    raise RuntimeError("degree-two metric anchor does not have one forward neighbour")
                previous, current = current, next_values[0]
            if current != start:
                collapsed_edges.add(tuple(sorted((start, current))))
    return {
        "node_ids": sorted(retained),
        "edges": [list(edge) for edge in sorted(collapsed_edges)],
        "suppressed_metric_anchor_ids": sorted(set(node_by_id) - retained),
    }


def strict_predicted_node_mapping(
    nodes: Sequence[Mapping[str, Any]],
    frame_teacher: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Map a node only when all same-event observations have one GT identity."""

    mapping: dict[int, str] = {}
    conflicts: dict[int, list[str]] = {}
    for node in nodes:
        node_id = int(node["id"])
        event = str(node["event"])
        identities = {
            str(frame_teacher[int(frame)]["identity"])
            for frame in node.get("observation_frames", ())
            if int(frame) in frame_teacher
            and frame_teacher[int(frame)].get("identity") is not None
            and str(frame_teacher[int(frame)]["event"]) == event
        }
        if len(identities) == 1:
            mapping[node_id] = next(iter(identities))
        elif len(identities) > 1:
            conflicts[node_id] = sorted(identities)
    return mapping, conflicts


def replay_typed_observation_world(
    *,
    traversal_records: Sequence[Mapping[str, Any]],
    teacher_observations: Sequence[Mapping[str, Any]],
    pose_by_sequence_index: Mapping[int, Mapping[str, Any]],
    observation_by_sequence_index: Mapping[int, GeometricSemanticObservation],
    graph_config: GSEGraphConfig,
    association_mode: str = "learned",
    association_backend: Any | None = None,
    node_generation_backend: Any | None = None,
) -> dict[str, Any]:
    """Replay one complete world from already materialized causal observations."""

    if not traversal_records or not teacher_observations:
        raise ValueError("one world replay requires traversals and Teacher observations")
    traversal_order = directed_euler_order(traversal_records)
    traversal_lookup = {str(record["traversal_id"]): record for record in traversal_records}
    observations_by_traversal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in teacher_observations:
        observations_by_traversal[str(record["traversal_id"])].append(record)
    for records in observations_by_traversal.values():
        records.sort(key=lambda item: (float(item["traversal_arc_m"]), int(item["global_sequence_index"])))

    if association_mode == "learned":
        if association_backend is not None or node_generation_backend is not None:
            raise ValueError("legacy learned association cannot receive a frozen backend")
        graph = GeometrySemanticEventGraph(graph_config)
    elif association_mode == "frozen_exit_token_ensemble":
        if association_backend is None:
            raise ValueError("frozen ensemble association requires its world score backend")
        graph = GeometrySemanticEventGraph(
            graph_config,
            association_reason="frozen_exit_token_ensemble",
            association_backend=association_backend,
        )
    elif association_mode == "frozen_ensemble_with_open_set_node_gate":
        if association_backend is None or node_generation_backend is None:
            raise ValueError("open-set node-gated ensemble requires pair and node backends")
        graph = GeometrySemanticEventGraph(
            graph_config,
            association_reason="frozen_exit_token_ensemble",
            association_backend=association_backend,
            node_generation_backend=node_generation_backend,
        )
    elif association_mode == "frozen_event_node_ensemble":
        if association_backend is None or node_generation_backend is None:
            raise ValueError("event-node ensemble requires pair and node backends")
        graph = GeometrySemanticEventGraph(
            graph_config,
            association_reason="frozen_exit_token_ensemble",
            association_backend=association_backend,
            node_generation_backend=node_generation_backend,
        )
    elif association_mode == "factorized_consensus_metric":
        if association_backend is None:
            raise ValueError("factorized graph requires its frozen consensus backend")
        graph = FactorizedDecisionGraph(
            graph_config,
            association_backend=association_backend,
            node_generation_backend=node_generation_backend,
        )
    elif association_mode == "rule":
        if association_backend is not None or node_generation_backend is not None:
            raise ValueError("rule association cannot receive a frozen learned backend")
        graph = RuleAssociatedGeometryEventGraph(graph_config)
    else:
        raise ValueError("unsupported association_mode")
    frame_teacher: dict[int, Mapping[str, Any]] = {}
    evaluator_decisions: list[dict[str, Any]] = []
    node_truth: dict[int, set[str]] = defaultdict(set)
    replayed_sequence_indices: list[int] = []
    replayed_sequence_set: set[int] = set()
    traversal_start_arc = 0.0
    first_observed_arc: float | None = None
    last_observed_route_arc = 0.0
    frame_index = 0
    for traversal_id in traversal_order:
        traversal = traversal_lookup[traversal_id]
        rows = observations_by_traversal.get(traversal_id, [])
        previous_local_arc: float | None = None
        for row_offset, teacher in enumerate(rows):
            local_arc = float(teacher["traversal_arc_m"])
            if row_offset > 0 and previous_local_arc is not None:
                if local_arc <= previous_local_arc:
                    raise RuntimeError("Teacher traversal observations are not strictly ordered")
            absolute_arc = traversal_start_arc + local_arc
            if first_observed_arc is None:
                first_observed_arc = absolute_arc
            route_arc = absolute_arc - first_observed_arc
            if route_arc + 1e-12 < last_observed_route_arc:
                raise RuntimeError("Euler replay route arc regressed")
            last_observed_route_arc = route_arc
            sequence_index = int(teacher["global_sequence_index"])
            if sequence_index in replayed_sequence_set:
                raise RuntimeError("Euler replay duplicated a sequence")
            replayed_sequence_indices.append(sequence_index)
            replayed_sequence_set.add(sequence_index)
            if sequence_index not in observation_by_sequence_index or sequence_index not in pose_by_sequence_index:
                raise RuntimeError("typed observation or replay pose is missing for a Teacher observation")
            pose = pose_by_sequence_index[sequence_index]
            observation = observation_by_sequence_index[sequence_index]
            if not isinstance(observation, GeometricSemanticObservation):
                raise TypeError("online graph input must be a typed GeometricSemanticObservation")
            identity = None if teacher.get("identity") is None else str(teacher["identity"])
            existing_truth_nodes = [
                node_id for node_id, identities in node_truth.items() if identity is not None and identity in identities
            ]
            before_truth = {node_id: set(values) for node_id, values in node_truth.items()}
            update_kwargs = {
                "frame_index": frame_index,
                "route_arc_m": route_arc,
                "xyz_m": pose["axis_xyz_m"],
                "yaw_deg": float(pose["yaw_deg"]),
                "observation": observation,
            }
            if association_mode in {
                "frozen_exit_token_ensemble",
                "frozen_ensemble_with_open_set_node_gate",
                "frozen_event_node_ensemble",
                "factorized_consensus_metric",
            }:
                update_kwargs["association_key"] = sequence_index
            result = graph.update(
                **update_kwargs,
            )
            target_node = int(result["node_id"])
            if identity is not None and (
                result["created"]
                or result["reason"] in {
                    "learned_association",
                    "rule_association",
                    "frozen_exit_token_ensemble",
                    "factorized_consensus_metric",
                }
            ):
                node_truth[target_node].add(identity)
            matched_truth = before_truth.get(target_node, set())
            decision = {
                **result,
                "global_sequence_index": sequence_index,
                "traversal_id": traversal_id,
                "evaluator_gt_identity": identity,
                "evaluator_association_opportunity": bool(
                    result["reason"] in {
                        "learned_association",
                        "rule_association",
                        "frozen_exit_token_ensemble",
                        "factorized_consensus_metric",
                        "ambiguous_association",
                    }
                    or str(result["reason"]).startswith("new_")
                ),
                "evaluator_has_existing_true_node": bool(existing_truth_nodes),
                "evaluator_matched_node_gt_identity": next(iter(matched_truth)) if len(matched_truth) == 1 else None,
            }
            evaluator_decisions.append(decision)
            frame_teacher[frame_index] = teacher
            frame_index += 1
            previous_local_arc = local_arc
        traversal_start_arc += float(traversal["length_m"])

    expected_sequences = {int(record["global_sequence_index"]) for record in teacher_observations}
    if replayed_sequence_set != expected_sequences or len(replayed_sequence_indices) != len(expected_sequences):
        raise RuntimeError("Euler replay did not consume every world sequence exactly once")
    collapsed = collapse_to_structural_graph(graph.nodes, graph.edges)
    retained_nodes = [graph.nodes[node_id] for node_id in collapsed["node_ids"]]
    mapping, conflicts = strict_predicted_node_mapping(retained_nodes, frame_teacher)
    teacher_graph = teacher_structural_graph(teacher_observations, traversal_records)
    teacher_positions: dict[str, list[np.ndarray]] = defaultdict(list)
    for teacher in teacher_observations:
        identity = teacher.get("identity")
        if identity is None:
            continue
        sequence_index = int(teacher["global_sequence_index"])
        teacher_positions[str(identity)].append(
            np.asarray(pose_by_sequence_index[sequence_index]["axis_xyz_m"], dtype=np.float64)
        )
    teacher_node_xyz_m = {
        identity: np.median(np.stack(values), axis=0).astype(float).tolist()
        for identity, values in sorted(teacher_positions.items())
        if identity in set(teacher_graph["node_ids"])
    }
    if set(teacher_node_xyz_m) != set(teacher_graph["node_ids"]):
        raise RuntimeError("Teacher structural graph lacks objective node coordinates")
    topology = instance_mapped_graph_f1(
        predicted_node_ids=collapsed["node_ids"],
        predicted_edges=collapsed["edges"],
        predicted_to_teacher=mapping,
        teacher_node_ids=teacher_graph["node_ids"],
        teacher_edges=teacher_graph["edges"],
    )
    return {
        "traversal_order": traversal_order,
        "replayed_sequences": len(replayed_sequence_indices),
        "observed_route_arc_m": last_observed_route_arc,
        "physical_route_length_m": traversal_start_arc,
        "nodes": graph.nodes,
        "edges": graph.edges,
        "decision_trace": evaluator_decisions,
        "collapsed_graph": collapsed,
        "predicted_to_teacher": {str(key): value for key, value in sorted(mapping.items())},
        "identity_conflicts": {str(key): value for key, value in sorted(conflicts.items())},
        "teacher_graph": teacher_graph,
        "teacher_node_xyz_m": teacher_node_xyz_m,
        "association_metrics": association_metrics(evaluator_decisions),
        "topology_metrics": topology,
        "association_mode": association_mode,
    }


def replay_gse_world(
    *,
    traversal_records: Sequence[Mapping[str, Any]],
    teacher_observations: Sequence[Mapping[str, Any]],
    pose_by_sequence_index: Mapping[int, Mapping[str, Any]],
    model_outputs: Mapping[str, np.ndarray],
    output_row_by_sequence_index: Mapping[int, int],
    graph_config: GSEGraphConfig,
    event_temperature: float,
    exit_presence_threshold: float,
) -> dict[str, Any]:
    """Replay one learned-output world in a physical Euler route."""

    sequence_indices = sorted({int(record["global_sequence_index"]) for record in teacher_observations})
    observations = {}
    for sequence_index in sequence_indices:
        if sequence_index not in output_row_by_sequence_index:
            raise RuntimeError("model output is missing for a Teacher observation")
        observations[sequence_index] = geometric_semantic_observation_from_arrays(
            model_outputs,
            int(output_row_by_sequence_index[sequence_index]),
            event_temperature=event_temperature,
            exit_presence_threshold=exit_presence_threshold,
        )
    return replay_typed_observation_world(
        traversal_records=traversal_records,
        teacher_observations=teacher_observations,
        pose_by_sequence_index=pose_by_sequence_index,
        observation_by_sequence_index=observations,
        graph_config=graph_config,
    )


def gse_graph_parameter_grid(
    *,
    event_probability_threshold: float,
    maximum_uncertainty: float,
    descriptor_minimum_similarity: float,
    exit_descriptor_minimum_similarity: float,
    exit_heading_tolerance_deg: float,
    exit_width_log_tolerance: float,
    exit_vertical_profile_tolerance: float,
) -> list[GSEGraphConfig]:
    """The predeclared 3^5 validation grid; calibrated quantities stay fixed."""

    result = []
    for stable_event_frames in (2, 3, 4):
        for minimum_event_travel_m in (2.0, 4.0, 6.0):
            for metric_anchor_interval_m in (20.0, 30.0, 40.0):
                for association_radius_m in (8.0, 12.0, 16.0):
                    for ambiguity_similarity_margin in (0.02, 0.05, 0.10):
                        result.append(
                            GSEGraphConfig(
                                stable_event_frames=stable_event_frames,
                                minimum_event_travel_m=minimum_event_travel_m,
                                metric_anchor_interval_m=metric_anchor_interval_m,
                                event_probability_threshold=event_probability_threshold,
                                maximum_uncertainty=maximum_uncertainty,
                                association_radius_m=association_radius_m,
                                descriptor_minimum_similarity=descriptor_minimum_similarity,
                                exit_heading_tolerance_deg=exit_heading_tolerance_deg,
                                exit_descriptor_minimum_similarity=exit_descriptor_minimum_similarity,
                                exit_width_log_tolerance=exit_width_log_tolerance,
                                exit_vertical_profile_tolerance=exit_vertical_profile_tolerance,
                                maximum_exit_count_difference=1,
                                ambiguity_similarity_margin=ambiguity_similarity_margin,
                            )
                        )
    if len(result) != 243 or len({tuple(config.to_dict().items()) for config in result}) != 243:
        raise RuntimeError("GSE graph parameter grid is not exactly 243 unique configurations")
    return result


def rule_graph_parameter_grid(
    *,
    event_probability_threshold: float,
    maximum_uncertainty: float = 1.0,
) -> list[GSEGraphConfig]:
    """The same 3^5 structural grid with descriptor-free rule association.

    The 20-degree exit tolerance is the pre-existing Phase-3 heading matching
    contract. Descriptor, width and vertical thresholds are inert because the
    rule graph never reads those learned quantities.
    """

    return gse_graph_parameter_grid(
        event_probability_threshold=event_probability_threshold,
        maximum_uncertainty=maximum_uncertainty,
        descriptor_minimum_similarity=-1.0,
        exit_descriptor_minimum_similarity=-1.0,
        exit_heading_tolerance_deg=20.0,
        exit_width_log_tolerance=1.0,
        exit_vertical_profile_tolerance=1.0,
    )


def aggregate_gse_world_replays(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot aggregate an empty GSE replay set")
    node_tp = sum(int(result["topology_metrics"]["node_true_positive"]) for result in results)
    node_predicted = sum(int(result["topology_metrics"]["predicted_node_count"]) for result in results)
    node_teacher = sum(int(result["topology_metrics"]["teacher_node_count"]) for result in results)
    edge_tp = sum(int(result["topology_metrics"]["edge_true_positive"]) for result in results)
    edge_predicted = sum(int(result["topology_metrics"]["predicted_edge_count"]) for result in results)
    edge_teacher = sum(int(result["topology_metrics"]["teacher_edge_count"]) for result in results)
    merge_attempts = sum(int(result["association_metrics"]["merge_attempts"]) for result in results)
    correct_merges = sum(int(result["association_metrics"]["correct_merges"]) for result in results)
    false_merges = sum(int(result["association_metrics"]["false_merges"]) for result in results)
    matchable = sum(
        sum(
            bool(decision.get("evaluator_association_opportunity"))
            and bool(decision.get("evaluator_has_existing_true_node"))
            for decision in result["decision_trace"]
        )
        for result in results
    )
    accepted_revisits = correct_merges
    component_errors = np.asarray(
        [int(result["topology_metrics"]["connected_component_error"]) for result in results],
        dtype=np.int64,
    )
    cycle_errors = np.asarray(
        [int(result["topology_metrics"]["cycle_rank_error"]) for result in results],
        dtype=np.int64,
    )
    return {
        "world_seed_replays": len(results),
        "node": {**_prf(node_tp, node_predicted, node_teacher), "true_positive": node_tp, "predicted": node_predicted, "teacher": node_teacher},
        "edge": {**_prf(edge_tp, edge_predicted, edge_teacher), "true_positive": edge_tp, "predicted": edge_predicted, "teacher": edge_teacher},
        "association": {
            "merge_attempts": merge_attempts,
            "correct_merges": correct_merges,
            "false_merges": false_merges,
            "precision": float(correct_merges / merge_attempts) if merge_attempts else 0.0,
            "recall": float(accepted_revisits / matchable) if matchable else 0.0,
            "false_loop_merge_rate": float(false_merges / merge_attempts) if merge_attempts else 0.0,
            "matchable_revisits": matchable,
        },
        "invariants": {
            "connected_component_mean_signed_error": float(component_errors.mean()),
            "connected_component_mean_absolute_error": float(np.abs(component_errors).mean()),
            "connected_component_exact_fraction": float(np.mean(component_errors == 0)),
            "cycle_rank_mean_signed_error": float(cycle_errors.mean()),
            "cycle_rank_mean_absolute_error": float(np.abs(cycle_errors).mean()),
            "cycle_rank_exact_fraction": float(np.mean(cycle_errors == 0)),
        },
    }


def select_gse_graph_sweep(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select graph parameters only among association-safe validation rows."""

    feasible = [
        row
        for row in rows
        if float(row["aggregate"]["association"]["precision"]) >= 0.98
        and float(row["aggregate"]["association"]["false_loop_merge_rate"]) <= 0.01
        and int(row["aggregate"]["association"]["merge_attempts"]) > 0
    ]
    if not feasible:
        raise RuntimeError("no GSE graph configuration satisfies the validation association safety gate")
    return max(
        feasible,
        key=lambda row: (
            min(float(row["aggregate"]["node"]["f1"]), float(row["aggregate"]["edge"]["f1"])),
            (float(row["aggregate"]["node"]["f1"]) + float(row["aggregate"]["edge"]["f1"])) / 2.0,
            -float(row["aggregate"]["invariants"]["connected_component_mean_absolute_error"]),
            -float(row["aggregate"]["invariants"]["cycle_rank_mean_absolute_error"]),
            -int(row["grid_index"]),
        ),
    )


def select_baseline_graph_sweep(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Outcome-neutral validation selection for a graph baseline.

    Baselines are not discarded when unsafe; their association result remains
    evidence. Selection therefore optimizes graph fidelity with the same fixed
    lexicographic ordering but does not impose the GSE safety qualification.
    """

    if not rows:
        raise ValueError("cannot select an empty baseline graph sweep")
    return max(
        rows,
        key=lambda row: (
            min(float(row["aggregate"]["node"]["f1"]), float(row["aggregate"]["edge"]["f1"])),
            (float(row["aggregate"]["node"]["f1"]) + float(row["aggregate"]["edge"]["f1"])) / 2.0,
            -float(row["aggregate"]["invariants"]["connected_component_mean_absolute_error"]),
            -float(row["aggregate"]["invariants"]["cycle_rank_mean_absolute_error"]),
            -int(row["grid_index"]),
        ),
    )


def offline_topology_scientific_gate(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    minimum_f1_improvement: float = 0.05,
    maximum_mean_signed_invariant_error: float = 0.25,
) -> dict[str, Any]:
    """Apply the pre-registered validation-only offline graph decision.

    Node and edge gains are each measured against the stronger corresponding
    deployable baseline.  The learned method must also make non-vacuous,
    high-precision associations and avoid a systematic topology-invariant bias.
    """

    gse = methods["gse_learned_association"]["selected_aggregate"]
    baselines = (methods["exit_only_rule_graph"], methods["nonlearning_geometry_rule_graph"])
    strongest_node = max(float(item["selected_aggregate"]["node"]["f1"]) for item in baselines)
    strongest_edge = max(float(item["selected_aggregate"]["edge"]["f1"]) for item in baselines)
    node_gain = float(gse["node"]["f1"]) - strongest_node
    edge_gain = float(gse["edge"]["f1"]) - strongest_edge
    invariant = gse["invariants"]
    component_bias = abs(float(invariant["connected_component_mean_signed_error"]))
    cycle_bias = abs(float(invariant["cycle_rank_mean_signed_error"]))
    association_safe = bool(
        int(gse["association"]["merge_attempts"]) > 0
        and float(gse["association"]["precision"]) >= 0.98
        and float(gse["association"]["false_loop_merge_rate"]) <= 0.01
    )
    passed = bool(
        node_gain >= minimum_f1_improvement
        and edge_gain >= minimum_f1_improvement
        and component_bias <= maximum_mean_signed_invariant_error
        and cycle_bias <= maximum_mean_signed_invariant_error
        and association_safe
    )
    return {
        "passed": passed,
        "status": "PASS_GSE_OFFLINE_TOPOLOGY_GATE_V1" if passed else "FAIL_GSE_OFFLINE_TOPOLOGY_GATE_V1",
        "node_f1_gain_over_strongest_main_baseline": node_gain,
        "edge_f1_gain_over_strongest_main_baseline": edge_gain,
        "required_each_f1_gain": minimum_f1_improvement,
        "strongest_baseline_node_f1": strongest_node,
        "strongest_baseline_edge_f1": strongest_edge,
        "connected_component_mean_signed_error_absolute": component_bias,
        "cycle_rank_mean_signed_error_absolute": cycle_bias,
        "maximum_mean_signed_invariant_error": maximum_mean_signed_invariant_error,
        "association_safe": association_safe,
    }


__all__ = [
    "DirectedTraversal",
    "aggregate_gse_world_replays",
    "collapse_to_structural_graph",
    "directed_euler_order",
    "gse_graph_parameter_grid",
    "offline_topology_scientific_gate",
    "rule_graph_parameter_grid",
    "strict_predicted_node_mapping",
    "teacher_structural_graph",
    "replay_gse_world",
    "replay_typed_observation_world",
    "select_gse_graph_sweep",
    "select_baseline_graph_sweep",
]
