#!/usr/bin/env python3
"""Execute approved Cano topology audit V2 with bounded parameter resampling."""

from __future__ import annotations

from typing import Any

import numpy as np

import execute_cano_100_topology_parent_candidate_audit_v1 as audit
from execute_cano_100_topology_parent_candidate_audit_v1 import (
    _grown_parameters,
    _normalize_result,
    _parameter_record,
    _reset,
    _spline_records,
    _stable_graph,
)
from subt_proc_gen.tunnel import (
    ConnectorTunnelGenerationParams,
    TunnelNetwork,
    TunnelNetworkParams,
)


PARAMETER_DRAW_BUDGET = 20
INTERNAL_TRIAL_BUDGET = 100
METHOD_ID = "v2_bounded_parameter_resampling_exact_cycle_rank"


def _connector_parameter_record(parameters: ConnectorTunnelGenerationParams) -> dict[str, float]:
    return {
        "segment_length_m": float(parameters.segment_length),
        "horizontal_noise_m": float(parameters.node_position_horizontal_noise),
        "vertical_noise_m": float(parameters.node_position_vertical_noise),
    }


def _attempt_requested_tunnel(
    network: TunnelNetwork,
    *,
    operation: str,
    index: int,
    flat: bool,
) -> dict[str, Any]:
    draws: list[dict[str, Any]] = []
    selected_tunnel = None
    selected_draw_index = None
    for draw_index in range(1, PARAMETER_DRAW_BUDGET + 1):
        if operation == "grown":
            parameters = _grown_parameters(flat)
            result = network.add_random_grown_tunnel(
                params=parameters,
                n_trials=INTERNAL_TRIAL_BUDGET,
            )
            parameter_values = _parameter_record(parameters)
        elif operation == "connector":
            parameters = ConnectorTunnelGenerationParams.random()
            if flat:
                parameters.node_position_vertical_noise = 0.0
            result = network.add_random_connector_tunnel(
                params=parameters,
                n_trials=INTERNAL_TRIAL_BUDGET,
            )
            parameter_values = _connector_parameter_record(parameters)
        else:
            raise ValueError(f"unsupported tunnel operation: {operation}")
        success, tunnel, schema = _normalize_result(result)
        draws.append(
            {
                "draw_index": draw_index,
                "success": success,
                "return_schema": schema,
                "returned_tunnel_id": int(tunnel.tunnel_id) if tunnel is not None else None,
                "parameters": parameter_values,
                "internal_trial_budget": INTERNAL_TRIAL_BUDGET,
            }
        )
        if success:
            selected_tunnel = tunnel
            selected_draw_index = draw_index
            break
    return {
        "operation": operation,
        "index": index,
        "success": selected_tunnel is not None,
        "selected_draw_index": selected_draw_index,
        "retained_tunnel_id": (
            int(selected_tunnel.tunnel_id) if selected_tunnel is not None else None
        ),
        "parameter_draw_budget": PARAMETER_DRAW_BUDGET,
        "parameter_draw_count": len(draws),
        "draws": draws,
    }


def _build_candidate(stratum: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    seed = int(candidate["topology_seed"])
    _reset(seed)
    flat = stratum["dimensionality"] == "flat"
    network_parameters = TunnelNetworkParams.from_defaults()
    network_parameters.collision_distance = 10.0
    network_parameters.min_distance_between_intersections = 30.0
    network_parameters.min_intersection_angle = np.deg2rad(30.0)
    network_parameters.max_inclination = np.deg2rad(30.0)
    network_parameters.flat = flat
    network = TunnelNetwork(params=network_parameters)
    operations: list[dict[str, Any]] = []

    for operation, requested_count in (
        ("grown", int(stratum["grown_tunnels"])),
        ("connector", int(stratum["connector_tunnels"])),
    ):
        for index in range(requested_count):
            outcome = _attempt_requested_tunnel(
                network,
                operation=operation,
                index=index,
                flat=flat,
            )
            operations.append(outcome)
            if not outcome["success"]:
                return {"generation_succeeded": False, "operations": operations}

    graph, node_ids = _stable_graph(network)
    splines = _spline_records(network, node_ids)
    return {
        "generation_succeeded": True,
        "operations": operations,
        "graph": graph,
        "splines": splines,
    }


def main() -> int:
    audit._build_candidate = _build_candidate
    audit.METHOD_ID = METHOD_ID
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
