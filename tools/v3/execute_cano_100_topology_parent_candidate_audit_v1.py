#!/usr/bin/env python3
"""Execute the approved topology-only Cano 120-candidate/100-parent audit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from generate_cano_audited_bundle import (
    _normalize_result,
    _source_precheck,
    _spline_records,
    _stable_graph,
)
from mtare_topo.data.cano_topology_parent_audit import (
    batch_diversity_audit,
    canonical_parent_identity,
    parent_metrics,
    select_accepted_candidates,
    split_audit,
)
from mtare_topo.governance import load_json, write_json
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import (
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
    Tunnel,
    TunnelNetwork,
    TunnelNetworkParams,
)


PILOT_DIR = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_five_topology_cpu_contract_pilot_v3_selector_coverage_seed0/"
    "artifacts/worlds"
)
INTERNAL_TRIAL_BUDGET = 100
METHOD_ID = "v1_single_parameter_draw"


def _reset(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    Node.set_global_counter(0)
    Tunnel.counter = 0


def _grown_parameters(flat: bool) -> GrownTunnelGenerationParams:
    # Match the pinned Cano batch script's published random ranges. The upstream
    # flat flag only affects connectors, so the approved flat stratum contract
    # also requires zero vertical tendency/noise in each grown-tunnel parameter.
    GrownTunnelGenerationParams._random_distance_range = (100, 300)
    GrownTunnelGenerationParams._random_horizontal_tendency_range_deg = (-40, 40)
    GrownTunnelGenerationParams._random_horizontal_noise_range_deg = (-30, 30)
    GrownTunnelGenerationParams._random_min_segment_length_fraction_range = (0.05, 0.05)
    GrownTunnelGenerationParams._random_max_segment_length_fraction_range = (0.10, 0.10)
    parameters = GrownTunnelGenerationParams.random()
    if flat:
        parameters.vertical_tendency = 0.0
        parameters.vertical_noise = 0.0
    return parameters


def _parameter_record(parameters: GrownTunnelGenerationParams) -> dict[str, float]:
    return {
        "distance_m": float(parameters.distance),
        "horizontal_tendency_rad": float(parameters.horizontal_tendency),
        "vertical_tendency_rad": float(parameters.vertical_tendency),
        "horizontal_noise_rad": float(parameters.horizontal_noise),
        "vertical_noise_rad": float(parameters.vertical_noise),
        "minimum_segment_length_m": float(parameters.min_segment_length),
        "maximum_segment_length_m": float(parameters.max_segment_length),
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
    operations = []

    for index in range(int(stratum["grown_tunnels"])):
        parameters = _grown_parameters(flat)
        result = network.add_random_grown_tunnel(
            params=parameters,
            n_trials=INTERNAL_TRIAL_BUDGET,
        )
        success, tunnel, schema = _normalize_result(result)
        operations.append(
            {
                "operation": "grown",
                "index": index,
                "success": success,
                "return_schema": schema,
                "retained_tunnel_id": int(tunnel.tunnel_id) if tunnel is not None else None,
                "parameters": _parameter_record(parameters),
                "internal_trial_budget": INTERNAL_TRIAL_BUDGET,
            }
        )
        if not success:
            return {"generation_succeeded": False, "operations": operations}

    for index in range(int(stratum["connector_tunnels"])):
        parameters = ConnectorTunnelGenerationParams.random()
        if flat:
            parameters.node_position_vertical_noise = 0.0
        result = network.add_random_connector_tunnel(
            params=parameters,
            n_trials=INTERNAL_TRIAL_BUDGET,
        )
        success, tunnel, schema = _normalize_result(result)
        operations.append(
            {
                "operation": "connector",
                "index": index,
                "success": success,
                "return_schema": schema,
                "retained_tunnel_id": int(tunnel.tunnel_id) if tunnel is not None else None,
                "parameters": {
                    "segment_length_m": float(parameters.segment_length),
                    "horizontal_noise_m": float(parameters.node_position_horizontal_noise),
                    "vertical_noise_m": float(parameters.node_position_vertical_noise),
                },
                "internal_trial_budget": INTERNAL_TRIAL_BUDGET,
            }
        )
        if not success:
            return {"generation_succeeded": False, "operations": operations}

    graph, node_ids = _stable_graph(network)
    splines = _spline_records(network, node_ids)
    return {
        "generation_succeeded": True,
        "operations": operations,
        "graph": graph,
        "splines": splines,
    }


def _pilot_identities() -> set[str]:
    identities = set()
    for world in sorted(PILOT_DIR.glob("P*")):
        identities.add(
            canonical_parent_identity(
                load_json(world / "graph.json"), load_json(world / "splines.json")
            )
        )
    if len(identities) != 5:
        raise RuntimeError("five-topology prerequisite identities are incomplete")
    return identities


def _candidate_registry(
    proposal: dict[str, Any], stratum: dict[str, Any], stratum_index: int
) -> list[dict[str, Any]]:
    """Return an explicit registry or expand a frozen arithmetic seed registry."""

    explicit = stratum.get("candidate_registry")
    if explicit is not None:
        return list(explicit)
    scope = proposal["frozen_candidate_scope"]
    count = int(stratum["candidate_count"])
    topology_base = int(scope["topology_seed_base"])
    geometry_base = int(scope["reserved_geometry_seed_base"])
    return [
        {
            "candidate_id": f"{stratum['stratum_id']}_C{candidate_index:02d}",
            "topology_seed": topology_base + 100 * stratum_index + candidate_index,
            "reserved_geometry_seed": geometry_base + 100 * stratum_index + candidate_index,
        }
        for candidate_index in range(1, count + 1)
    ]


def _excluded_identities(proposal: dict[str, Any]) -> set[str]:
    identities = _pilot_identities()
    predecessor = proposal.get("predecessor_evidence")
    if predecessor:
        manifest_path = (
            PROJECT_ROOT
            / str(predecessor)
            / "artifacts/accepted_parent_manifest.json"
        ).resolve()
        if not manifest_path.is_relative_to(PROJECT_ROOT) or not manifest_path.is_file():
            raise RuntimeError("predecessor accepted-parent manifest is unavailable")
        manifest = load_json(manifest_path)
        identities.update(
            str(item["canonical_parent_identity"])
            for item in manifest.get("parents", [])
        )
    return identities


def _render_parent(
    graph: dict[str, Any],
    splines: dict[str, Any],
    destination: Path,
    *,
    parent_id: str,
    stratum_id: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for tunnel in splines["tunnels"]:
        points = np.asarray(tunnel["points"], dtype=np.float64)
        color = "#d95f02" if tunnel["type"] == "connector" else "#1b9e77"
        axes[0].plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
        axes[1].plot(points[:, 0], points[:, 2], color=color, linewidth=1.5)
    coordinates = np.asarray([node["xyz"] for node in graph["nodes"]], dtype=np.float64)
    degrees = np.asarray([node["degree"] for node in graph["nodes"]], dtype=np.float64)
    sizes = 10.0 + 12.0 * np.clip(degrees, 1, 5)
    axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=degrees, s=sizes, cmap="viridis", zorder=3)
    axes[1].scatter(coordinates[:, 0], coordinates[:, 2], c=degrees, s=sizes, cmap="viridis", zorder=3)
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Complete X-Y topology")
    axes[1].set(xlabel="x [m]", ylabel="z [m]", title="Complete X-Z topology")
    for axis in axes:
        axis.axis("equal")
        axis.grid(True, alpha=0.25)
    figure.suptitle(f"{parent_id} | {stratum_id} | split=train | full graph+spline")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    proposal = load_json(run_dir / "config/source_config")
    source_before = _source_precheck()
    excluded_identities = _excluded_identities(proposal)

    candidates_root = run_dir / "artifacts/candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "previews/train_only_maps").mkdir(parents=True, exist_ok=True)
    candidate_results: list[dict[str, Any]] = []

    for stratum_index, stratum in enumerate(
        proposal["frozen_candidate_scope"]["strata"], start=1
    ):
        for candidate in _candidate_registry(proposal, stratum, stratum_index):
            candidate_id = candidate["candidate_id"]
            print(f"CANDIDATE_START {candidate_id}", flush=True)
            primary = _build_candidate(stratum, candidate)
            result: dict[str, Any] = {
                "candidate_id": candidate_id,
                "stratum_id": stratum["stratum_id"],
                "dimensionality": stratum["dimensionality"],
                "requested_grown_tunnels": stratum["grown_tunnels"],
                "requested_connector_tunnels": stratum["connector_tunnels"],
                "topology_seed": candidate["topology_seed"],
                "reserved_geometry_seed": candidate["reserved_geometry_seed"],
                "generation_succeeded": primary["generation_succeeded"],
                "operations": primary["operations"],
                "replay_identical": False,
                "candidate_valid": False,
                "failure_reasons": [],
            }
            if not primary["generation_succeeded"]:
                result["failure_reasons"].append("requested_tunnel_generation_failed")
            else:
                replay = _build_candidate(stratum, candidate)
                replay_identical = bool(
                    replay["generation_succeeded"]
                    and primary["graph"] == replay["graph"]
                    and primary["splines"] == replay["splines"]
                )
                metrics = parent_metrics(
                    primary["graph"],
                    primary["splines"],
                    dimensionality=stratum["dimensionality"],
                    requested_grown=int(stratum["grown_tunnels"]),
                    requested_connector=int(stratum["connector_tunnels"]),
                )
                result["replay_identical"] = replay_identical
                result["metrics"] = metrics
                if not replay_identical:
                    result["failure_reasons"].append("same_seed_replay_mismatch")
                if not metrics["passed_geometry_free_contract"]:
                    result["failure_reasons"].extend(
                        key for key, passed in metrics["checks"].items() if not passed
                    )
                if metrics["canonical_parent_identity"] in excluded_identities:
                    result["failure_reasons"].append("duplicates_excluded_predecessor_identity")
                result["candidate_valid"] = not result["failure_reasons"]
                candidate_dir = candidates_root / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                write_json(candidate_dir / "graph.json", primary["graph"])
                write_json(candidate_dir / "splines.json", primary["splines"])
            write_json(run_dir / "metrics" / f"{candidate_id}.json", result)
            candidate_results.append(result)
            print(
                f"CANDIDATE_END {candidate_id} valid={result['candidate_valid']} "
                f"reasons={result['failure_reasons']}",
                flush=True,
            )

    seen: set[str] = set()
    for result in sorted(candidate_results, key=lambda item: str(item["candidate_id"])):
        if not result["candidate_valid"]:
            continue
        identity = result["metrics"]["canonical_parent_identity"]
        if identity in seen:
            result["candidate_valid"] = False
            result["failure_reasons"].append("duplicate_candidate_canonical_identity")
            write_json(run_dir / "metrics" / f"{result['candidate_id']}.json", result)
        else:
            seen.add(identity)

    accepted, strata_summary = select_accepted_candidates(candidate_results)
    split_metrics = split_audit(accepted)
    diversity = batch_diversity_audit(accepted, strata_summary)
    split_pass = split_metrics == {
        "counts": {"train": 80, "validation": 10, "development_test": 10},
        "parent_and_identity_sets_pairwise_disjoint": True,
        "parent_id_count": 100,
        "canonical_identity_count": 100,
    }

    for item in accepted:
        item["artifact_directory"] = (
            f"artifacts/candidates/{item['candidate_id']}"
        )
    write_json(run_dir / "artifacts/candidate_manifest.json", {"candidates": candidate_results})
    write_json(run_dir / "artifacts/accepted_parent_manifest.json", {"parents": accepted})
    write_json(run_dir / "artifacts/split_manifest.json", {
        "split_atom": "topology_parent",
        "parents": [
            {key: item[key] for key in ("parent_id", "stratum_id", "split", "topology_seed", "reserved_geometry_seed", "canonical_parent_identity")}
            for item in accepted
        ],
    })

    preview_ids = []
    for stratum_id in sorted(strata_summary):
        first_train = next(
            (
                item
                for item in accepted
                if item["stratum_id"] == stratum_id
                and item["accepted_rank_in_stratum"] == 1
            ),
            None,
        )
        if first_train is None:
            continue
        candidate_id = first_train["candidate_id"]
        graph = load_json(candidates_root / candidate_id / "graph.json")
        splines = load_json(candidates_root / candidate_id / "splines.json")
        _render_parent(
            graph,
            splines,
            run_dir / "previews/train_only_maps" / f"{stratum_id}.png",
            parent_id=candidate_id,
            stratum_id=stratum_id,
        )
        preview_ids.append(candidate_id)
    write_json(
        run_dir / "previews/provenance.json",
        {
            "split": "train_only",
            "selection": "accepted rank 1 in each frozen stratum; no best-case selection",
            "parent_ids": preview_ids,
            "units": "m",
            "method": "complete graph+spline X-Y/X-Z visualization",
            "claim": "Supports or contradicts structural coverage of the frozen topology strata only.",
        },
    )

    source_after = _source_precheck()
    source_unchanged = source_before == source_after
    scope = {
        "candidate_topology_constructions": 120,
        "same_seed_topology_replays_attempted": sum(result["generation_succeeded"] for result in candidate_results),
        "retained_topology_parents": len(accepted),
        "train_parents": split_metrics["counts"]["train"],
        "validation_parents": split_metrics["counts"]["validation"],
        "development_test_parents": split_metrics["counts"]["development_test"],
        "meshes": 0,
        "anchors": 0,
        "lidar_observations": 0,
        "rays": 0,
        "teacher_labels": 0,
        "formal_dataset_samples": 0,
        "training_samples": 0,
        "models": 0,
        "trajectories": 0,
        "mtare_changes": 0,
    }
    overall_pass = bool(
        len(candidate_results) == 120
        and diversity["passed"]
        and split_pass
        and len(preview_ids) == 10
        and source_unchanged
    )
    summary = {
        "schema_version": "cano_100_topology_parent_candidate_audit_summary_v1",
        "method_id": METHOD_ID,
        "overall_status": (
            "PASS_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
            if overall_pass
            else "FAIL_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
        ),
        "scope": scope,
        "strata": strata_summary,
        "split_audit": split_metrics,
        "diversity_audit": diversity,
        "candidate_valid_count": sum(result["candidate_valid"] for result in candidate_results),
        "candidate_invalid_count": sum(not result["candidate_valid"] for result in candidate_results),
        "invalid_candidates": [
            {"candidate_id": result["candidate_id"], "failure_reasons": result["failure_reasons"]}
            for result in candidate_results
            if not result["candidate_valid"]
        ],
        "train_only_visual_parent_ids": preview_ids,
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_unchanged,
        "flat_implementation_note": "Cano TunnelNetworkParams.flat only zeros connector vertical noise; the adapter additionally zeros grown vertical tendency/noise to realize the approved flat strata without modifying upstream source.",
        "claim_boundary": "Topology graph/spline source and split audit only; no mesh, LiDAR, label, formal dataset, model, online graph or navigation claim.",
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
