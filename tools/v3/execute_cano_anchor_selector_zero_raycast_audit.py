#!/usr/bin/env python3
"""Execute the approved five-topology anchor-selector audit without mesh or rays."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from cano_five_topology_support import build_parent, canonical_exports
from mtare_topo.data.cano_contract_pilot import (
    ANCHOR_CANDIDATE_STEP_M,
    ANCHORS_PER_PARENT,
    MINIMUM_ANCHOR_SPACING_M,
    PARENT_REGISTRY,
    anchor_selection_audit,
    canonical_document_hash,
    graph_family_metrics,
    select_canonical_anchors,
)
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_five_topology_cpu_contract_pilot_v2b_runner_paths_seed0"
)
SOURCE_HASHES = {
    "P01_straight_turn": {
        "graph": "ce61b6a51973e6bad3068e2a1540d30ed9507c7de5e94bd6083947cc469ac7dd",
        "splines": "d990a614e234af63e71419f5ee17131378f58eab39021aa3fbdfc6a7da7254eb",
    },
    "P02_branch_deadend": {
        "graph": "ce002aba331d616068d3445872bd09512c7e174a9999506b8163c341d7645409",
        "splines": "f7a0fd27db5422ef64154066094ec14fb9924edc29de286627bcb88961f274a1",
    },
    "P03_loop_bottleneck": {
        "graph": "25fb95660d2926da3fe6be3a0a10cb09a23c56a3c71700e9ecbad23511333966",
        "splines": "de1e7785fd70bff42df0cfd878b828ce0181fe46004f20e0668bde744cbc28b7",
    },
    "P04_chamber_multiexit": {
        "graph": "faacf075247a6b4a307062607c8b69a18d84bffe82a396a714c1bbbef51bb3c2",
        "splines": "4d1962848f061eb25b7611a348802b990dbd2d0f5864fdc8691b93893fd5c94e",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_parent(parent_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    world_dir = SOURCE_RUN / "artifacts/worlds" / parent_id
    graph_path = world_dir / "graph.json"
    spline_path = world_dir / "splines.json"
    observed = {"graph": _sha256(graph_path), "splines": _sha256(spline_path)}
    if observed != SOURCE_HASHES[parent_id]:
        raise RuntimeError(
            f"{parent_id}: sealed graph/spline identity mismatch: {observed}"
        )
    return load_json(graph_path), load_json(spline_path), {
        "source": "sealed_v2b_read_only",
        "graph_path": str(graph_path.relative_to(PROJECT_ROOT)),
        "splines_path": str(spline_path.relative_to(PROJECT_ROOT)),
        "hashes": observed,
    }


def _plot_parent(
    path: Path,
    parent_id: str,
    graph: dict[str, Any],
    splines: dict[str, Any],
    anchors: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for tunnel in splines["tunnels"]:
        points = np.asarray(tunnel["points"], dtype=np.float64)
        axes[0].plot(points[:, 0], points[:, 1], color="0.25", linewidth=1.4)
        axes[1].plot(points[:, 0], points[:, 2], color="0.25", linewidth=1.4)
    selected = np.asarray([item["axis_xyz_m"] for item in anchors], dtype=np.float64)
    event = np.asarray([item["priority"] == 0 for item in anchors], dtype=bool)
    for axis, first, second in ((axes[0], 0, 1), (axes[1], 0, 2)):
        axis.scatter(
            selected[~event, first],
            selected[~event, second],
            s=28,
            color="#1874b4",
            edgecolor="white",
            linewidth=0.35,
            label="balanced arc fill",
            zorder=3,
        )
        axis.scatter(
            selected[event, first],
            selected[event, second],
            s=85,
            marker="*",
            color="#d62728",
            edgecolor="black",
            linewidth=0.45,
            label="structural event anchor",
            zorder=4,
        )
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].set_title("complete X-Y spline and all anchors")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Z [m]")
    axes[1].set_title("complete X-Z spline and all anchors")
    axes[0].legend(loc="best")
    figure.suptitle(
        f"{parent_id} | 50 target-aware anchors | min spacing "
        f"{audit['minimum_pairwise_distance_m']:.3f} m | "
        f"per tunnel {audit['selected_per_tunnel']}"
    )
    figure.text(
        0.5,
        0.01,
        "Gate 0 zero-ray selector audit; graph/spline coordinates in Cano world frame. "
        "This figure proves sampling coverage only, not LiDAR, learning or navigation.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_summary(path: Path, metrics: list[dict[str, Any]]) -> None:
    parent_ids = [item["parent_id"] for item in metrics]
    spacing = [item["anchor_audit"]["minimum_pairwise_distance_m"] for item in metrics]
    spread = [item["anchor_audit"]["tunnel_count_spread"] for item in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(parent_ids, spacing, color="#1874b4")
    axes[0].axhline(MINIMUM_ANCHOR_SPACING_M, color="#d62728", linestyle="--", label="required 5 m")
    axes[0].set_ylabel("minimum pairwise distance [m]")
    axes[0].set_title("global spacing contract")
    axes[0].legend()
    axes[1].bar(parent_ids, spread, color="#2ca02c")
    axes[1].axhline(3, color="#d62728", linestyle="--", label="maximum spread 3")
    axes[1].set_ylabel("max selected-per-tunnel minus min")
    axes[1].set_title("branch-balance contract")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Five-topology target-cardinality anchor-selector audit")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def execute(run_dir: Path) -> dict[str, Any]:
    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    if source_state.get("state") != "FAILED":
        raise RuntimeError("the frozen v2b source run is not in sealed FAILED state")

    metrics: list[dict[str, Any]] = []
    input_manifest = []
    new_topology_constructions = 0
    for registry in PARENT_REGISTRY:
        parent_id = registry["parent_id"]
        if parent_id == "P05_slope_multiheight":
            network, _, generation = build_parent(parent_id, registry["topology_seed"])
            graph, splines = canonical_exports(network)
            if not generation["all_constructions_succeeded"]:
                raise RuntimeError("P05 graph-only construction failed")
            new_topology_constructions += 1
            p05_dir = run_dir / "artifacts/p05_graph_only"
            p05_dir.mkdir(parents=True, exist_ok=True)
            write_json(p05_dir / "graph.json", graph)
            write_json(p05_dir / "splines.json", splines)
            source = {
                "source": "one_frozen_template_graph_only_construction",
                "topology_seed": registry["topology_seed"],
                "generation": generation,
                "graph_path": str((p05_dir / "graph.json").relative_to(PROJECT_ROOT)),
                "splines_path": str((p05_dir / "splines.json").relative_to(PROJECT_ROOT)),
                "hashes": {
                    "graph": _sha256(p05_dir / "graph.json"),
                    "splines": _sha256(p05_dir / "splines.json"),
                },
            }
        else:
            graph, splines, source = _load_frozen_parent(parent_id)

        family = graph_family_metrics(parent_id, graph, splines)
        if not family["passed"]:
            raise RuntimeError(f"{parent_id}: frozen topology-family contract failed")
        anchors = select_canonical_anchors(graph, splines, None)
        replay = select_canonical_anchors(graph, splines, None)
        anchor_hash = canonical_document_hash({"anchors": anchors})
        replay_hash = canonical_document_hash({"anchors": replay})
        audit = anchor_selection_audit(graph, splines, anchors)
        deterministic_replay = anchor_hash == replay_hash
        if not audit["passed"] or not deterministic_replay:
            raise RuntimeError(
                f"{parent_id}: anchor audit failed: audit={audit}, replay={deterministic_replay}"
            )
        anchor_path = run_dir / "artifacts/anchors" / f"{parent_id}.json"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            anchor_path,
            {
                "parent_id": parent_id,
                "coordinate_frame": "cano_world",
                "sensor_origins_present": False,
                "anchors": anchors,
            },
        )
        record = {
            "parent_id": parent_id,
            "source": source,
            "graph_family": family,
            "anchor_audit": audit,
            "selector": {
                "method": "event-first target-aware round-robin interpolated arc fill",
                "candidate_step_m": ANCHOR_CANDIDATE_STEP_M,
                "target_count": ANCHORS_PER_PARENT,
                "minimum_spacing_m": MINIMUM_ANCHOR_SPACING_M,
                "anchor_sha256": anchor_hash,
                "replay_anchor_sha256": replay_hash,
                "deterministic_replay": deterministic_replay,
            },
        }
        write_json(run_dir / "metrics" / f"{parent_id}.json", record)
        _plot_parent(
            run_dir / "previews" / f"{parent_id}_complete_anchor_map.png",
            parent_id,
            graph,
            splines,
            anchors,
            audit,
        )
        metrics.append(record)
        input_manifest.append({"parent_id": parent_id, **source})

    _plot_summary(run_dir / "previews/five_topology_anchor_contract_summary.png", metrics)
    write_json(run_dir / "artifacts/input_manifest.json", {"parents": input_manifest})
    write_json(
        run_dir / "previews/provenance.json",
        {
            "question": "Can the corrected selector produce 50 event-preserving, branch-balanced, globally spaced anchors on every fixed topology?",
            "figures": [
                f"previews/{item['parent_id']}_complete_anchor_map.png" for item in metrics
            ]
            + ["previews/five_topology_anchor_contract_summary.png"],
            "coordinate_frame": "cano_world",
            "units": "metres",
            "selection": "All five parents and all 250 anchors are shown; no best-case selection.",
            "claim_boundary": "Sampling-contract evidence only; no mesh, LiDAR, formal data, learning, trajectory, online graph or navigation evidence.",
        },
    )
    summary = {
        "schema_version": "cano_anchor_selector_zero_raycast_audit_v1",
        "overall_status": "PASS_CANO_ANCHOR_SELECTOR_ZERO_RAYCAST_AUDIT",
        "scope": {
            "sealed_graph_spline_parents_reused": 4,
            "new_graph_only_topology_constructions": new_topology_constructions,
            "new_meshes": 0,
            "canonical_anchors": sum(item["anchor_audit"]["selected_anchor_count"] for item in metrics),
            "raycast_observations": 0,
            "rays": 0,
            "formal_dataset_worlds": 0,
            "formal_dataset_samples": 0,
            "training_samples": 0,
            "models": 0,
            "trajectories": 0,
            "topology_runtime_changes": 0,
            "mtare_changes": 0,
        },
        "all_parent_audits_passed": all(item["anchor_audit"]["passed"] for item in metrics),
        "all_replays_identical": all(item["selector"]["deterministic_replay"] for item in metrics),
        "minimum_pairwise_distance_across_parents_m": min(
            item["anchor_audit"]["minimum_pairwise_distance_m"] for item in metrics
        ),
        "maximum_tunnel_count_spread": max(
            item["anchor_audit"]["tunnel_count_spread"] for item in metrics
        ),
        "parents": [item["parent_id"] for item in metrics],
        "claim_boundary": "This PASS authorizes review of anchor sampling only. It does not pass the five-topology LiDAR pilot or authorize data generation/training.",
    }
    write_json(run_dir / "metrics/summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
