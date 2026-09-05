#!/usr/bin/env python3
"""Execute selector-v2 full-spline coverage audit without mesh or raycasting."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import PROJECT_ROOT
from cano_five_topology_support import build_parent, canonical_exports
from execute_cano_anchor_selector_zero_raycast_audit import (
    SOURCE_RUN,
    _load_frozen_parent,
    _plot_parent,
    _sha256,
)
from mtare_topo.data.cano_contract_pilot import (
    ANCHOR_CANDIDATE_STEP_M,
    ANCHORS_PER_PARENT,
    MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M,
    MINIMUM_ANCHOR_SPACING_M,
    PARENT_REGISTRY,
    anchor_selection_audit,
    canonical_document_hash,
    graph_family_metrics,
    select_canonical_anchors,
)
from mtare_topo.governance import load_json, write_json


def _plot_coverage_summary(path: Path, records: list[dict[str, Any]]) -> None:
    parent_ids = [item["parent_id"] for item in records]
    spacing = [item["anchor_audit"]["minimum_pairwise_distance_m"] for item in records]
    coverage = [
        item["anchor_audit"]["maximum_same_tunnel_or_shared_event_coverage_radius_m"]
        for item in records
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(parent_ids, spacing, color="#1874b4")
    axes[0].axhline(
        MINIMUM_ANCHOR_SPACING_M,
        color="#d62728",
        linestyle="--",
        label="minimum allowed 5 m",
    )
    axes[0].set_ylabel("minimum pairwise distance [m]")
    axes[0].set_title("global anchor-spacing contract")
    axes[0].legend()
    axes[1].bar(parent_ids, coverage, color="#2ca02c")
    axes[1].axhline(
        MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M,
        color="#d62728",
        linestyle="--",
        label="maximum allowed 7.5 m",
    )
    axes[1].set_ylabel("maximum full-spline coverage radius [m]")
    axes[1].set_title("same-tunnel or shared-event coverage contract")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Selector v2: arc-length quota and full-spline coverage audit")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def execute(run_dir: Path) -> dict[str, Any]:
    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    if source_state.get("state") != "FAILED":
        raise RuntimeError("the v2b graph/spline source is not sealed FAILED evidence")

    records: list[dict[str, Any]] = []
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
            raise RuntimeError(f"{parent_id}: topology-family contract failed")
        anchors = select_canonical_anchors(graph, splines, None)
        replay = select_canonical_anchors(graph, splines, None)
        anchor_hash = canonical_document_hash({"anchors": anchors})
        replay_hash = canonical_document_hash({"anchors": replay})
        deterministic_replay = anchor_hash == replay_hash
        audit = anchor_selection_audit(graph, splines, anchors)
        if not audit["passed"] or not deterministic_replay:
            raise RuntimeError(
                f"{parent_id}: selector-v2 audit failed: {audit}, replay={deterministic_replay}"
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
                "method": "event-preserving arc-length fill quotas with MILP spacing and full-spline coverage constraints",
                "candidate_step_m": ANCHOR_CANDIDATE_STEP_M,
                "target_count": ANCHORS_PER_PARENT,
                "minimum_spacing_m": MINIMUM_ANCHOR_SPACING_M,
                "maximum_coverage_radius_m": MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M,
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
        records.append(record)
        input_manifest.append({"parent_id": parent_id, **source})

    _plot_coverage_summary(
        run_dir / "previews/five_topology_spacing_coverage_summary.png", records
    )
    write_json(run_dir / "artifacts/input_manifest.json", {"parents": input_manifest})
    write_json(
        run_dir / "previews/provenance.json",
        {
            "question": "Does selector v2 cover every complete spline while retaining exact count, events and global spacing?",
            "figures": [
                f"previews/{item['parent_id']}_complete_anchor_map.png" for item in records
            ]
            + ["previews/five_topology_spacing_coverage_summary.png"],
            "coordinate_frame": "cano_world",
            "units": "metres",
            "selection": "All five parents and all 250 anchors shown; no best-case selection.",
            "claim_boundary": "Anchor-sampling evidence only; no mesh, LiDAR, dataset, model, trajectory, online graph or navigation evidence.",
        },
    )
    summary = {
        "schema_version": "cano_anchor_selector_coverage_audit_v2",
        "overall_status": "PASS_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2",
        "scope": {
            "sealed_graph_spline_parents_reused": 4,
            "new_graph_only_topology_constructions": new_topology_constructions,
            "new_meshes": 0,
            "canonical_anchors": sum(item["anchor_audit"]["selected_anchor_count"] for item in records),
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
        "all_parent_audits_passed": all(item["anchor_audit"]["passed"] for item in records),
        "all_arc_length_fill_quotas_match": all(
            item["anchor_audit"]["fill_quotas_match"] for item in records
        ),
        "all_replays_identical": all(item["selector"]["deterministic_replay"] for item in records),
        "minimum_pairwise_distance_across_parents_m": min(
            item["anchor_audit"]["minimum_pairwise_distance_m"] for item in records
        ),
        "maximum_full_spline_coverage_radius_across_parents_m": max(
            item["anchor_audit"]["maximum_same_tunnel_or_shared_event_coverage_radius_m"]
            for item in records
        ),
        "parents": [item["parent_id"] for item in records],
        "claim_boundary": "This PASS validates selector-v2 sampling only and requires visual user review before any separately approved LiDAR rerun.",
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
