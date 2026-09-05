#!/usr/bin/env python3
"""Build the frozen C13--C24 topology-only corrective candidate pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from execute_cano_100_topology_parent_candidate_audit_v1 import _pilot_identities
from execute_cano_100_topology_parent_candidate_audit_v2 import _build_candidate
from generate_cano_audited_bundle import _source_precheck
from mtare_topo.data.cano_topology_parent_audit import (
    build_arithmetic_candidate_registry,
    parent_metrics,
    select_corrective_train_validation_pairs,
)
from mtare_topo.governance import load_json, write_json


METHOD_ID = "aee_corrective_c13_c24_v2_bounded_parameter_resampling"
HISTORICAL_V2 = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
)


def _historical_identities() -> set[str]:
    manifest = load_json(HISTORICAL_V2 / "artifacts/candidate_manifest.json")
    identities = set(_pilot_identities())
    identities.update(
        str(item["metrics"]["canonical_parent_identity"])
        for item in manifest["candidates"]
        if item.get("candidate_valid") is True
    )
    return identities


def _registry(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    scope = proposal["frozen_candidate_scope"]
    return build_arithmetic_candidate_registry(
        scope["strata"],
        first_candidate_index=int(scope["first_candidate_index"]),
        candidates_per_stratum=int(scope["candidates_per_stratum"]),
        topology_seed_base=int(scope["topology_seed_base"]),
        geometry_seed_base=int(scope["reserved_geometry_seed_base"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    proposal = load_json(run_dir / "config/source_config")
    source_before = _source_precheck()
    excluded = _historical_identities()
    strata = proposal["frozen_candidate_scope"]["strata"]
    strata_by_id = {str(item["stratum_id"]): item for item in strata}
    registry = _registry(proposal)
    if len(registry) != 120 or len(strata_by_id) != 10:
        raise RuntimeError("corrective registry is not exactly 10 strata x 12 candidates")

    candidate_root = run_dir / "artifacts/candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in registry:
        candidate_id = str(candidate["candidate_id"])
        stratum = strata_by_id[str(candidate["stratum_id"])]
        print(f"CANDIDATE_START {candidate_id}", flush=True)
        primary = _build_candidate(stratum, candidate)
        record: dict[str, Any] = {
            **candidate,
            "dimensionality": str(stratum["dimensionality"]),
            "requested_grown_tunnels": int(stratum["grown_tunnels"]),
            "requested_connector_tunnels": int(stratum["connector_tunnels"]),
            "generation_succeeded": bool(primary["generation_succeeded"]),
            "operations": primary["operations"],
            "replay_identical": False,
            "candidate_valid": False,
            "failure_reasons": [],
        }
        if not primary["generation_succeeded"]:
            record["failure_reasons"].append("requested_tunnel_generation_failed")
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
                dimensionality=str(stratum["dimensionality"]),
                requested_grown=int(stratum["grown_tunnels"]),
                requested_connector=int(stratum["connector_tunnels"]),
            )
            record["replay_identical"] = replay_identical
            record["metrics"] = metrics
            if not replay_identical:
                record["failure_reasons"].append("same_seed_replay_mismatch")
            record["failure_reasons"].extend(
                key for key, passed in metrics["checks"].items() if not passed
            )
            if metrics["canonical_parent_identity"] in excluded:
                record["failure_reasons"].append("duplicates_historical_identity")
            record["candidate_valid"] = not record["failure_reasons"]
            destination = candidate_root / candidate_id
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "graph.json", primary["graph"])
            write_json(destination / "splines.json", primary["splines"])
        write_json(run_dir / "metrics" / f"{candidate_id}.json", record)
        results.append(record)
        print(
            f"CANDIDATE_END {candidate_id} valid={record['candidate_valid']} "
            f"reasons={record['failure_reasons']}",
            flush=True,
        )

    seen: set[str] = set()
    for record in sorted(results, key=lambda item: int(item["candidate_index"])):
        if not record["candidate_valid"]:
            continue
        identity = str(record["metrics"]["canonical_parent_identity"])
        if identity in seen:
            record["candidate_valid"] = False
            record["failure_reasons"].append("duplicate_corrective_candidate_identity")
            write_json(run_dir / "metrics" / f"{record['candidate_id']}.json", record)
        else:
            seen.add(identity)

    selected, selection_audit = select_corrective_train_validation_pairs(results)
    for item in selected:
        item["artifact_directory"] = f"artifacts/candidates/{item['candidate_id']}"
    source_after = _source_precheck()
    source_unchanged = source_before == source_after
    scope = {
        "candidate_topology_constructions": len(results),
        "same_seed_topology_replays_maximum": len(results),
        "selected_topology_parents": len(selected),
        "corrective_train_parents": sum(item["split"] == "corrective_train" for item in selected),
        "corrective_validation_parents": sum(
            item["split"] == "corrective_validation" for item in selected
        ),
        "meshes": 0,
        "lidar_observations": 0,
        "teacher_labels": 0,
        "training_samples": 0,
        "models": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "mtare_changes": 0,
    }
    passed = bool(selection_audit["passed"] and source_unchanged)
    status = (
        "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        if passed
        else "FAIL_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
    )
    write_json(run_dir / "artifacts/candidate_manifest.json", {"candidates": results})
    write_json(run_dir / "artifacts/selected_parent_manifest.json", {"parents": selected})
    write_json(
        run_dir / "artifacts/split_manifest.json",
        {
            "split_atom": "coordinate_bearing_topology_parent",
            "parents": selected,
        },
    )
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_corrective_topology_candidate_audit_v1",
            "overall_status": status,
            "method_id": METHOD_ID,
            "scope": scope,
            "selection_audit": selection_audit,
            "historical_identity_count": len(excluded),
            "source_unchanged": source_unchanged,
            "claim_boundary": (
                "Topology-only corrective train/validation parent selection; no mesh, "
                "sensor sample, teacher, training, model, graph replay or planner result."
            ),
        },
    )
    print(status, flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
