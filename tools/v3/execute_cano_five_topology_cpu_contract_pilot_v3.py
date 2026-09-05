#!/usr/bin/env python3
"""Execute five-topology CPU LiDAR v3 and enforce selector-v2 coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

from execute_cano_five_topology_cpu_contract_pilot import execute as execute_legacy_pilot
from mtare_topo.data.cano_contract_pilot import PARENT_REGISTRY, anchor_selection_audit
from mtare_topo.governance import load_json, write_json


SELECTOR_AUDIT_RUN = Path(
    "results/gate0_baseline/"
    "gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_generated_anchors(run_dir: Path) -> dict[str, Any]:
    parent_audits: dict[str, Any] = {}
    for registry in PARENT_REGISTRY:
        parent_id = registry["parent_id"]
        world_dir = run_dir / "artifacts/worlds" / parent_id
        graph = load_json(world_dir / "graph.json")
        splines = load_json(world_dir / "splines.json")
        anchor_document = load_json(world_dir / "anchors.json")
        audit = anchor_selection_audit(graph, splines, anchor_document["anchors"])
        if not audit["passed"]:
            raise RuntimeError(f"{parent_id}: selector-v2 post-generation audit failed: {audit}")
        parent_audits[parent_id] = audit

        metric_path = run_dir / "metrics" / f"{parent_id}.json"
        metric_document = load_json(metric_path)
        metric_document["metrics"]["selector_v2_audit"] = audit
        write_json(metric_path, metric_document)

    return {
        "method": (
            "structural events plus exact arc-length fill quotas, 0.5 m candidate "
            "lattice, global spacing >=5 m and full-spline same-tunnel/shared-event "
            "coverage radius <=7.5 m"
        ),
        "parents_passed": sum(item["passed"] for item in parent_audits.values()),
        "all_arc_length_fill_quotas_match": all(
            item["fill_quotas_match"] for item in parent_audits.values()
        ),
        "minimum_pairwise_distance_across_parents_m": min(
            item["minimum_pairwise_distance_m"] for item in parent_audits.values()
        ),
        "maximum_full_spline_coverage_radius_across_parents_m": max(
            item["maximum_same_tunnel_or_shared_event_coverage_radius_m"]
            for item in parent_audits.values()
        ),
        "parents": parent_audits,
    }


def execute(run_dir: Path) -> dict[str, Any]:
    summary = execute_legacy_pilot(run_dir)
    selector_audit = _audit_generated_anchors(run_dir)
    if (
        selector_audit["parents_passed"] != 5
        or not selector_audit["all_arc_length_fill_quotas_match"]
        or selector_audit["minimum_pairwise_distance_across_parents_m"] < 5.0
        or selector_audit["maximum_full_spline_coverage_radius_across_parents_m"] > 7.5
    ):
        raise RuntimeError(f"aggregate selector-v2 contract failed: {selector_audit}")

    selector_summary = load_json(SELECTOR_AUDIT_RUN / "metrics/summary.json")
    summary["schema_version"] = "cano_five_topology_cpu_contract_pilot_summary_v3"
    summary["selector_v2_audit"] = selector_audit
    summary["selector_v2_prerequisite"] = {
        "path": str(SELECTOR_AUDIT_RUN),
        "overall_status": selector_summary["overall_status"],
        "summary_sha256": _sha256(SELECTOR_AUDIT_RUN / "metrics/summary.json"),
        "run_state_sha256": _sha256(SELECTOR_AUDIT_RUN / "RUN_STATE.json"),
    }
    summary["claim_boundary"].insert(
        0,
        "Selector v2 is enforced here only as a static LiDAR sampling contract; it is not a learned structural representation.",
    )

    by_parent = {item["parent_id"]: item for item in summary["parents"]}
    for parent_id, audit in selector_audit["parents"].items():
        by_parent[parent_id]["selector_v2_audit"] = audit
    write_json(run_dir / "metrics/summary.json", summary)

    provenance_path = run_dir / "previews/provenance.json"
    provenance = load_json(provenance_path)
    provenance["selector_v2"] = (
        "Every displayed anchor passed exact count, structural-event coverage, "
        "arc-length fill quota, >=5 m global spacing and <=7.5 m full-spline coverage."
    )
    provenance["limitations"] = summary["claim_boundary"]
    write_json(provenance_path, provenance)
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
