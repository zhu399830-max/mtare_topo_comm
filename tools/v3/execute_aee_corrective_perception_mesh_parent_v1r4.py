#!/usr/bin/env python3
"""Materialize one sealed parent with exact memory-bounded Cano containment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
import execute_cano_100_parent_perception_mesh_m1r as m1r
from _bootstrap import PROJECT_ROOT
from execute_aee_corrective_perception_mesh_v1 import SOURCE_RUN, _parents, _strata
from generate_cano_audited_bundle import _source_precheck
from mtare_topo.data.cano_memory_bounded_geometry import (
    DEFAULT_SCRATCH_LIMIT_BYTES,
    patched_cano_points_inside,
)
from mtare_topo.governance import write_json


def execute_parent(run_dir: Path, parent_index: int, expected_parent_id: str) -> dict:
    started = time.monotonic()
    parents = _parents()
    if parent_index < 0 or parent_index >= len(parents):
        raise RuntimeError("parent index outside frozen twenty-parent manifest")
    parent = parents[parent_index]
    if parent["parent_id"] != expected_parent_id:
        raise RuntimeError("parent index/id mismatch")
    meshes_root = run_dir / "artifacts/meshes"
    previews_root = run_dir / "previews/train_complete_maps"
    receipts_root = run_dir / "metrics/worker_receipts"
    if not meshes_root.is_dir() or not previews_root.is_dir() or not receipts_root.is_dir():
        raise RuntimeError("batch runner did not create fresh worker roots")
    root = meshes_root / parent["parent_id"] / "primary"
    metric_path = run_dir / "metrics" / f"{parent['parent_id']}.json"
    preview_path = previews_root / f"{parent['parent_id']}_complete_xy_xz.png"
    receipt_path = receipts_root / f"{parent_index:02d}_{parent['parent_id']}.json"
    if root.exists() or metric_path.exists() or preview_path.exists() or receipt_path.exists():
        raise RuntimeError("parent worker is one-shot and refuses partial or prior outputs")

    source_before = _source_precheck()
    strata = _strata()
    m0.SOURCE_V2 = SOURCE_RUN
    with patched_cano_points_inside():
        primary, _, graph, splines = m0._materialize(
            parent, strata[parent["source_stratum_id"]], root, role="primary"
        )
    primary, vertices = m1r._sanitize_materialization(root, primary, graph, splines)
    memory_contract = {
        "implementation": "exact_query_row_chunking_v1",
        "scratch_limit_bytes": DEFAULT_SCRATCH_LIMIT_BYTES,
        "patched_operations": [
            "points_inside_of_tunnel_section",
            "ids_points_inside_ptcl_sphere",
        ],
        "nearest_reference_order_unchanged": True,
        "argmin_first_tie_unchanged": True,
        "geometry_thresholds_changed": False,
    }
    record = {
        "parent_id": parent["parent_id"],
        "parent_index": parent_index,
        "split": parent["split"],
        "source_parent": parent,
        "primary": primary,
        "memory_bounded_geometry": memory_contract,
        "immutable_asset": {
            "mesh_sha256": primary["mesh_sha256"],
            "generated_exactly_once_in_fresh_process": True,
            "cross_run_obj_hash_equality_required": False,
            "remeshing_substitution_forbidden": True,
        },
    }
    preview_created = False
    if parent["split"] == "corrective_train":
        m0._render_complete_train_map(
            parent,
            vertices,
            graph,
            splines,
            preview_path,
            stage_label="AEE CORRECTIVE MESH V1R4",
        )
        preview_created = True
    source_after = _source_precheck()
    if source_before != source_after:
        raise RuntimeError("fixed Cano source changed within parent worker")
    write_json(metric_path, record)
    receipt = {
        "schema_version": "aee_corrective_perception_mesh_parent_receipt_v1r4",
        "parent_index": parent_index,
        "parent_id": parent["parent_id"],
        "split": parent["split"],
        "mesh_sha256": primary["mesh_sha256"],
        "identity_checks_pass": all(primary["identity_checks"].values()),
        "mesh_audit_pass": primary["mesh_audit"]["passed"] is True,
        "sanitation_pass": primary["sanitation"]["passed"] is True,
        "memory_bounded_geometry": memory_contract,
        "preview_created": preview_created,
        "source_unchanged": True,
        "duration_seconds": time.monotonic() - started,
    }
    if not (
        receipt["identity_checks_pass"]
        and receipt["mesh_audit_pass"]
        and receipt["sanitation_pass"]
        and preview_created == (parent["split"] == "corrective_train")
    ):
        raise RuntimeError(f"parent qualification failed: {receipt}")
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--parent-index", required=True, type=int)
    parser.add_argument("--expected-parent-id", required=True)
    args = parser.parse_args()
    result = execute_parent(
        args.run_dir.resolve(), args.parent_index, args.expected_parent_id
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
