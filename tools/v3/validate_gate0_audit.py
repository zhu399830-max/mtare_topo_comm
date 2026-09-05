#!/usr/bin/env python3
"""Validate the machine-readable Gate-0 interface and benchmark audit.

This validator checks evidence structure and internal consistency.  A valid
audit may still conclude that Gate 0 is blocked; it must not convert missing
worlds, seed control, or metrics into a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVIDENCE_STATUSES = {"VERIFIED", "PARTIAL", "UNVERIFIED", "CONFLICT"}
RUN_CLASSIFICATIONS = {"VALID_CANDIDATE", "INVALID", "INSUFFICIENT_EVIDENCE", "MISSING"}


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_documents(
    interface: dict[str, Any],
    worlds: dict[str, Any],
    benchmark: dict[str, Any],
    baseline_matrix: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    _require(interface.get("schema_version") == "gate0_interface_contract_v1",
             "interface schema_version is not gate0_interface_contract_v1", errors)
    _require(interface.get("decision_status") in {"DRAFT_FOR_USER_REVIEW", "FROZEN"},
             "interface decision_status is invalid", errors)
    boundary = interface.get("system_boundary")
    _require(isinstance(boundary, dict), "interface system_boundary must be an object", errors)
    if isinstance(boundary, dict):
        _require(bool(boundary.get("retained_components")),
                 "interface retained_components must not be empty", errors)
        _require(bool(boundary.get("replaced_components")),
                 "interface replaced_components must not be empty", errors)

    interfaces = interface.get("interfaces")
    _require(isinstance(interfaces, list) and bool(interfaces),
             "interface interfaces must be a non-empty list", errors)
    interface_ids: set[str] = set()
    if isinstance(interfaces, list):
        required_fields = {
            "id", "direction", "topic", "message_type", "frame", "producer",
            "consumer", "required_for_replacement", "rate_hz", "status", "evidence",
        }
        for index, item in enumerate(interfaces):
            if not isinstance(item, dict):
                errors.append(f"interfaces[{index}] must be an object")
                continue
            missing = sorted(required_fields - set(item))
            _require(not missing, f"interfaces[{index}] missing fields: {missing}", errors)
            interface_id = item.get("id")
            _require(isinstance(interface_id, str) and bool(interface_id),
                     f"interfaces[{index}].id must be non-empty", errors)
            if isinstance(interface_id, str):
                _require(interface_id not in interface_ids,
                         f"duplicate interface id: {interface_id}", errors)
                interface_ids.add(interface_id)
            _require(item.get("status") in EVIDENCE_STATUSES,
                     f"interfaces[{index}].status is invalid", errors)
            _require(isinstance(item.get("evidence"), list) and bool(item.get("evidence")),
                     f"interfaces[{index}].evidence must be non-empty", errors)

    _require(worlds.get("schema_version") == "gate0_world_inventory_v1",
             "world inventory schema_version is invalid", errors)
    world_rows = worlds.get("worlds")
    _require(isinstance(world_rows, list) and bool(world_rows),
             "world inventory worlds must be a non-empty list", errors)
    world_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(world_rows, list):
        required_world_fields = {
            "id", "underground_relevance", "asset", "historical_exposure",
            "strict_test_clean", "eligibility", "evidence",
        }
        for index, item in enumerate(world_rows):
            if not isinstance(item, dict):
                errors.append(f"worlds[{index}] must be an object")
                continue
            missing = sorted(required_world_fields - set(item))
            _require(not missing, f"worlds[{index}] missing fields: {missing}", errors)
            world_id = item.get("id")
            _require(isinstance(world_id, str) and bool(world_id),
                     f"worlds[{index}].id must be non-empty", errors)
            if isinstance(world_id, str):
                _require(world_id not in world_by_id, f"duplicate world id: {world_id}", errors)
                world_by_id[world_id] = item
            _require(isinstance(item.get("historical_exposure"), list),
                     f"worlds[{index}].historical_exposure must be a list", errors)
            if item.get("strict_test_clean") is True:
                _require(not item.get("historical_exposure"),
                         f"world {world_id} claims clean strict-test status despite historical exposure", errors)

    _require(benchmark.get("schema_version") == "gate0_benchmark_proposal_v1",
             "benchmark schema_version is invalid", errors)
    development = benchmark.get("development_worlds")
    strict_test = benchmark.get("strict_test_worlds")
    _require(isinstance(development, list) and len(development) >= 3,
             "benchmark must propose at least three development worlds", errors)
    _require(isinstance(strict_test, list), "strict_test_worlds must be a list", errors)
    if isinstance(development, list):
        for world_id in development:
            _require(world_id in world_by_id, f"unknown development world: {world_id}", errors)
    if isinstance(strict_test, list):
        for world_id in strict_test:
            _require(world_id in world_by_id, f"unknown strict-test world: {world_id}", errors)
            if world_id in world_by_id:
                _require(world_by_id[world_id].get("strict_test_clean") is True,
                         f"strict-test world is not clean: {world_id}", errors)

    strict_status = benchmark.get("strict_test_status")
    _require(strict_status in {"READY", "BLOCKED_NO_CLEAN_WORLDS"},
             "benchmark strict_test_status is invalid", errors)
    if strict_status == "READY":
        _require(isinstance(strict_test, list) and len(strict_test) >= 2,
                 "READY benchmark requires at least two clean strict-test worlds", errors)
    else:
        required_slots = benchmark.get("required_new_strict_test_worlds")
        _require(isinstance(required_slots, int) and required_slots >= 2,
                 "blocked benchmark must require at least two new strict-test worlds", errors)
        warnings.append("Gate 0 remains blocked because two clean strict-test worlds do not exist")

    seed_control = benchmark.get("protocol", {}).get("seed_control", {})
    _require(seed_control.get("status") in EVIDENCE_STATUSES,
             "benchmark protocol.seed_control.status is invalid", errors)
    if seed_control.get("status") != "VERIFIED":
        warnings.append("Seed control is not verified; no run may be called a frozen statistical baseline")

    _require(baseline_matrix.get("schema_version") == "gate0_baseline_rerun_matrix_v1",
             "baseline matrix schema_version is invalid", errors)
    runs = baseline_matrix.get("historical_runs")
    _require(isinstance(runs, list) and bool(runs),
             "baseline historical_runs must be a non-empty list", errors)
    if isinstance(runs, list):
        for index, item in enumerate(runs):
            if not isinstance(item, dict):
                errors.append(f"historical_runs[{index}] must be an object")
                continue
            _require(item.get("classification") in RUN_CLASSIFICATIONS,
                     f"historical_runs[{index}].classification is invalid", errors)
            _require(bool(item.get("evidence")),
                     f"historical_runs[{index}].evidence must not be empty", errors)
            if item.get("final_benchmark_eligible") is True:
                _require(item.get("seed_status") == "VERIFIED",
                         f"historical run {item.get('run_id')} cannot be final-eligible without verified seed", errors)

    reruns = baseline_matrix.get("required_reruns")
    _require(isinstance(reruns, list) and bool(reruns),
             "baseline required_reruns must be a non-empty list", errors)
    if isinstance(reruns, list):
        for index, item in enumerate(reruns):
            _require(item.get("world") in world_by_id or item.get("world", "").startswith("strict_test_slot_"),
                     f"required_reruns[{index}] refers to an unknown world", errors)
            _require(item.get("authorization_status") == "NOT_AUTHORIZED",
                     f"required_reruns[{index}] must remain NOT_AUTHORIZED in this audit", errors)

    return {
        "schema_version": "gate0_audit_validation_v1",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "interfaces": len(interfaces) if isinstance(interfaces, list) else 0,
            "worlds": len(world_rows) if isinstance(world_rows, list) else 0,
            "development_worlds": len(development) if isinstance(development, list) else 0,
            "strict_test_worlds": len(strict_test) if isinstance(strict_test, list) else 0,
            "historical_runs": len(runs) if isinstance(runs, list) else 0,
            "required_reruns": len(reruns) if isinstance(reruns, list) else 0,
        },
        "gate_conclusion": "GATE_MIXED" if not errors else "AUDIT_INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, required=True)
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-matrix", type=Path, required=True)
    args = parser.parse_args()
    report = validate_documents(
        _load(args.interface),
        _load(args.worlds),
        _load(args.benchmark),
        _load(args.baseline_matrix),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
