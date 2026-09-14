#!/usr/bin/env python3
"""Freeze an identity-only card/spec; never select rows or execute an experiment."""
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance_surface_selection import (
    CARD_ID, INPUT_PATHS, INVENTORY, PARENTS, POLICY, SCHEMA, SEAL_SHA256,
    digest, validate_surface_selection_card,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(root=PROJECT_ROOT):
    root = Path(root).resolve(strict=True)
    card_rel = "configs/v3/gate3/data_cards/gse_surface_identity_selection_v1.json"
    spec_rel = "configs/v3/gate3/gse_surface_identity_selection_v1.json"
    if any((root / p).exists() for p in (card_rel, spec_rel)):
        raise FileExistsError("card/spec immutable: no overwrite or silent refreeze")
    seal_path = root / INVENTORY / "artifacts/evidence_sha256.txt"
    if sha(seal_path) != SEAL_SHA256:
        raise ValueError("historical identity inventory seal drift")
    selected = {}
    for line in seal_path.read_text().splitlines():
        h, p = line.split("  ", 1)
        if p in INPUT_PATHS:
            if p in selected:
                raise ValueError("duplicate sealed input identity")
            selected[p] = h
    if sorted(selected) != INPUT_PATHS:
        raise ValueError("source seal does not contain exact72 required identity files")
    # Only hashes of the 72 explicitly allowed sealed metadata files are read.
    for p, h in selected.items():
        if sha(root / p) != h:
            raise ValueError("sealed metadata file drift: " + p)
    scope = {"parent_ids": PARENTS, "input_files_sha256": selected,
        "source_seal": {"path": INVENTORY + "/artifacts/evidence_sha256.txt", "sha256": SEAL_SHA256},
        "policy": POLICY, "expected_observations": {"fit": 2880, "calibration": 240, "development": 240},
        "unknown_counts": {"unique_source_frames": None, "structure_entities": None, "labels": None, "duration_s": None},
        "time_basis": "sealed decision order and route arc; acquisition clock/history spacing unknown",
        "resources": {"wall_time_s": 120, "host_ram_bytes": 1073741824, "gpu_bytes": 0, "output_bytes": 33554432},
        "forbidden_payloads": ["scans", "poses", "mesh", "construction", "teacher", "models", "C08-C10", "benchmark"]}
    approval = {"status": "APPROVED", "approved_by": "user-standing-scope-authorization",
        "approved_at": "2026-09-07", "authorized_operations": ["audit"], "authorized_gates": [3],
        "scope_sha256": digest(scope),
        "scope": "One selection of16physicaledges per70C01-C07parent,one causal5frame observation peredge and3variants;read exact72sealedidentityfiles only;no original arrays/labels/model/training.",
        "confirmation_reference": "User2026-09-07 PLEASE IMPLEMENT THIS PLAN:GSE-Graph surface relation execution, followed by 设置一个目标一直执行. The approved plan explicitly supersedes21decision/48nomination and authorizes exact manifest preparation under standing scope-bound authority. No invented additional approval or training authorization."}
    card = {"schema_version": SCHEMA, "card_id": CARD_ID, "operation": "audit",
        "purpose": "Freeze actual single-observation identity manifest, not structural labels or continuous trajectories.",
        "source_provenance": "Pinned completed20260906 inventory seal;only70per-parent intervals,parent_population,parent_split JSON. Source metadata has been validated;original P1a/P1b payload is not reopened.",
        "limitations": "3360 planned variant observations represent1120edge units/70parents,not independent places. C07 historically exposed. No physical continuity,duration,history spacing,labels or training eligibility inferred.",
        "scope": scope, "scope_sha256": digest(scope), "approval": approval,
        "scientific_gate_pass": False, "training_eligibility": False}
    report = validate_surface_selection_card(card)
    if not report.passed:
        raise ValueError("card invalid: " + "; ".join(report.errors))
    sources = [card_rel, "docs/GSE_SURFACE_RELATION_EXECUTION_V1.md", "tools/v3/_bootstrap.py",
        "tools/v3/freeze_gse_surface_selection_v1.py", "tools/v3/run_gse_surface_selection_v1.py",
        "src/mtare_topo/__init__.py", "src/mtare_topo/data/__init__.py",
        "src/mtare_topo/data/gse_surface_selection_v1.py", "src/mtare_topo/governance.py",
        "src/mtare_topo/governance_surface_selection.py", "src/mtare_topo/governance_identity_inventory.py",
        "src/mtare_topo/governance_partial_structure_training.py", "tools/v3/preflight.py", "tools/v3/create_run.py"]
    for p in sources:
        if p != card_rel and not (root / p).is_file():
            raise FileNotFoundError(p)
    with (root / card_rel).open("x", encoding="utf-8") as stream:
        json.dump(card, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    run = "results/gate3_semantics/gate3_20260907_gse_surface_identity_selection_v1_seed20260906"
    command = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1", "PYTHONHASHSEED=20260906", "/usr/bin/python3",
        "tools/v3/run_gse_surface_selection_v1.py", "--spec", str(root / spec_rel), "--run-dir", str(root / run)]
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260907",
        "slug": "gse_surface_identity_selection_v1", "seed": 20260906, "operation": "audit",
        "data_card": card_rel, "config_path": card_rel, "user_authorization": approval,
        "question": "Does the newly approved single-observation16physicaledges/parent rule yield3360paired causal observations under the original parent split?",
        "method": "Validate pinned full identity reports;stable SHA256 edge/direction ordering;first/middle/last peredge;preserve all three variants and original C07 split.",
        "baseline": "Sealed complete70parent210task inventory;no21decision restriction in this independently revised perception selection. No model/teacher score used.",
        "fallback": "Fail and seal once on source/identity/quota/resource drift;no replacement parents,partial-pair filling,duplicate edge or retry. Identity selection is not a label or scientificPASS.",
        "command": command, "wall_time_cap_s": 120,
        "estimated_cost": {"compute": "CPU standard library only;72sealed JSON metadata files,0original scans/models/labels", "host_ram_gb": 1, "gpu_vram_gb": 0, "disk_gb": .032, "wall_time_hours": 120 / 3600},
        "acceptance_criteria": ["70parents/210tasks/1120distinctphysicaledges/3360variantobservations;2880fit240cal240dev,originalsplit.",
            "Every observation has exactly5causal sourceframes on one direction,sourceID paired across3variants;actual unique frames and routearc spacing disclosed.",
            "No C08-C10/original array/teacher/scan/model/training reads;no continuousgraph or scientificPASS claim.",
            "120s/1GiBhost/32MiB evidence;immutable run,rawlogs/sourcehashes/RUN_STATE/seal retained on failure."],
        "expected_evidence": ["Exact per-parent row/frame manifest and population;input hashes,environment,command,rawlog,metrics,state,SHA256seal."],
        "source_sha256": {p: sha(root / p) for p in sorted(sources)}, "freeze_status": "FROZEN_SELECTION_AUTHORS_STOPPED"}
    with (root / spec_rel).open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"spec": spec_rel, "spec_sha256": sha(root / spec_rel), "scope_sha256": digest(scope), "executed": False}))


if __name__ == "__main__":
    freeze()
