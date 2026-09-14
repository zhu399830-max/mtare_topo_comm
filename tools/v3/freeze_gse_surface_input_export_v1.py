#!/usr/bin/env python3
"""Freeze selected sensor input scope. Reads headers/hash indices, no payload."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_surface_input_scope_v1 import compile_input_scope, access_summary
from mtare_topo.governance_surface_input import CARD_ID, SCHEMA, ENVIRONMENT, validate_surface_input_card
from mtare_topo.governance_surface_selection import digest

SIDECAR = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
CARD_REL = "configs/v3/gate3/data_cards/gse_surface_input_export_v1.json"
SPEC_REL = "configs/v3/gate3/gse_surface_input_export_v1.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current_environment():
    import numpy
    import zarr
    import numcodecs
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "zarr": zarr.__version__, "numcodecs": numcodecs.__version__}


def freeze(root=PROJECT_ROOT):
    root = Path(root).resolve(strict=True)
    if (root / CARD_REL).exists() or (root / SPEC_REL).exists():
        raise FileExistsError("no overwrite or refreeze")
    if current_environment() != ENVIRONMENT:
        raise ValueError("exact declared existing data sidecar required")
    scope, manifest, preparation_reads = compile_input_scope(root)
    approval = {"status": "APPROVED", "approved_by": "user-standing-scope-authorization", "approved_at": "2026-09-07",
        "authorized_operations": ["data_export"], "authorized_gates": [3], "scope_sha256": digest(scope),
        "scope": "One lossless export of the independently selected3360causal observations/16800variantframes fromC01-C07; six exact scan/motion/identity fields only, no labels/physicalteacher/models/training.",
        "confirmation_reference": "User2026-09-07 PLEASE IMPLEMENT THIS PLAN:GSE-Graph surface relation execution, followed by 设置一个目标一直执行; continuation uses prior standing authority. This card binds the new measured3360population and exact chunk collateral; no invented new approval or authority to train."}
    card = {"schema_version": SCHEMA, "card_id": CARD_ID, "operation": "data_export",
        "purpose": "Reuse selected LiDAR and relative motion losslessly, isolate construction/teacher truth and preserve source lineage for observed surface work.",
        "limitations": "Input-only export, no qualified visible labels or physical root reachability. C07 exposed development, shared compressed chunks include explicitly listed nonselected rows which are never exported as training examples. No actual scan payload opened during preparation.",
        "scope": scope, "scope_sha256": digest(scope), "approval": approval,
        "scientific_gate_pass": False, "training_eligibility": False}
    checked = validate_surface_input_card(card)
    if not checked.passed:
        raise ValueError("input card invalid: " + "; ".join(checked.errors))
    sources = [CARD_REL, "docs/GSE_SURFACE_RELATION_EXECUTION_V1.md", "docs/GSE_SURFACE_ROBOT_TEACHER_STATUS_V1.md",
        "src/mtare_topo/__init__.py", "src/mtare_topo/data/__init__.py", "src/mtare_topo/governance.py",
        "src/mtare_topo/governance_surface_input.py", "src/mtare_topo/governance_surface_selection.py",
        "src/mtare_topo/governance_identity_inventory.py", "src/mtare_topo/governance_partial_structure_training.py",
        "src/mtare_topo/data/gse_surface_input_scope_v1.py", "src/mtare_topo/data/gse_surface_input_export_v1.py",
        "tools/v3/_bootstrap.py", "tools/v3/preflight.py", "tools/v3/create_run.py",
        "tools/v3/freeze_gse_surface_input_export_v1.py", "tools/v3/run_gse_surface_input_export_v1.py"]
    for path in sources:
        if path != CARD_REL and not (root / path).is_file():
            raise FileNotFoundError(path)
    with (root / CARD_REL).open("x", encoding="utf8") as stream:
        json.dump(card, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    run_rel = "results/gate3_semantics/gate3_20260907_gse_surface_input_export_v1_seed20260906"
    command = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "PYTHONHASHSEED=20260906", SIDECAR, "tools/v3/run_gse_surface_input_export_v1.py",
        "--spec", str(root / SPEC_REL), "--run-dir", str(root / run_rel)]
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260907", "slug": "gse_surface_input_export_v1",
        "seed": 20260906, "operation": "data_export", "data_card": CARD_REL, "config_path": CARD_REL,
        "user_authorization": approval, "question": "Can the exact3360selected observations be transported losslessly as only LiDAR and causal relative motion, with bounded/excluded chunk collateral?",
        "method": "Pinned selection and P1a/P1b seals; derive exact chunks from actual verified headers, decode each selected chunk once per task, identity-check fiveframes and save six input arrays only.",
        "baseline": "Sealed original scan/valid and stored relative motion; byte-exact float32 inputs, no model or teacher score.",
        "fallback": "Fail and seal on input/identity/environment/resource drift; no new samples, silent fill, retry, labels or model execution. Physical-root labels remain paused for missing deployment collision/capability contract.",
        "command": command, "wall_time_cap_s": 3600,
        "estimated_cost": {"compute": "CPU data sidecar only,210tasks,selected raw range/valid0.96768GB plus motion/identity; actual collateral in exact card", "host_ram_gb": 4, "gpu_vram_gb": 0, "disk_gb": 2, "wall_time_hours": 1},
        "acceptance_criteria": ["210tasks/3360observations/16800uniquevariantframes; original1120edge/70parent and2880/240/240split preserved.",
            "All source bytes match pinned seals; actual headers reproduce exact chunk permissions and report clipped/padded collateral separately.",
            "Lossless array roundtrip and exact sourceID/frame_row/motioncurrentorigin checks; no GT geometry,absolute pose,construction,model or labels.",
            "Single immutable run,3600s/4GiBhost/2GiBevidence/zeroGPU;failure rawlog,state and SHA256seal retained."],
        "expected_evidence": ["210lossless input shards,source identity manifest,per-task read and roundtrip report,exact source/command/environment/freeze hashes,rawlog,metrics,RUN_STATE,seal. No scientific model score."],
        "source_sha256": {p: sha(root / p) for p in sorted(sources)},
        "environment": ENVIRONMENT, "python_executable_sha256": sha(Path(SIDECAR).resolve(strict=True)),
        "sidecar_freeze": subprocess.check_output([SIDECAR, "-m", "pip", "freeze", "--all"], text=True),
        "preparation_metadata_reads_sha256": preparation_reads, "access_summary": access_summary(scope),
        "freeze_status": "FROZEN_INPUT_EXPORT_AUTHORS_STOPPED"}
    spec["sidecar_freeze_sha256"] = hashlib.sha256(spec["sidecar_freeze"].encode()).hexdigest()
    with (root / SPEC_REL).open("x", encoding="utf8") as stream:
        json.dump(spec, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({"spec": SPEC_REL, "spec_sha256": sha(root / SPEC_REL), "population": scope["population"],
        "files": len(scope["file_sha256"]), "metadata_reads": len(preparation_reads), "access": spec["access_summary"], "executed": False}, allow_nan=False))


if __name__ == "__main__":
    freeze()
