#!/usr/bin/env python3
"""Freeze the fit/C07 local composition-slot Teacher feasibility audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1r2.json"
SOURCE_SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1r2.json"
ATTRIBUTION = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_failure_attribution_v1r2_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
CARD = ROOT / "configs/v3/gate3/data_cards/local_composition_slot_teacher_feasibility_v1.json"
SPEC = ROOT / "configs/v3/gate3/local_composition_slot_teacher_feasibility_v1.json"
RUN_ID = "gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists(): raise RuntimeError("local composition-slot card/spec already exists")
    attribution = _load(ATTRIBUTION / "metrics/attribution/summary.json")
    if attribution.get("decision") != "STOP_GLOBAL_ENDPOINT_ANCHOR_REGRESSION_AND_DESIGN_OBSERVABLE_LOCAL_COMPOSITION_INTERFACE":
        raise RuntimeError("local composition-slot audit requires the exact 3-seed attribution decision")
    source_card = _load(SOURCE_CARD); source_spec = _load(SOURCE_SPEC); card = copy.deepcopy(source_card)
    card["card_id"] = "local_composition_slot_teacher_feasibility_v1"
    card["purpose"] = "Determine whether dual-observed physical endpoint attachments can be represented losslessly as exchangeable local composition clusters before any new model is implemented or trained."
    card["teacher_source"] = "Sealed P1b construction-program endpoint_neighbor and disconnected_overlap, masked only by the sealed 0.25 m endpoint-observability sidecar. No LiDAR values, model outputs, world identity or future test domains are read."
    card["estimated_cost"] = {"compute": "CPU-only pass over 491196 Teacher sequences; no sensor decode or model.", "disk_gb": 0.2, "gpu": 0, "gpu_memory_gb": 0, "host_ram_gb": 4, "wall_time_hours": 1}
    card["failure_policy"] = "Fail closed on non-symmetric attachment, non-clique components, lossy slot reconstruction, overlap collision, capacity overflow, input drift or population drift. Do not repair relations, split clusters, read C08 or train a model."
    card["retention"] = "Retain split/task cluster distributions, capacity choice, losslessness and isolation checks, PNG/PDF/SVG, logs, RUN_STATE and SHA-256 seal."
    card["sampling"]["effective_sample_count"] = 491_196; card["sampling"]["raw_frame_count"] = 491_196; card["sampling"]["effective_structure_event_count"] = 3_225_423
    card["sampling"]["independent_sampling_units"] = "70 disjoint topology parents: C01-C06 fit=60 parents/180 geometry tasks/426552 sequences; C07=10 parents/30 tasks/64644 sequences."
    card["sampling"]["rule"] = "Read every P1b fit/C07 Teacher row and aligned observability row exactly once; no LiDAR tensor, model, shuffle, optimization or threshold fitting."
    card["sampling"]["structure_event_counts"] = {"fit_parent_worlds": 60, "fit_tasks": 180, "fit_sequences": 426_552, "fit_observable_positive_attachments": 2_782_487, "c07_parent_worlds": 10, "c07_tasks": 30, "c07_sequences": 64_644, "c07_observable_positive_attachments": 442_936, "total_sequences": 491_196}
    card["trajectories"] = list(source_card["trajectories"][:2]); card["leakage_audit"]["model_inference_count"] = 0; card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exact fit/C07 tasks, rows and observable-positive attachment counts must reproduce 180/30, 426552/64644 and 2782487/442936.",
        "partition": "Every induced observable attachment connected component must be a clique and endpoint membership must be unique.",
        "losslessness": "Decoding equality of non-dustbin slot labels must reconstruct every qualified attachment with zero pair mismatch.",
        "isolation": "No decoded same-slot pair may be marked disconnected_overlap.",
        "capacity": "Choose the smallest of 8/16/32 covering ceil(1.25*fit maximum clusters); the selected capacity must have zero C07 overflow.",
        "determinism": "Immediate rederivation of every canonical label vector must be bit exact.",
        "resources": "CPU only; wall time <=1 h, host RSS <=4 GiB and output <=0.2 GiB.",
        "forbidden": "Sensor ranges, model forwards, optimizers, C08/C09/C10, graphs and M-TARE reads are all zero.",
    }
    approval = {"approved_at": "2026-09-03T14:12:00+08:00", "approved_by": "user-standing-authorization", "authorized_gates": [3], "authorized_operations": ["audit"], "confirmation_reference": "The user requested uninterrupted automatic best-in-plan execution without repeated approvals.", "scope": "One immutable zero-training fit/C07 local composition-slot Teacher feasibility audit; no C08+, model, graph or M-TARE.", "status": "APPROVED"}
    card["approval"] = approval; card["status"] = CARD_STATUS; _write(CARD, card)
    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "local_composition_slot_teacher_feasibility_v1", "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3, "operation": "audit", "working_directory": str(ROOT), "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": "Can all dual-observed physical endpoint relations in fit and C07 be represented losslessly by a bounded exchangeable local composition-slot assignment?",
        "method": "Connected-component decomposition of the observable endpoint-attachment graph, exact clique/reconstruction/overlap checks and fit-only 8/16/32 capacity selection with 25 percent margin.",
        "baseline": "The same sealed O(E^2) observable attachment matrices; slot decoding must reproduce them exactly.", "fallback": card["failure_policy"], "user_authorization": approval, "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"fit_parent_worlds": 60, "fit_tasks": 180, "fit_rows": 426_552, "fit_positive_pairs": 2_782_487, "c07_parent_worlds": 10, "c07_tasks": 30, "c07_rows": 64_644, "c07_positive_pairs": 442_936, "teacher_rows": 491_196, "sensor_rows": 0, "model_forward_rows": 0, "optimizer_steps": 0, "c08_rows": 0, "graph_replays": 0, "mtare_worlds": 0},
        "estimated_cost": card["estimated_cost"], "expected_evidence": ["Fit/C07 cluster-count and cluster-size distributions.", "Exact clique, lossless reconstruction and disconnected-overlap isolation evidence.", "Fit-selected capacity and C07 overflow count.", "210 task rows, PNG/PDF/SVG, logs, hashes, RUN_STATE and seal."],
        "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Local composition slot Teacher feasibility", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "3600s", "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}", "PYTHONHASHSEED=0", PYTHON, "tools/v3/run_local_composition_slot_teacher_feasibility_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)],
    }
    frozen_inputs = {}
    inputs = [CARD, SOURCE_CARD, SOURCE_SPEC, ATTRIBUTION / "RUN_STATE.json", ATTRIBUTION / "metrics/summary.json", ATTRIBUTION / "metrics/attribution/summary.json", ATTRIBUTION / "artifacts/evidence_sha256.txt", P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt", OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/task_manifest.json", OBS / "artifacts/evidence_sha256.txt"]
    frozen_inputs.update({str(path.relative_to(ROOT)): _sha(path) for path in inputs}); spec["frozen_inputs"] = frozen_inputs
    tools = {"runner": "tools/v3/run_local_composition_slot_teacher_feasibility_v1.py", "freezer": "tools/v3/freeze_local_composition_slot_teacher_feasibility_v1_spec.py", "executor": "tools/v3/execute_local_composition_slot_teacher_feasibility_v1.py", "teacher_module": "src/mtare_topo/evaluation/local_composition_slot_teacher.py", "teacher_tests": "tests/v3/unit/test_local_composition_slot_teacher.py", "executor_tests": "tests/v3/unit/test_execute_local_composition_slot_teacher_feasibility_v1.py", "observability_module": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()}; _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__": main()
