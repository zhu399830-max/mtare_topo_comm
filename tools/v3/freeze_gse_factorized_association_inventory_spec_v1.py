#!/usr/bin/env python3
"""Freeze the Factorized GSE-Graph association inventory card and spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_factorized_association_inventory_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_inventory_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_association_inventory_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_action_conditioned_state_feasibility_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update({
        "card_id": "gse_factorized_association_inventory_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_INVENTORY_V1",
        "purpose": "Inventory whether junction/terminal identities have full learned exit tokens, causal inbound edge geometry and safe-association positive/hard-negative pairs before any route-conditioned model is authorized.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-27T23:59:00+08:00",
        "authorized_operations": ["training"], "authorized_gates": [3],
        "scope": "One immutable CPU-only C01-C08 read-only inventory; zero optimization, inference, threshold selection, C09/C10/strict/M-TARE or graph/planner access.",
        "confirmation_reference": "User granted standing authority to execute the evidence-supported best in-scope route without routine approval prompts.",
    }
    card["source"] = {
        "raw_sources": [
            "Sealed corrected C01-C08 Teacher with 188126 causal observations.",
            "Three sealed 146D feature arrays and full six-token exit archives with heading, width, vertical profile and descriptor.",
            "Sealed C01-C08 open-set pair cache; no raw LiDAR, checkpoint or test-world data is read.",
        ],
        "license_or_allowed_use": "Local research use of project procedural worlds and locally trained frozen outputs with provenance.",
        "corrected_teacher_run": TEACHER, "frozen_component_run": VERIFIER,
    }
    card["sampling"] = {
        "raw_frame_count": 252430, "effective_sample_count": 188126,
        "fit_observations": 142184, "selection_observations": 45942,
        "spatial_interval_m": 1.0,
        "independent_units": "80 topology parents; 792 fit and 274 selection decision-node identities before inventory checks.",
        "temporal_context": "Current plus up to four earlier observation-level learned geometry states on the same directed traversal; never crosses reset.",
        "rule": "Decision nodes are junction/terminal only. Online pairs are same-parent, strictly-past candidates at 3D distance <=16m; full token archives remain unthresholded.",
    }
    card["methods"] = {
        "main": "Count decision identities, full-token validity, causal five-observation inbound profiles, incident physical edges, and online positive/hard-negative pairs by family.",
        "baseline": "Existing current-observation association population and its failed open-set/token-aware models.",
        "fallback": "If profile or pair coverage is insufficient, redesign the association data contract before any model training.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exact fit/selection decision rows and identities 19743/5551/417/375 and 6865/1974/146/128.",
        "profile": "At least 90% of decision identities in each partition have one causal >=5-observation inbound geometry profile.",
        "pairs": "Every topology family has nonzero online decision positive and hard-negative pairs; selection has nonzero cross-edge positives.",
        "reporting": "All families and both partitions reported without threshold or family selection.",
    }
    card["estimated_cost"] = {"compute": "CPU-only frozen-array inventory", "disk_gb": .25, "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": .25}
    card["retention"] = "Retain metrics, per-family CSV, PNG/PDF/SVG with source JSON, log, config, RUN_STATE and seal."
    card["evidence"] = {
        "machine_metrics": "Exact row/identity/profile/pair/token counts and zero forbidden operations.",
        "complete_visual_review": "Two-panel profile coverage and per-family pair inventory retained in raster/vector forms.",
        "failure_policy": "Any source, alignment, token, causal-profile, population or family-pair defect seals FAIL; no repair inside the run.",
    }
    card["failure_policy"] = card["evidence"]["failure_policy"]
    write_json(CARD, card)
    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json", f"{VERIFIER}/artifacts/evidence_sha256.txt", f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend([
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{VERIFIER}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "inventory_contract": "src/mtare_topo/evaluation/gse_factorized_association_inventory.py",
        "executor": "tools/v3/execute_gse_factorized_association_inventory_v1.py",
        "runner": "tools/v3/run_gse_factorized_association_inventory_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_association_inventory_spec_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260827",
        "slug": "gse_factorized_association_inventory_v1", "seed": 0, "operation": "audit",
        "question": "Does C01-C08 contain enough causal full-token, incident-edge geometry, positive revisit and hard-negative evidence for one route-conditioned association proof?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            "Exact 60/142184 fit and 20/45942 selection observations with decision rows/identities 19743/5551/417/375 and 6865/1974/146/128.",
            "Three full-token archives align; >=90% fit and selection decision identities have a causal five-observation inbound profile.",
            "Every family has online positive and hard-negative decision pairs, selection cross-edge positives are nonzero, and all optimizer/inference/C09/C10/M-TARE counts are zero.",
        ],
        "expected_counts": {"fit_worlds": 60, "fit_observations": 142184, "selection_worlds": 20, "selection_observations": 45942, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Decision identity/profile/pair/token inventory, per-family CSV, PNG/PDF/SVG and source JSON, environment, log, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU-only frozen-array inventory", "disk_gb": .25, "wall_time_hours": .25},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "960s", PYTHON,
                    "tools/v3/run_gse_factorized_association_inventory_v1.py", "--spec", str(SPEC),
                    "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
