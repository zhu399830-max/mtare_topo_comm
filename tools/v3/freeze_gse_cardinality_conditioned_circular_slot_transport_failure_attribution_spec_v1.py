#!/usr/bin/env python3
"""Freeze the one formal slot-transport failure-attribution spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("slot attribution spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"slot attribution card invalid: {validation.errors}")
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_cardinality_conditioned_circular_slot_transport_three_seed_training_v1_seed0"
    prior_attribution = "results/gate3_semantics/gate3_20260829_gse_circular_exit_set_process_failure_attribution_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json", f"{training}/metrics/selection/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        f"{prior_attribution}/RUN_STATE.json", f"{prior_attribution}/metrics/summary.json", f"{prior_attribution}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    for seed in range(3):
        inputs.append(f"{training}/artifacts/models/seed{seed}/summary.json")
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted((PROJECT_ROOT / training / f"artifacts/models/seed{seed}/development_predictions").glob("*.npz")))
    tools = {
        "slot_transport": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "executor": "tools/v3/execute_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.py",
        "runner": "tools/v3/run_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_cardinality_conditioned_circular_slot_transport_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Why do bijective circular slots improve coarse multi-exit coverage while failing strict two-degree complete sets and high-precision whole-set refusal?",
        "method": "Read all sealed C07/C08 Teacher and three-seed predictions once; compare single-seed and aligned ensemble, circular mean/bin argmax/three-bin local mode, predicted/oracle count, cardinality-stratified error, target two-bin mass, seed disagreement and eight inference-only confidence scores.",
        "baseline": "Formal ensemble circular-mean C07/C08 exact-set at two degrees 0.341285/0.277773 and prior set-process 0.250371/0.225752.",
        "fallback": "If no simple decoder, ensemble or count coupling explains both splits and frozen confidence remains insufficient, select representation-level strict-localization and confidence failure; no training, threshold relaxation or graph compensation.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 10 C07 plus 10 C08 worlds, 21548/24394 observations and 45504/51537 exits; all joins exact.",
            "Reproduce formal ensemble two-degree exact-set fractions to 1e-12 and all decoder tolerance curves monotonic.",
            "Report every seed and ensemble under mean, argmax and fixed immediate-neighbour local mode at 2/4/10 degrees, with predicted and oracle-count diagnostics.",
            "Select each predeclared confidence threshold only on C07 at exit precision at least 0.995 and transfer unchanged to C08.",
            "Zero optimizer, new inference, C09/C10/M-TARE/graph/planner; all frozen inputs unchanged."
        ],
        "expected_counts": {"c07_worlds": 10, "c08_worlds": 10, "c07_observations": 21548, "c08_observations": 24394, "c07_exits": 45504, "c08_exits": 51537, "seeds": 3, "decoder_modes": 3, "tolerances": 3, "optimizer_steps": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Decoder and cardinality comparison plus matched-error and target-mass attribution.", "C07-selected/C08-transferred confidence AUC, precision, recall and accepted coverage.", "PNG/PDF/SVG/source, CSV, environment, log, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only deterministic frozen-output attribution", "wall_time_hours": 0.1, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", PYTHON, "tools/v3/run_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
