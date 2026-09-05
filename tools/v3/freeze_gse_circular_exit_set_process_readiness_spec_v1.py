#!/usr/bin/env python3
"""Freeze one immutable causal circular exit-set process readiness."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

RUN_ID="gate3_20260829_gse_circular_exit_set_process_readiness_v1_seed0"
SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_circular_exit_set_process_readiness_v1.json"
DATA_CARD="configs/v3/gate3/data_cards/gse_circular_exit_set_process_readiness_v1.json"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def main()->int:
    if SPEC.exists():raise RuntimeError("exit-set readiness spec already exists")
    card=load_json(PROJECT_ROOT/DATA_CARD);validation=validate_data_card(card)
    if not validation.passed:raise RuntimeError(f"exit-set Data Card invalid: {validation.errors}")
    dataset="results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    teacher="results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    v2="results/gate3_semantics/gate3_20260829_gse_circular_peak_geometry_three_seed_training_v2_seed0"
    inputs=[DATA_CARD,f"{dataset}/RUN_STATE.json",f"{dataset}/metrics/summary.json",f"{dataset}/artifacts/shard_manifest.json",f"{dataset}/artifacts/evidence_sha256.txt",f"{teacher}/RUN_STATE.json",f"{teacher}/metrics/summary.json",f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",f"{teacher}/artifacts/evidence_sha256.txt",f"{v2}/RUN_STATE.json",f"{v2}/metrics/summary.json",f"{v2}/artifacts/evidence_sha256.txt","configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json"]
    tools={
        "set_process":"src/mtare_topo/representation/gse_circular_exit_set_process.py",
        "backbone":"src/mtare_topo/representation/gse_circular_peak_geometry_model.py",
        "circular_layers":"src/mtare_topo/representation/phase3_structural_semantics.py",
        "executor":"tools/v3/execute_gse_circular_exit_set_process_readiness_v1.py",
        "source_audit":"tools/v3/execute_gse_circular_peak_geometry_model_readiness_v1.py",
        "runner":"tools/v3/run_gse_circular_exit_set_process_readiness_v1.py",
        "freezer":"tools/v3/freeze_gse_circular_exit_set_process_readiness_spec_v1.py",
        "tests":"tests/v3/unit/test_gse_circular_exit_set_process.py",
        "backbone_tests":"tests/v3/unit/test_gse_circular_peak_geometry_model.py",
        "governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py"
    }
    spec={
        "schema_version":"v3_run_spec_v1","gate":3,"execution_phase":3,"date":"20260829","slug":"gse_circular_exit_set_process_readiness_v1","seed":0,"operation":"audit","data_card":DATA_CARD,
        "question":"Can one normalized cardinality-conditioned circular finite-set process expose 1-4 continuous executable exits without free-query ghosts, local-peak thresholds or loss of rotation/permutation contracts?",
        "method":"Zero-training 769784-parameter scans-only model: proven causal circular backbone, normalized 180-bin set intensity, explicit 1-4 count, permutation-invariant set likelihood and count-conditioned continuous decode.",
        "baseline":"Frozen V2 independent local-peak AP 0.110522/0.097221 with no 0.995-precision threshold; no baseline performance rerun.",
        "fallback":"Any population, causality, mass-budget, cardinality, near-exit, rotation, permutation or finite-backward failure stops before training.",
        "user_authorization":card["approval"],
        "acceptance_criteria":[
            "Exact 80 worlds, 188126 observations, 396913 exits and cardinality 1/2/3/4 counts 7525/154279/24458/1864.",
            "Scans-only 769784-parameter interface has zero free queries, normalized mass, explicit count and no existence threshold.",
            "Correct finite set/cardinality ranks below shifted, duplicate, ghost and wrong-count synthetic outputs; wrap and bins two apart remain distinct.",
            "40-column rotation, batch permutation, repeat and five-frame history contracts hold with max error <=3e-5.",
            "Eight real rows cover counts 1-4, masks and partitions with finite outputs/loss/backward.",
            "Zero optimizer/checkpoint/threshold/C09/C10/M-TARE/graph/planner and frozen inputs unchanged."
        ],
        "expected_counts":{"worlds":80,"raw_frames":252430,"observations":188126,"fit_observations":142184,"selection_observations":45942,"visible_exits":396913,"count1":7525,"count2":154279,"count3":24458,"count4":1864,"real_rows":8,"parameters":769784,"optimizer_steps":0,"checkpoint_writes":0,"threshold_selection_steps":0,"graph_replays":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0},
        "expected_evidence":["Full population/reference/join and hidden-input audit.","Set/cardinality ordering, wrap/near-exit decode and real finite backward.","Rotation/permutation/repeat/history errors and typed contract.","PNG/PDF/SVG/source, row table, environment, log, RUN_STATE and seal."],
        "estimated_cost":{"compute":"CPU-only full reference audit and eight-row forward/backward","wall_time_hours":0.08,"host_ram_gb":4,"gpu_memory_gb":0,"disk_gb":0.1,"gpu":"none"},
        "frozen_inputs":{path:sha256(PROJECT_ROOT/path) for path in sorted(inputs)},
        "frozen_tools":{name:{"path":path,"sha256":sha256(PROJECT_ROOT/path)} for name,path in tools.items()},
        "working_directory":str(PROJECT_ROOT),
        "command":["/usr/bin/timeout","1200s",PYTHON,"tools/v3/run_gse_circular_exit_set_process_readiness_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC,spec);print(SPEC.relative_to(PROJECT_ROOT));return 0

if __name__=="__main__":raise SystemExit(main())
