#!/usr/bin/env python3
"""Freeze one exact field-recovery card/spec using existing sealed inventories."""
import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json, preflight, load_json
from mtare_topo.governance_field_recovery import (
    SCHEMA, INPUT_FIELDS, OUTPUT_FIELDS, RESTRICTIONS, selection_sha256,
    validate_scoped_field_recovery_card,
)


P1A = "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
TINY = "results/gate3_semantics/gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0"
INVENTORY = "results/gate3_semantics/gate3_20260905_gse_composition_inventory_v1r_seed0"
CHECKPOINT = "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/artifacts/models/seed0/selected.pt"
CHECKPOINT_SHA = "8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SLUG = "gse_composition_field_recovery_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"


def sha(relative):
    digest = hashlib.sha256()
    with (PROJECT_ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def documents():
    manifest = f"{TINY}/artifacts/tiny_overfit/sample_manifest.json"
    cache = f"{TINY}/artifacts/tiny_overfit/frozen_tiny_features.npz"
    records = json.loads((PROJECT_ROOT / manifest).read_text())
    selected = [{k: r[k] for k in ("task", "row_index", "source_global_sequence_index")} for r in records]
    inventory_file = f"{INVENTORY}/artifacts/observation_inventory.json"
    inventory = json.loads((PROJECT_ROOT / inventory_file).read_text())
    expected = {manifest: "a6027d8b4cf054afdc531d6a480a817a26d11cf19f0c08ec14cb626241f57c19",
                cache: "e3bdd24d764388036ed366ca099d27153f216ac43fcaf8af946ecc408e47adac",
                CHECKPOINT: CHECKPOINT_SHA,
                f"{P1A}/artifacts/evidence_sha256.txt": "79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668",
                f"{P1B}/artifacts/evidence_sha256.txt": "f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47",
                f"{TINY}/artifacts/evidence_sha256.txt": "57fc01e05cff81644cf0f840844eeed7ef610a4186dff703276537f512c05588",
                f"{INVENTORY}/artifacts/evidence_sha256.txt": "e532f9366ff5dc129c6112fafcd33b9c74057adfd6b58b45d37b1ab34e5a7213"}
    for name, digest in expected.items():
        if sha(name) != digest:
            raise ValueError(f"previously frozen source drift: {name}")
    expected[inventory_file] = sha(inventory_file)
    digest = selection_sha256(selected)
    approval = {"status": "APPROVED", "approved_by": "user", "approved_at": "2026-09-05",
                "authorized_operations": ["data_export"], "authorized_gates": [3],
                "scope": "Recover missing shared-frame fields from the original 180 C01 observations using the already designated frozen seed0; no new training or labels.",
                "confirmation_reference": "User's full PLEASE IMPLEMENT THIS PLAN GSE-Graph composition plan and standing authorization for routine in-scope work; same-observation field verification/recovery explicitly included.",
                "selection_sha256": digest, "checkpoint_sha256": CHECKPOINT_SHA}
    card = {"schema_version": SCHEMA, "card_id": SLUG, "purpose": approval["scope"], "approval": approval,
            "teacher_source": "Existing P1b frame indices and causal relative odometry only; no primitive/node targets used by model. Existing 100 node identities used only to verify inventory count.",
            "sampling_rule": "Exactly the original 18 selected five-frame windows per C01 mixed parent; source sequences distance-sampled at 1 m, no new selection or deletion; per-window measured distances retained below.",
            "license_or_allowed_use": "Existing project-authorized procedural research dataset and frozen model; local derivative export only, no redistribution.",
            "partition": "fit", "duration_s": None, "time_basis": "distance_sampled_no_acquisition_clock",
            "independent_sampling_unit": "topology_parent", "observation_count": 180, "parent_count": 10,
            "node_count": 100, "unique_source_frame_count": 900, "frames_per_observation": 5,
            "worlds": sorted({r["task"].split("__")[0] for r in records}), "selected_rows": selected,
            "selection_sha256": digest, "input_fields": list(INPUT_FIELDS), "output_fields": list(OUTPUT_FIELDS),
            "restrictions": dict.fromkeys(RESTRICTIONS, True), "sealed_sources": expected,
            "checkpoint": {"path": CHECKPOINT, "sha256": CHECKPOINT_SHA, "seed": 0, "frozen": True, "selection": "none"},
            "inference": {"primary_observations": 180, "repeat_observations": 18, "total_observations": 198,
                          "repeat_task": sorted({r["task"] for r in records})[0]},
            "trajectory_metadata": [{k: row[k] for k in ("task", "row_index", "traversal_id_metadata_only", "window_frame_rows",
                                       "window_measured_chord_length_m", "whole_traversal_length_m")} for row in inventory],
            "leakage_audit": "Only listed C01 range/valid chunks and window/relative-pose metadata are decoded; source seal indices are metadata. No C07-C10 assets, identity/absolute pose into model, future frames, teacher labels, training, threshold or checkpoint selection.",
            "population_note": "180 observations are not 180 independent environments: 10 topology parents, 100 old node identities, 20/120/38/2 old degree inventory (not new event targets)."}
    if not validate_scoped_field_recovery_card(card).passed:
        raise ValueError(validate_scoped_field_recovery_card(card).errors)
    source_files = sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "src/mtare_topo").rglob("*.py"))
    source_files += ["tools/v3/_bootstrap.py", "tools/v3/preflight.py", "tools/v3/create_run.py",
                     "tools/v3/run_gse_composition_field_recovery_v1.py", "tools/v3/freeze_gse_composition_field_recovery_v1.py",
                     "tools/v3/train_primitive_relation_model_v1.py", "tools/v3/train_gse_structural_node_tiny_overfit_v1.py",
                     "tests/v3/unit/test_gse_field_recovery_runner.py", "tests/v3/unit/test_gse_field_recovery_card.py",
                     "tests/v3/unit/test_gse_scoped_model_input.py"]
    run = f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260905", "slug": SLUG, "seed": 0,
            "operation": "data_export", "question": "Can the same frozen backbone exactly reproduce the sealed legacy cache and recover common-frame geometry without training or new teacher labels?",
            "method": "Exact task/row sensor allowlist; frozen seed0 CUDA TF32-off inference; all 32 primitive slots and six raw fields saved without teacher filtering; 18-observation repeat.",
            "baseline": "Sealed legacy 180x64x44 features and 180x64 confidence arrays; exact elementwise comparison, not model performance ranking.",
            "fallback": "Any drift or nonzero cache/repeat mismatch fails and stops; no changed rows, thresholds, checkpoint, precision or retry.",
            "data_card": CARD, "config_path": CARD, "user_authorization": approval,
            "command": ["env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "OMP_NUM_THREADS=1", "PYTHONHASHSEED=0",
                        "/usr/bin/timeout", "--kill-after=15s", "600s", PYTHON,
                        "tools/v3/run_gse_composition_field_recovery_v1.py", "--spec", str(PROJECT_ROOT / SPEC), "--run-dir", str(PROJECT_ROOT / run)],
            "estimated_cost": {"disk_gb": .05, "wall_time_hours": 1/6, "host_ram_gb": 8, "gpu_vram_gb": 4,
                               "compute": "198 total frozen inference windows on existing RTX 5090 D; zero optimizer steps"},
            "expected_versions": {"python": "3.13.5", "numpy": "2.1.3", "zarr": "2.18.7", "torch": "2.9.0+cu129"},
            "selection_manifest": manifest, "legacy_cache": cache,
            "sensor_root": f"{P1A}/artifacts/dataset/fit", "teacher_root": f"{P1B}/artifacts/teacher/fit",
            "source_seals": [f"{P1A}/artifacts/evidence_sha256.txt", f"{P1B}/artifacts/evidence_sha256.txt"],
            "source_sha256": {path: sha(path) for path in source_files},
            "acceptance_criteria": ["Exactly 180 main and 18 repeat inference observations, 900 unique source frames from the original ten C01 tasks.",
                 "All legacy features/confidences and first-task repeats are elementwise identical; six raw fields finite and shape-correct.",
                 "All model parameters frozen with identical state hash and no gradients; no teacher identity in model input or new labels.",
                 "Accessed source chunks match original seals; tool/checkpoint/input hashes unchanged; <=600s,8GiB host,4GiB GPU reserved.",
                 "Save one immutable result with metadata/log/arrays/metrics/RUN_STATE/seal; component success is not scientific Gate PASS."],
            "expected_evidence": ["Exact Data Card/spec, source/tool/checkpoint hashes, CUDA environment, per-task raw log, ten six-field archives, row/frame manifest, metrics, RUN_STATE and SHA-256 seal."]}
    return card, spec


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("card/spec already exists; do not overwrite")
    card, spec = documents()
    if args.freeze:
        write_json(PROJECT_ROOT / CARD, card); write_json(PROJECT_ROOT / SPEC, spec)
        report = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
        print(json.dumps({"card": CARD, "spec": SPEC, "preflight_passed": report.passed, "errors": report.errors}))
        return int(not report.passed)
    print(json.dumps({"valid_card": True, "observations": 180, "inferences": 198, "source_tools": len(spec["source_sha256"]), "files_written": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
