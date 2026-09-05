#!/usr/bin/env python3
"""Bind the completed C02 metadata selection to one frozen inference run."""
import argparse
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json, preflight
from mtare_topo.governance_field_recovery import INPUT_FIELDS, selection_sha256
from mtare_topo.governance_head_inference import (
    SCHEMA, GEOMTEACHER_FIELDS, OUTPUT_FIELDS, RESTRICTIONS, INFERENCE_COUNTS,
    validate_head_development_inference_card,
)
from run_gse_composition_inventory_v1 import sha

SLUG = "gse_head_development_inference_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("registration already exists")
    old_path = "configs/v3/gate3/data_cards/gse_point_axis_probe_v1.json"
    old = load_json(PROJECT_ROOT / old_path)
    previous = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_point_axis_probe_v1.json")
    metadata = "results/gate3_semantics/gate3_20260905_gse_head_development_metadata_v1_seed0"
    training = "results/gate3_semantics/gate3_20260905_gse_point_axis_probe_v1_seed0"
    selection_path = metadata + "/artifacts/selected_rows.json"
    sealed = {
        previous["source_seals"][0]: "79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668",
        previous["source_seals"][1]: "f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47",
        metadata + "/artifacts/evidence_sha256.txt": "83fb4df8c3b55b17d82f569e3b5c3c8cf126a2ce7b3d854a22a52606ed05adab",
        selection_path: "9243ddbec8d27d8b965ef8cd5742c63814df131e775e75ad5b4e7d9b4dd07ef6",
        training + "/artifacts/evidence_sha256.txt": "2e1e72069e1c97422cfb2ac46853370247b1b0882aaaf774a9bdb0361de4a7c7",
    }
    for path, digest in sealed.items():
        if sha(PROJECT_ROOT / path) != digest:
            raise ValueError("input index or selection drift")
    rows = json.loads((PROJECT_ROOT / selection_path).read_text())
    checkpoints = {"backbone": old["checkpoint"], "head": {
        "path": training + "/artifacts/raw_no_offset_final.pt",
        "sha256": "2f31bc8fd56d4aaa77b597030c0e9b9ffbd50efc5aa09390ffb8e9d5d3e49843",
        "seed": 0, "frozen": True, "selection": "none"}}
    # Registration reads sealed manifests, not model tensors or sensor arrays.
    for checkpoint in checkpoints.values():
        sealed[checkpoint["path"]] = checkpoint["sha256"]
    sealed[old_path] = sha(PROJECT_ROOT / old_path)
    digest = selection_sha256(rows)
    scope = ("Exact sealed C02 ten-parent development180 observations/900 unique frames/1489 visible fragments; "
             "frozen raw_no_offset final head versus frozen legacy seed0, each180 primary+18 repeat outputs. "
             "Existing axis targets only for scoring. No optimizer/new labels/calibration/checkpoint selection/offset checkpoint/C07-C10/graph.")
    approval = {"status": "APPROVED", "approved_by": "user_standing_scope", "approved_at": "2026-09-05",
        "authorized_operations": ["data_export"], "authorized_gates": [3], "scope": scope,
        "selection_sha256": digest,
        "checkpoint_sha256": {k: v["sha256"] for k, v in checkpoints.items()},
        "confirmation_reference": "User approved GSE-Graph implementation and continuous in-scope execution without repeated approval. PLAN/PROGRESS completed metadata stage explicitly permits this separately frozen development inference, not training or strict test."}
    card = load_json(PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_head_development_metadata_v1.json")
    for key in ("target_observations", "fields", "source_global_sequence_indices"):
        del card[key]
    card.update(schema_version=SCHEMA, card_id=SLUG, purpose=scope,
        teacher_source="Existing sealed P1b visible axis control points and masks only, scoring side; no identity or target in model inputs. No new teacher.",
        observation_count=180, unique_source_frame_count=900, visible_fragment_count=1489,
        frames_per_observation=5, selected_rows=rows, selection_sha256=digest,
        input_fields=list(INPUT_FIELDS), geomteacher_fields=list(GEOMTEACHER_FIELDS), output_fields=list(OUTPUT_FIELDS),
        source_roots={"sensor": previous["sensor_root"], "teacher": previous["teacher_root"]},
        source_seals=dict(zip(("sensor", "teacher"), previous["source_seals"][:2])),
        metadata_selection_path=selection_path, checkpoints=checkpoints, sealed_sources=sealed,
        restrictions=dict.fromkeys(RESTRICTIONS, True), approval=approval,
        inference={**INFERENCE_COUNTS, "repeat_task": card["tasks"][0]})
    report = validate_head_development_inference_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    files = sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "src/mtare_topo").rglob("*.py"))
    files += ["tools/v3/" + name for name in (
        "_bootstrap.py", "preflight.py", "create_run.py", "run_gse_composition_inventory_v1.py",
        "run_gse_composition_field_recovery_v1.py", "run_gse_head_development_metadata_v1.py",
        "run_gse_head_development_inference_v1.py", "freeze_gse_head_development_inference_v1.py")]
    files += ["tests/v3/unit/" + name for name in (
        "test_gse_head_inference_card.py", "test_gse_head_development_inference.py",
        "test_gse_head_development_inference_runner.py")]
    run = f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0"
    spec = dict(schema_version="v3_run_spec_v1", gate=3, date="20260905", slug=SLUG, seed=0,
        operation="data_export", data_card=CARD, config_path=CARD, user_authorization=approval,
        question="Does the frozen raw head retain a geometry-only advantage over the legacy model on fixed C02 head-development parents? Not whole-model unseen or detector qualification.",
        method=scope, baseline="Exact legacy frozen seed0 axis predictions on the identical180 observations, identical geometry matcher and metrics.",
        fallback="Stop on drift, nonfinite outputs, nondeterminism or resource failure; no substituted samples, retries, new training or offset branch expansion.",
        command=["env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1", "PYTHONHASHSEED=0", PYTHON,
                 "tools/v3/run_gse_head_development_inference_v1.py", "--spec", str(PROJECT_ROOT / SPEC), "--run-dir", str(PROJECT_ROOT / run)],
        expected_versions=previous["expected_versions"], source_sha256={p: sha(PROJECT_ROOT / p) for p in files},
        estimated_cost={"disk_gb": 0.1, "wall_time_hours": 1/30, "host_ram_gb": 4, "gpu_ram_gb": 4,
                        "compute": "One frozen CUDA inference;120s hard cap,4GiB host/GPU;zero optimizer,newlabels,calibration,threshold or model selection."},
        acceptance_criteria=[
            "Exact180 selected observations,900 unique frames,1489 visible targets; only declared source fields; source/tool hashes unchanged.",
            "Both outputs finite and first18 bitwise repeatable; old full-forward axes equal cached-memory reconstruction; checkpoint states unchanged and zero gradients/optimizer.",
            "All180 per-observation geometry metrics and ten-parent macro comparisons retained, including direction-unknown population and unpenalized surplus queries. No cherry-picked resolution/sample/model.",
            "Report development raw-v-legacy differences without inventing detector/generalization or scientific Gate PASS; original offset STOP and historical failed run unchanged.",
            "Resource bounds,11SVG,raw outputs/targets/read hashes/log/environment/state and output SHA256 seal complete."],
        expected_evidence=["180 predictions per model plus18 exact repeats;180 cached backbone windows;two NPZ,360 observation metric records,selection,11 full-population SVG,source ledger,summary,environment,logs,state,seal."])
    if args.freeze:
        write_json(PROJECT_ROOT / CARD, card)
        write_json(PROJECT_ROOT / SPEC, spec)
        checked = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
        print(json.dumps({"passed": checked.passed, "errors": checked.errors}))
        return int(not checked.passed)
    print(json.dumps({"valid_card": True, "files_written": 0, "observations": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
