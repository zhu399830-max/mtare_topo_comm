#!/usr/bin/env python3
"""Freeze the binary16 probability-parity corrective cache export."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, validate_run_spec, write_json


RUN_ID="gate3_20260829_gse_explicit_composer_cache_export_v1r_seed0"
CARD=PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_explicit_composer_cache_export_v1r.json"
SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_explicit_composer_cache_export_v1r.json"
OLD_CARD=PROJECT_ROOT/"configs/v3/gate3/data_cards/gse_explicit_composer_cache_export_v1.json"
OLD_SPEC=PROJECT_ROOT/"configs/v3/gate3/gse_explicit_composer_cache_export_v1.json"
FAILED="results/gate3_semantics/gate3_20260829_gse_explicit_composer_cache_export_v1_seed0"
PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->int:
    failed=load_json(PROJECT_ROOT/FAILED/"RUN_STATE.json")
    if failed.get("state")!="FAILED" or failed.get("overall_status")!="FAIL_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1" or "token_count_probability" not in (PROJECT_ROOT/FAILED/"logs/01_seed0_export.log").read_text(encoding="utf-8"):
        raise RuntimeError("sealed V1 probability-parity failure is required")
    card=load_json(OLD_CARD)
    card.update({
        "card_id":"gse_explicit_composer_cache_export_v1r",
        "status":"APPROVED_FOR_ONE_IMMUTABLE_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_V1R",
        "purpose":"Correct only V1's over-strict cross-process exact comparison for CUDA-derived binary16 probabilities: primary explicit state remains bit exact; count/transport/reveal probabilities may differ by at most one binary16 epsilon and must preserve every discrete decision.",
        "approval":{"status":"APPROVED","approved_by":"user-standing-authorization","approved_at":"2026-08-29T21:35:00+08:00","scope":"One immutable V1R three-seed C01-C08 geometry-only cache export; same 564,378 frozen forwards and field allow-list, with exact primary state plus <=one binary16 epsilon derived probability and exact decisions; no optimizer/checkpoint/C09/C10/graph/planner/M-TARE.","authorized_operations":["audit"],"authorized_gates":[3],"confirmation_reference":"User explicitly instructed autonomous evidence-optimal choice; V1 failed before fit export and the read-only attribution proved exact repeat, exact primary fields, and only binary16 derived-probability last-bit variation."},
        "failure_policy":"Any primary-field difference, derived probability error above np.finfo(float16).eps, count/transport argmax or reveal@0.5 change, source/schema/resource drift, forbidden field or test/graph access seals FAIL. No partial cache may train a Composer.",
    })
    card["cache_contract"]["cross_process_parity"]={"primary_fields":"bit_exact","derived_probability_fields":["token_count_probability","transport_row_probability","transport_reveal_probability"],"absolute_limit":"np.finfo(np.float16).eps = 0.0009765625","decision_invariants":["token_count_argmax","transport_row_argmax","transport_reveal_at_0p5"],"provenance":"V1 current-process repeat was bit exact; historical differences were 7.63e-6/4.88e-4/1.91e-6 only in CUDA-derived probabilities, with all decisions equal."}
    report=validate_data_card(card)
    if not report.passed:raise RuntimeError("generated V1R card invalid: "+"; ".join(report.errors))
    write_json(CARD,card)
    old=load_json(OLD_SPEC)
    inputs=dict(old["frozen_inputs"])
    additional=[str(OLD_CARD.relative_to(PROJECT_ROOT)),str(OLD_SPEC.relative_to(PROJECT_ROOT)),f"{FAILED}/RUN_STATE.json",f"{FAILED}/metrics/summary.json",f"{FAILED}/artifacts/evidence_sha256.txt",f"{FAILED}/logs/01_seed0_export.log"]
    inputs.update({path:sha(PROJECT_ROOT/path) for path in additional})
    tools={
        "data_card":str(CARD.relative_to(PROJECT_ROOT)),
        "cache_schema":"src/mtare_topo/data/gse_explicit_composer_cache.py",
        "parity_contract":"src/mtare_topo/data/gse_explicit_composer_cache_parity.py",
        "composer_module":"src/mtare_topo/representation/gse_typed_composers.py",
        "schema_tests":"tests/v3/unit/test_gse_explicit_composer_cache.py",
        "parity_tests":"tests/v3/unit/test_gse_explicit_composer_cache_parity.py",
        "composer_tests":"tests/v3/unit/test_gse_typed_composers.py",
        "v1_exporter":"tools/v3/export_gse_explicit_composer_cache_v1.py",
        "corrective_exporter":"tools/v3/export_gse_explicit_composer_cache_v1r.py",
        "v1_runner_helpers":"tools/v3/run_gse_explicit_composer_cache_export_v1.py",
        "runner":"tools/v3/run_gse_explicit_composer_cache_export_v1r.py",
        "freezer":"tools/v3/freeze_gse_explicit_composer_cache_export_spec_v1r.py",
        "governance":"src/mtare_topo/governance.py","preflight":"tools/v3/preflight.py","create_run":"tools/v3/create_run.py",
    }
    spec={**old,
        "slug":"gse_explicit_composer_cache_export_v1r",
        "question":"Can the frozen explicit cache be exported with bit-exact primary state and bounded binary16 cross-process probability parity without changing any Composer decision?",
        "method":"Same V1 frozen inference and allow-list serialization; primary fields bit exact, only three CUDA-derived float16 probability fields use the representation-defined one-epsilon bound plus exact decision parity.",
        "baseline":"V1 exact-all-float16 comparison, which failed at seed0 preflight because current repeat was exact but historical derived probabilities differed by <=0.00048828125 with identical decisions.",
        "fallback":"Any primary drift, >one-epsilon probability difference or decision change stops the cache route; do not widen tolerance or mix copied and inferred worlds.",
        "data_card":str(CARD.relative_to(PROJECT_ROOT)),"config_path":str(CARD.relative_to(PROJECT_ROOT)),"user_authorization":card["approval"],
        "acceptance_criteria":["Exactly 3 seeds x 80 worlds x 188,126 rows, split 142,184/21,548/24,394 per seed.","All eight primary explicit fields are bit exact on 60 C07/C08 seed-world comparisons.","Only count/transport/reveal probability may differ, each by <=0.0009765625; every count/transport argmax and reveal@0.5 decision is identical.","Every cache contains exactly the eleven allow-listed fields; all finite and valid; <=16GiB GPU and <=4GiB result.","Zero optimizer/checkpoint/C09/C10/M-TARE/graph/planner and complete evidence seal."],
        "expected_evidence":["Failed V1 attribution binding; three seed manifests and 240 typed NPZs; per-field exact/bounded errors and decision parity over 60 development worlds; size/runtime/resources figure, logs, environment, RUN_STATE and seal."],
        "frozen_inputs":inputs,
        "frozen_tools":{name:{"path":path,"sha256":sha(PROJECT_ROOT/path)} for name,path in tools.items()},
        "command":["/usr/bin/timeout","14400s",PYTHON,"tools/v3/run_gse_explicit_composer_cache_export_v1r.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/"results/gate3_semantics"/RUN_ID)],
    }
    report=validate_run_spec(spec)
    if not report.passed:raise RuntimeError("generated V1R spec invalid: "+"; ".join(report.errors))
    write_json(SPEC,spec);print(CARD.relative_to(PROJECT_ROOT));print(SPEC.relative_to(PROJECT_ROOT));return 0


if __name__=="__main__":raise SystemExit(main())
