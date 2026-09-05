#!/usr/bin/env python3
"""Freeze, never execute, the independently authorized C01 coordinate control."""
from copy import deepcopy
import json
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_coordinate_control import SCHEMA, TRAINING, RESTRICTIONS, validate_coordinate_control_card
from run_gse_composition_field_recovery_v1 import sha

SLUG = "gse_coordinate_control_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
SOURCE_SPEC = "configs/v3/gate3/gse_point_axis_probe_v1.json"


def documents():
    old = load_json(PROJECT_ROOT / SOURCE_SPEC)
    card = load_json(PROJECT_ROOT / old["data_card"])
    card.update({"schema_version": SCHEMA, "card_id": SLUG, "training": deepcopy(TRAINING),
        "restrictions": dict.fromkeys(RESTRICTIONS, True), "visible_fragment_count": 1452, "compute_device": "cuda",
        "purpose": "Same180 C01 matched-budget coordinate intervention: raw_coordinates versus mean_broadcast_coordinates; independent of main structure-line progress and no offset training.",
        "source_roots": {"sensor": old["sensor_root"], "teacher": old["teacher_root"]},
        "source_seals": {role: old["source_seals"][i] for i, role in enumerate(("sensor", "teacher", "reference"))},
        "prediction_reference_root": old["prediction_reference_root"],
        "supervision_boundary": "Same existing1452 axis/mask targets only. Exact student tensors/point count/validity/features/token provenance/head/initial parameters/order/budget; only raw XYZ vs corresponding token-mean XYZ changes. No point identity/new teacher/event loss.",
        "matching_population": "All180 observations/all32 candidates,1452 existing targets; pure geometry Hungarian matching, surplus candidates counted and saved but not false-positive penalized. FIT comparison only.",
        "selection_policy": "Both predeclared variants, final step540 only; no seed/checkpoint selection, no threshold fitting, no retry, no C02 inference through this card."})
    card["approval"].update({"authorized_operations": ["training"], "scope": card["purpose"],
        "confirmation_reference": "User's latest approved convergence plan explicitly requests parallel scientific work and RTX5090 priority, including matched-budget raw-coordinate versus mean-broadcast control. Standing in-scope execution authorization, not a new user dialogue or reuse of old offset experiment approval."})
    for path in (SOURCE_SPEC, old["data_card"]): card["sealed_sources"][path] = sha(PROJECT_ROOT / path)
    report = validate_coordinate_control_card(card)
    if not report.passed: raise ValueError(report.errors)
    spec = deepcopy(old)
    spec.update({"slug": SLUG, "data_card": CARD, "config_path": CARD, "training": deepcopy(TRAINING),
        "question": "Does retaining point-level raw coordinates improve fitted axis geometry over broadcasting token-mean coordinates when head, initialization, features, sample order and update budget are identical?",
        "method": "GPU-first same C01 180/900/1452; frozen designated seed0 backbone; original six-field reference and memory-control parity; configure_variant(initial,raw_no_offset) for both new coordinate variants; B1 Adam.001/three seed0 permutations/540steps each; teacher only in geometry loss/scoring.",
        "baseline": "Mean-broadcast coordinates with identical point count, validity, token indices, frozen features, parameter initialization and optimizer schedule. NOT historical900-token decoder. Initial coordinate-dependent outputs are NOT required equal. Frozen old axes retained for reference only.",
        "fallback": "System/input/gradient/resource/ledger drift stops this run. Report positive/negative/tied outcome without promoting scientificGate or changing threshold/budget. This independent comparison does not block parallel main structure work; C02 requires a separate card.",
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            "Exact original180 C01 rows/10 parents/900 unique frames/1452 targets; old seed0 six-field and memory/control parity exact; source seal paths prefiltered to C01 allowed fields before resolve.",
            "Both heads start with identical parameter hashes; both frozen-zero offset modules; only coordinate inputs differ; all540 schedules/Adam settings identical; initial outputs may differ.",
            "Finite real-input gradients,1080 total optimizer steps,backbone/initial state unchanged,no new labels/no C02-C10/no graph; final-only weights and optimizer states preserved.",
            "Report all180/all32 initial/final predictions, matched coordinate/transverse/direction errors, unresolved orientations/surplus candidates,per-parent differences and allXY/XZ views. No scientificGate pass or detection/generalization claim.",
            "Within1200s/4GiB host/4GiB GPU; freeze command/input/tools/env; no overwrite/retry; raw logs,exact ledger,RUN_STATE and SHA256 seal required."],
        "expected_evidence": ["Exact independently authorized card/spec/source hashes; same initial parameters, coordinate-specific initial outputs,1080-step log and final optimizer checkpoints; all candidate axes,existing target cache,complete metrics and11SVG previews;source read inventory,RUN_STATE andSHA256 seal."],
        "estimated_cost": {"disk_gb": .25, "wall_time_hours": 1/3, "host_ram_gb": 4, "gpu_vram_gb": 4,
            "compute": "RTX5090 preferred;<=1200s,180 frozen parity+180 memory cache windows,2 zero-update gradient checks,1080 B1 updates,720 initial/final head evaluation windows;0 C02/test/graph."}})
    executable = old["command"][6]
    spec["command"] = ["env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OMP_NUM_THREADS=1", "PYTHONHASHSEED=0", executable, "tools/v3/run_gse_coordinate_control_v1.py",
        "--spec", str(PROJECT_ROOT / SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec))]
    paths = set(old["source_sha256"])
    paths.update(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "src").rglob("*.py"))
    paths.update({"tools/v3/run_gse_coordinate_control_v1.py", "tools/v3/freeze_gse_coordinate_control_v1.py",
                  "tests/v3/unit/test_gse_coordinate_control_card.py", "tests/v3/unit/test_gse_coordinate_control_runner.py",
                  "tests/v3/unit/test_gse_coordinate_ablation.py"})
    spec["source_sha256"] = {path: sha(PROJECT_ROOT / path) for path in sorted(paths)}
    return card, spec


def main():
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("refuse overwrite of frozen coordinate control card/spec")
    card, spec = documents()
    write_json(PROJECT_ROOT / CARD, card); write_json(PROJECT_ROOT / SPEC, spec)
    report = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
    print(json.dumps({"passed": report.passed, "errors": report.errors, "warnings": report.warnings, "card": CARD, "spec": SPEC}))
    return int(not report.passed)


if __name__ == "__main__": raise SystemExit(main())
