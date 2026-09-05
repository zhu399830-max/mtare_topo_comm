#!/usr/bin/env python3
"""Prepare a bounded factorial from seal metadata; never decode payloads or run."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json, preflight, write_json
from mtare_topo.governance_class_balance_factorial import (
    SCHEMA, TRAINING, EVALUATION, BALANCE_POLICY, RANK_POLICY, RESOURCES, RESTRICTIONS,
    CORRECTIVE_SEAL_SHA256, RANK_SEAL_SHA256, validate_class_balance_factorial_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS, EXPORT_SEAL_SHA256
from freeze_gse_partial_structure_export_v1 import selected_seal_entries

SLUG = "gse_class_balance_factorial_v1"
CARD = f"configs/v3/gate3/data_cards/{SLUG}.json"
SPEC = f"configs/v3/gate3/{SLUG}.json"
CORRECTIVE_SPEC = "configs/v3/gate3/gse_geometry_bound_training_v1.json"
RANK_SPEC = "configs/v3/gate3/gse_geometry_bound_rank_v1r.json"
METHOD_DOC = "docs/GSE_CLASS_BALANCE_FACTORIAL_V1.md"
UNCHANGED_SOURCES = (
    "src/mtare_topo/representation/gse_region_queries.py",
    "src/mtare_topo/representation/gse_geometry_bound_losses.py",
    "src/mtare_topo/representation/gse_geometry_bound_training.py",
    "src/mtare_topo/representation/gse_partial_structure_training.py",
    "src/mtare_topo/evaluation/gse_partial_structure.py",
    "src/mtare_topo/data/gse_partial_training_inputs.py",
)
TOOL_SOURCES = (
    "tools/v3/run_gse_class_balance_factorial_v1.py", "tools/v3/freeze_gse_class_balance_factorial_v1.py",
    "tools/v3/freeze_gse_partial_structure_export_v1.py", "tools/v3/run_gse_geometry_bound_training_v1.py",
    "tools/v3/run_gse_geometry_bound_rank_v1.py", "tools/v3/run_gse_partial_structure_training_v1.py",
    "tools/v3/run_gse_supported_construction_teacher_v1.py", "tools/v3/run_gse_assignment_attribution_v1.py",
    "tools/v3/_bootstrap.py",
    "tests/v3/unit/test_gse_partial_structure_training_runner.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    corrective = load_json(PROJECT_ROOT / CORRECTIVE_SPEC)
    old = load_json(PROJECT_ROOT / corrective["data_card"])
    rank = load_json(PROJECT_ROOT / RANK_SPEC)
    sources = deepcopy(old["sources"])
    bound, references = {}, {}
    initial_digest = None
    for name, specpath, spec, digest, key in (
        ("corrective_reference", CORRECTIVE_SPEC, corrective, CORRECTIVE_SEAL_SHA256, "baseline_geometry_summary"),
        ("rank_reference", RANK_SPEC, rank, RANK_SEAL_SHA256, "baseline_rank_summary"),
    ):
        root = "results/gate3_semantics/" + build_run_id(spec)
        seal = root + "/artifacts/evidence_sha256.txt"
        payload = root + "/metrics/summary.json"
        wanted = {payload, root + "/config/run_spec.json", root + "/config/data_card.json"}
        if name == "corrective_reference":
            wanted.add(root + "/artifacts/shared_initial_state.pt")
        entries = selected_seal_entries(PROJECT_ROOT / seal, wanted, digest)
        if name == "corrective_reference":
            initial_digest = entries[root + "/artifacts/shared_initial_state.pt"]
        sources[key] = {"path": payload, "sha256": entries[payload]}
        bound[payload], bound[seal] = entries[payload], digest
        for path, snapshot in ((specpath, "run_spec.json"), (spec["data_card"], "data_card.json")):
            actual = sha(PROJECT_ROOT / path)
            if actual != entries[root + "/config/" + snapshot]:
                raise ValueError("completed source spec/card snapshot drift")
            bound[path] = actual
        references[name] = {"spec": specpath, "card": spec["data_card"], "seal": seal}
    immutable = {path: sha(PROJECT_ROOT / path) for path in UNCHANGED_SOURCES}
    if any(corrective.get("source_sha256", {}).get(path) != digest for path, digest in immutable.items()):
        raise ValueError("original head/geometry loss/training/evaluation/loader source drift")
    exportseal = old["export_reference"]["seal"]
    exported = selected_seal_entries(PROJECT_ROOT / exportseal, {v["path"] for v in old["sources"].values()}, EXPORT_SEAL_SHA256)
    bound[exportseal] = EXPORT_SEAL_SHA256
    for source in old["sources"].values():
        if exported[source["path"]] != source["sha256"]:
            raise ValueError("original four exported source hashes changed")
        bound[source["path"]] = source["sha256"]
    bound[METHOD_DOC] = sha(PROJECT_ROOT / METHOD_DOC)
    purpose = "Disentangle member and event class balance in a fixed-budget 2x2 factorial; reuse sealed00, train10/01/11 on each of three original cached-head branches, no thresholds or best-factor selection."
    card = {key: deepcopy(old[key]) for key in (*IDENTITY_FIELDS, "effective_counts", "export_reference")}
    card.update({
        "schema_version": SCHEMA, "operation": "training", "card_id": SLUG, "purpose": purpose,
        "teacher_source": "Only four unchanged sealed C01 partial training inputs. Two completed summary payloads are read-only comparison, not training targets; baseline initial-state digest is seal metadata only, no old weight bytes read.",
        "scope_limitations": "Exact180 fit observations only, incomplete events872/14/0 and partial members2152/13489. Report all factors and independent fixed0.5/argmax plus ranking together; no full classification/detection/generalization/scientific PASS.",
        "base_card": deepcopy(old), "sources": sources, "sealed_sources": bound, **references,
        "baseline_initial_state_sha256": initial_digest,
        "training": deepcopy(TRAINING), "evaluation": deepcopy(EVALUATION), "balance_policy": deepcopy(BALANCE_POLICY),
        "rank_policy": deepcopy(RANK_POLICY), "resources": dict(RESOURCES), "restrictions": dict.fromkeys(RESTRICTIONS, True),
        "capacity_ready": False, "scientific_gate_pass": False,
        "approval": {"status": "APPROVED", "approved_by": "user", "approved_at": "2026-09-05", "scope": purpose,
            "authorized_operations": ["training"], "authorized_gates": [3], "selection_sha256": old["selection_sha256"],
            "sources": deepcopy(sources), "balance_policy": deepcopy(BALANCE_POLICY), "training": deepcopy(TRAINING),
            "baseline_initial_state_sha256": initial_digest,
            "confirmation_reference": "The user-approved GSE-Graph plan and standing scope-bound authorization cover the publicly selected member/event balance factorial after the sealed final-rank diagnosis. No new approval conversation is fabricated; this card independently binds six payloads, fixed weights, all three factors and2700updates. Old training/data_export permissions are not reused."},
    })
    report = validate_class_balance_factorial_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260905", "slug": SLUG, "seed": 0,
        "operation": "training", "data_card": CARD, "config_path": CARD, "user_authorization": deepcopy(card["approval"]),
        "question": "Do member and event class balancing separately improve fixed-decision partial structural fitting, given strong member ranking but weak event ranking?",
        "method": "2x2 factors00reused,10/01/11 each GT/pred/no-relations,seed0/sameinitial/order/Adam.001/B18/300steps. GlobalN/(K*nclass) per-label weights, original denominators, geometry binding and all other losses unchanged.",
        "baseline": "Sealed geometry-bound00 and its completed rankV1R summaries only; no old weights/predictions/history read. Same predicted axes without explicit relations is paired ablation,GT is partial reference upper input.",
        "fallback": "Source/population/initialization/schedule/nonfinite/empty-supervision/resource drift FAIL and seal. Report all factors without extra epochs,threshold search or choosing best factor; no scientific PASS.",
        "training": deepcopy(TRAINING), "evaluation": deepcopy(EVALUATION), "balance_policy": deepcopy(BALANCE_POLICY),
        "rank_policy": deepcopy(RANK_POLICY), "wall_time_cap_s": 1800, "immutable_baseline_source_sha256": immutable,
        "expected_counts": {"observations": 180, "parents": 10, "unique_frames": 900, "visible_fragments": 1452,
            "centers_per_branch": 1206, "events_per_branch": 886, "positive_members_per_branch": 2152,
            "negative_members_per_branch": 13489, "optimizer_steps": 2700, "head_inference_windows": 3240,
            "backbone_windows": 0, "new_sensor_frames": 0, "checkpoint_reads": 0, "xy_svg_figures": 180, "rank_svg_figures": 30},
        "estimated_cost": {"compute": "Sequential CUDA0 nine fits;2700updates3240initial/final eval; reference176.8seconds plus output,cap1800seconds", "wall_time_hours": .5, "host_ram_gb": 4, "gpu_vram_gb": 4, "disk_gb": .5},
        "acceptance_criteria": [
            "Exact six sealed sources,original180C01/900frames/1452fragments/1206centers/886partialevents/2152positive13489negative unchanged;00 is evidence only.",
            "Only two declared loss balance factors change;fixed global weights and original denominators,geometry-only unique binding;all nine fits same initialization/order/budget and fixedfinal300.",
            "All factors and parents reported with original0.5/three-class argmax,partial two-known-class metrics plus final AP/AUROC/BCE;GTmemberF1.90 and two-known-eventmacro.95 are tiny-capacity targets only,not full sciencePASS.",
            "2700updates3240evaluationwindows,0backbone/newscan/oldweights;1800s4GiBhost/GPU.5GB;full210SVG and weights/predictions/logs/config/env/state/seal,no overwrite/retry."],
        "expected_evidence": ["All nine initial/final heads,predictions,independent scores and final ranks;00comparison and factor interactions;180XY/XZ and30rankSVG;sharedinitial SHA/schedule,loss components,supervision,rawlogs,environment,RUN_STATE and SHAseal."],
        "expected_versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "zarr", "scipy")}},
    }
    spec["command"] = ["env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONHASHSEED=0", corrective["command"][6],
        "tools/v3/run_gse_class_balance_factorial_v1.py", "--spec", str(PROJECT_ROOT / SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec))]
    spec["command_sha256"] = hashlib.sha256(json.dumps(spec["command"], separators=(",", ":")).encode()).hexdigest()
    paths = {str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "src").rglob("*.py")}
    paths.update(TOOL_SOURCES)
    for pattern in ("test_gse_class_balance*.py", "test_gse_geometry_bound*.py", "test_gse_binary_ranking.py", "test_gse_assignment_runner.py"):
        paths.update(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "tests/v3/unit").glob(pattern))
    spec["source_sha256"] = {path: sha(PROJECT_ROOT / path) for path in sorted(paths)}
    return card, spec


def main():
    if (PROJECT_ROOT / CARD).exists() or (PROJECT_ROOT / SPEC).exists():
        raise RuntimeError("refuse overwrite of factorial card/spec")
    card, spec = documents()
    write_json(PROJECT_ROOT / CARD, card)
    write_json(PROJECT_ROOT / SPEC, spec)
    report = preflight(spec, load_json(PROJECT_ROOT / "results/project_status.json"), PROJECT_ROOT)
    print(json.dumps({"passed": report.passed, "errors": report.errors, "warnings": report.warnings, "card": CARD, "spec": SPEC}))
    return int(not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
