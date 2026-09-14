#!/usr/bin/env python3
"""Freeze one non-overwriting nomination audit spec; never execute the audit."""
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_review_inventory_reader_v1 import sha_file
from mtare_topo.governance_review_nomination import validate_review_nomination_card


def main():
    card_path = Path("configs/v3/gate3/data_cards/gse_review_nomination_v1.json")
    spec_path = PROJECT_ROOT / "configs/v3/gate3/gse_review_nomination_v1.json"
    if spec_path.exists():
        raise FileExistsError("spec is immutable; do not overwrite or silently refreeze")
    card = json.loads((PROJECT_ROOT / card_path).read_text())
    report = validate_review_nomination_card(card)
    if not report.passed:
        raise ValueError("nomination card invalid: " + "; ".join(report.errors))
    template = json.loads((PROJECT_ROOT / "configs/v3/gate3/gse_review_source_inventory_v1.json").read_text())
    run = "results/gate3_semantics/gate3_20260906_gse_review_nomination_v1_seed20260906"
    command = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1", "PYTHONHASHSEED=20260906",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_review_nomination_v1.py", "--spec", str(spec_path), "--run-dir", str(PROJECT_ROOT / run)]
    source = set(template["source_sha256"])
    source -= {"configs/v3/gate3/data_cards/gse_review_source_inventory_v1.json",
               "docs/GSE_REVIEW_SOURCE_INVENTORY_V1.md", "tools/v3/run_gse_review_source_inventory_v1.py"}
    source.update({str(card_path), "docs/GSE_REVIEW_NOMINATION_POLICY_V1.md",
        "docs/GSE_REVIEW_NOMINATION_SOURCE_SCOPE_DRAFT_V1.md",
        "src/mtare_topo/governance_review_nomination.py",
        "src/mtare_topo/data/gse_review_nomination_v1.py",
        "src/mtare_topo/data/gse_review_quota_assignment_v1.py",
        "src/mtare_topo/data/gse_review_nomination_reader_v1.py",
        "tools/v3/run_gse_review_nomination_v1.py", "tools/v3/freeze_gse_review_nomination_v1.py"})
    # The historical index documents are the exact already-measured population,
    # not permission to reopen their original scan/geometry source fields.
    source.update(card["scope"]["source_scope"]["inventory_source_files"])
    for relative, expected in card["scope"]["source_scope"]["inventory_source_files"].items():
        if sha_file(PROJECT_ROOT / relative) != expected:
            raise ValueError("sealed inventory changed: " + relative)
    policy = card["scope"].get("policy")
    # The independently fixed complete card digest already binds policy content;
    # bind actual implementation/doc bytes separately in the command snapshot.
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "date": "20260906",
        "slug": "gse_review_nomination_v1", "seed": 20260906, "operation": "audit",
        "data_card": str(card_path), "config_path": str(card_path), "user_authorization": card["approval"],
        "question": "Can the fixed four-strata32fit/8calibration/8development review candidates be selected without duplicate physical edges/nodes, resplitting parents or changing21decisions?",
        "method": "Fixed boundary-node/midpoint-corridor/host-disconnected-azimuth nominations;maximum-cardinality integer quota assignment with parent coverage before seed/hash tie-break. Direct codebook/base/realized ordering check. No visibility labels.",
        "baseline": "Frozen identity inventory:55eligibleparents/918directedtraversals/459physicaledges/2546windows;original fixed8/2/2perstratum and entire70parent split. Synthetic brute-force allocator reference, no historical model score.",
        "fallback": "Source/identity/chunk/resource error fails once and seals. Quota shortage is explicit incomplete candidate evidence:stop affected export/training;no resplit,shortening,duplicate filling or retry. Fullquota is not human-label or scientificPASS.",
        "command": command, "expected_counts": {"eligible_parents": 55, "eligible_tasks": 165,
            "eligible_directed_traversals": 918, "physical_edges": 459, "overlapping_windows": 2546,
            "unique_variant_observation_rows": 62718, "unique_variant_raw_frame_identities": 73734},
        "unknown_counts": {"nominations": None, "independent_candidate_targets": None, "selected_clips": None,
            "human_confirmed_labels": None},
        "estimated_cost": {"compute": "CPU-only candidate metadata audit;330JSONs+three scopedP1b fields;0scan/model/training",
            "wall_time_hours": 1 / 6, "host_ram_gb": 4, "gpu_vram_gb": 0, "disk_gb": .25},
        "wall_time_cap_s": 600,
        "acceptance_criteria": [
            "Exact sealed eligible population and frozen full-parent partition;165construction/165codebook ordering verified;P1b3fields only,62718usedrows withchunkcollateral disclosed.",
            "No duplicate hostphysicaledge across directions/windows/strata, no duplicate targetnode acrossincidentedges;deterministic jointallocation retains maximum feasible fixedquota.",
            "Complete per-parent nominations/witnesses and exactselection/deficits/sourcehashes/chunkledger;nomination IDs/strata never student input or blind labels.",
            "CPU<=600s,host<=4GiB,evidence<=250MB;0scan/pose/model/optimizer/newteacher/humanlabels/C08-C10;no overwrite/retry or scientificPASS.",
        ],
        "expected_evidence": ["All nominations and proxy witnesses;uniquejointquotaassignment and deficits;rawsourcehash/chunkledger,configuration,environment,command,rawlog,metrics,RUN_STATE,SHA256seal."],
        "expected_versions": template["expected_versions"],
        "source_sha256": {path: sha_file(PROJECT_ROOT / path) for path in sorted(source)},
        "freeze_status": "FROZEN_ALL_AUTHORS_STOPPED"}
    with spec_path.open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"spec": str(spec_path), "spec_sha256": sha_file(spec_path),
        "scope_sha256": card["scope_sha256"], "command_created_not_executed": True}))


if __name__ == "__main__":
    main()
