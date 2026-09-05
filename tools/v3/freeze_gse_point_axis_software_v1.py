#!/usr/bin/env python3
"""Freeze the synthetic-only readout implementation and all source hashes."""
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_gse_composition_software_contract_v1 import sha

SLUG = "gse_point_axis_software_v1"
SPEC = PROJECT_ROOT / f"configs/v3/gate3/{SLUG}.json"


def main():
    if SPEC.exists():
        raise RuntimeError("refuse overwriting frozen spec")
    spec = load_json(PROJECT_ROOT / "configs/v3/gate3/gse_composition_software_contract_v1.json")
    spec.update({"slug": SLUG,
        "question": "Can a thin point-distinguishing slot-conditioned surface-to-axis readout remove the proven mean/raw convex-pooling limitation while preserving causal backbone isolation and bounded resources?",
        "method": "Synthetic CPU software verification of point-level slot/background support and low-rank slot-conditioned surface-to-axis offsets; existing backbone retained in evaluation/no-gradient mode. No dataset, checkpoint, optimizer or teacher generation.",
        "baseline": "Existing mean-coordinate convex readout and raw-coordinate no-offset ablation; oracle synthetic weights only demonstrate expressivity, not trained performance.",
        "fallback": "Stop on failed software, invariance, gradient, isolation or resource check; no actual-data training or scientific threshold adjustment.",
        "expected_counts": {"tests":48, "dataset_frames":0, "dataset_worlds":0, "optimizer_steps":0, "checkpoint_reads":0, "mtare_runs":0},
        "tests": [f"tests/v3/unit/{name}.py" for name in (
            "test_gse_point_axis_readout", "test_gse_point_axis_head",
            "test_gse_coordinate_pooling_limit", "test_gse_raw_coordinate_adapter_limit")],
        "acceptance_criteria": [
            "Exactly48 tests pass, zero failure/error/skip, unchanged frozen sources and environment.",
            "Point support distinguishes same-token layers; offset votes represent axes outside surface hull with finite nonzero gradients.",
            "Multi-source return can vote for different slot axes; no GT grouping enters the deployment adapter.",
            "Point/slot permutations preserve outputs; SE3 of vote operator with vector offsets preserves geometry (not a neural invariance claim).",
            "Invalid NaN padding cannot contaminate values/gradients; all-invalid and valid nonfinite input rejected; degenerate chord flagged without deleting controls.",
            "Full1x5x16x720=57600-point/32-slot random-backbone forward/backward; old parameter values unchanged and no backbone gradients; all thin-head gradients finite.",
            "CPU child <=120s/4GiB; no dataset/checkpoint/optimizer/test-world/graph/closed-loop and no scientific Gate promotion."],
        "expected_evidence": ["Spec/source hashes, versions, raw pytest log, JUnit, peak child RSS, summary, RUN_STATE and SHA256 seal."]})
    spec["user_authorization"].update({
        "scope": "User-approved same-plan frontend correction after measured coordinate-pooling failure; synthetic software only, no new training or teacher.",
        "confirmation_reference": "User GSE-Graph implementation plan and standing in-scope execution authorization; docs/PLAN.md 2026-09-05 16:26 permits thin readout software readiness."})
    spec["command"] = ["env", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1", "PYTHONHASHSEED=0",
        spec["command"][0], "tools/v3/run_gse_point_axis_software_v1.py", "--spec", str(SPEC), "--run-dir",
        str(PROJECT_ROOT / f"results/gate3_semantics/gate3_20260905_{SLUG}_seed0")]
    paths = set(spec["source_sha256"]) | set(spec["tests"])
    paths |= {str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "src").rglob("*.py")}
    paths |= {"src/mtare_topo/representation/gse_point_axis_readout.py",
              "tools/v3/run_gse_point_axis_software_v1.py", "tools/v3/freeze_gse_point_axis_software_v1.py"}
    spec["source_sha256"] = {p:sha(PROJECT_ROOT / p) for p in sorted(paths)}
    write_json(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
