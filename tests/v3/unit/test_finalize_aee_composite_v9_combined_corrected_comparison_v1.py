from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "finalize_aee_composite_v9_combined_corrected_comparison_v1",
        ROOT / "tools/v3/finalize_aee_composite_v9_combined_corrected_comparison_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _seal(module, root: Path, run: Path, paths: list[Path]) -> str:
    target = run / "artifacts/evidence_sha256.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{module.sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    ), encoding="utf-8")
    return module.sha256(target)


def _fixture(tmp_path: Path, *, bad_checkpoint_quota: bool = False):
    module = _load()
    audit = tmp_path / "audit"
    audit_paths = [
        audit / "RUN_STATE.json", audit / "metrics/summary.json",
        audit / "config/source_manifest.json", audit / "config/source_provenance.json",
        audit / "metrics/stochastic_analysis.json",
    ]
    _write(audit_paths[0], {"state": "COMPLETED", "overall_status": module.AUDIT_STATUS})
    _write(audit_paths[1], {
        "overall_status": module.AUDIT_STATUS, "source_case_count": 90, "v9_case_count": 30,
    })
    rows = [{
        "index": index,
        "case_id": f"source_{index:03d}",
        "block_id": f"block_{index // 9}",
        "method_family": ("original_mtare", "m1d_topology", "layered_gt_map_oracle")[index % 3],
        "source": "failed_source_run" if index < 69 else "recovery_run",
    } for index in range(90)]
    _write(audit_paths[2], {
        "case_count": 90, "block_count": 10,
        "family_case_counts": {
            "layered_gt_map_oracle": 30, "m1d_topology": 30, "original_mtare": 30,
        },
        "source_counts": {"failed_source_run": 69, "recovery_run": 21},
        "schedule_file_sha256": "a" * 64,
        "schedule_content_sha256": "b" * 64,
        "cases": rows,
    })
    _write(audit_paths[3], {"source_mutation_permitted": False})
    _write(audit_paths[4], {
        "case_count": 90, "compatibility_bridge": {"mapped_field_count": 90},
    })
    audit_seal = _seal(module, tmp_path, audit, audit_paths)

    corrected = tmp_path / "corrected"
    corrected_paths = [
        corrected / "RUN_STATE.json", corrected / "metrics/summary.json",
        corrected / "config/case_schedule.json", corrected / "config/paired_source.json",
        corrected / "config/readiness_source.json", corrected / "config/schedule_audit.json",
    ]
    _write(corrected_paths[0], {
        "state": "COMPLETED", "overall_status": module.CORRECTED_STATUS,
    })
    _write(corrected_paths[1], {
        "overall_status": module.CORRECTED_STATUS, "completed_case_count": 30,
    })
    schedule = []
    for block in range(10):
        for seed in range(3):
            checkpoint_seed = 0 if bad_checkpoint_quota else seed
            schedule.append({
                "index": block * 9 + seed,
                "case_id": f"corrected_{block}_{seed}",
                "block_id": f"block_{block}",
                "method_family": "m1d_topology",
                "checkpoint_seed": checkpoint_seed,
            })
    _write(corrected_paths[2], {"cases": schedule})
    for path in corrected_paths[3:]:
        _write(path, {"bound": True})
    corrected_seal = _seal(module, tmp_path, corrected, corrected_paths)

    tool = tmp_path / "tool.py"
    tool.write_text("pass\n", encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    card = tmp_path / "card.proposal.json"
    final_spec = tmp_path / "final.json"
    final_card = tmp_path / "card.json"
    _write(card, {
        "status": module.PENDING_STATUS,
        "approval": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "sources": {},
    })
    _write(proposal, {
        "status": module.PENDING_STATUS,
        "user_authorization": {"status": "APPROVED_AFTER_EXACT_SOURCE_BINDING"},
        "source_audit_run": "audit",
        "corrected_v5_run": "corrected",
        "frozen_tools": {
            "fixture": {"path": "tool.py", "sha256": module.sha256(tool)},
            "data_card_proposal": {"path": "card.proposal.json", "sha256": module.sha256(card)},
        },
    })
    return module, proposal, card, final_spec, final_card, audit_seal, corrected_seal


def test_finalizer_binds_composed_and_corrected_sources(tmp_path: Path):
    module, proposal, card, final_spec, final_card, audit_seal, corrected_seal = _fixture(tmp_path)
    module.finalize(
        proposal_spec_path=proposal, proposal_card_path=card,
        final_spec_path=final_spec, final_card_path=final_card, project_root=tmp_path,
    )
    spec = json.loads(final_spec.read_text())
    assert spec["source_audit"]["seal_sha256"] == audit_seal
    assert spec["corrected_v5_run"]["seal_sha256"] == corrected_seal
    assert spec["source_audit"]["schedule_content_sha256"] == "b" * 64
    assert spec["corrected_v5_run"]["expected_status"] == module.CORRECTED_STATUS
    assert json.loads(final_card.read_text())["approval"]["status"] == "APPROVED"


def test_finalizer_rejects_corrected_checkpoint_quota_drift(tmp_path: Path):
    module, proposal, card, final_spec, final_card, _, _ = _fixture(
        tmp_path, bad_checkpoint_quota=True
    )
    with pytest.raises(RuntimeError, match="checkpoint"):
        module.finalize(
            proposal_spec_path=proposal, proposal_card_path=card,
            final_spec_path=final_spec, final_card_path=final_card, project_root=tmp_path,
        )
    assert not final_spec.exists()
    assert not final_card.exists()
