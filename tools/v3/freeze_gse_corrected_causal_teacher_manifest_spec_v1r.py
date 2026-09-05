#!/usr/bin/env python3
"""Freeze the sparse-index-preserving V1R Data Card and run spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


V1_CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_corrected_causal_teacher_manifest_v1.json"
V1_SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_corrected_causal_teacher_manifest_v1.json"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_corrected_causal_teacher_manifest_v1r.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate2/gse_corrected_causal_teacher_manifest_v1r.json"
RUN_ID = "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: str) -> dict[str, str]:
    return {"path": path, "sha256": _sha256(PROJECT_ROOT / path)}


def main() -> int:
    card = deepcopy(load_json(V1_CARD))
    card.update(
        {
            "card_id": "gse_corrected_causal_teacher_manifest_v1r",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_SPARSE_INDEX_PRESERVING_REPLACEMENT",
            "purpose": "Generate the unchanged corrected C01-C08 causal Teacher while preserving every immutable sparse complete-export global_sequence_index exactly; V1 remains a sealed checker failure.",
        }
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T13:00:00+08:00",
        "authorized_operations": ["teacher_generation"],
        "authorized_gates": [2],
        "scope": "One V1R replacement over the unchanged 80 C01-C08 worlds, 16078 traversals, 188126 observations, 1031 labels, 76 change-point identities and 74064 pairs. Preserve sparse global IDs exactly; zero C09/C10/M-TARE/model/training.",
        "confirmation_reference": "User instructed Codex to stop requesting routine choices and automatically select the evidence-supported optimal option; option A was the documented recommendation for this isolated checker defect.",
    }
    card["source"]["failed_v1_run"] = (
        "results/gate2_representation/"
        "gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0"
    )
    card["sampling"]["global_identity_mapping"] = (
        "global_sequence_index is an immutable identity in the complete C01-C10 export, not a compact row. "
        "The C01-C08 subset must preserve exactly 188126 unique strictly increasing source IDs, minimum 0, "
        "maximum 208227, nine internal gap blocks and 20102 omitted C09/C10 slots. Array consumers create an explicit compact mapping."
    )
    card["split"]["replacement_boundary"] = (
        "V1 failed only because its checker incorrectly required contiguous 0..188125 IDs. V1R changes no observation, "
        "label, identity, pair, split, threshold or model rule; only the acceptance expression is corrected."
    )
    card["evidence"]["machine_metrics"] = (
        "All V1 exact counts plus source/output global-index digests, exact source order, uniqueness, strict monotonicity, "
        "minimum/maximum, internal omitted count and gap-block count; source verification, logs and complete seal."
    )
    card["evidence"]["failure_policy"] = (
        "Any source/tool/card/environment/count/schema/identity/priority/pair/index digest or immutable-field drift seals FAIL. "
        "V1 remains immutable; do not renumber IDs, reuse V1 artifacts or retry V1R."
    )
    write_json(CARD_PATH, card)

    spec = deepcopy(load_json(V1_SPEC))
    spec.update(
        {
            "date": "20260827",
            "slug": "gse_corrected_causal_teacher_manifest_v1r",
            "question": "Can the corrected C01-C08 causal Teacher be regenerated with every sparse complete-export global identity preserved exactly, while all V1 scientific counts remain unchanged?",
            "method": "Regenerate from the same sealed old Teacher and causal proof. Preserve every non-event field and sparse global_sequence_index byte-value; require exact source order/digest, uniqueness, strict monotonicity, min=0, max=208227, nine gap blocks and 20102 omitted internal C09/C10 slots. Apply the unchanged 1031 labels/76 identities and regenerate the unchanged 74064 pairs.",
            "baseline": "Immutable V1 checker failure: all scientific quantities passed but a wrong contiguous 0..188125 assertion rejected valid sparse complete-export IDs.",
            "fallback": "Seal FAIL and stop. Never renumber global identities, reuse V1 output, alter labels/pairs or adapt training around an index mismatch.",
            "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
            "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
            "command": [
                "/usr/bin/timeout",
                "--signal=INT",
                "--kill-after=60s",
                "600s",
                "/home/zeng-workstation/anaconda3/bin/python",
                "tools/v3/run_gse_corrected_causal_teacher_manifest_v1r.py",
                "--spec",
                str(SPEC_PATH),
                "--run-dir",
                str(PROJECT_ROOT / "results/gate2_representation" / RUN_ID),
            ],
        }
    )
    spec["user_authorization"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T13:00:00+08:00",
        "scope": card["approval"]["scope"],
        "confirmation_reference": card["approval"]["confirmation_reference"],
    }
    spec["acceptance_criteria"][0] = (
        "Exact 80 worlds, 16078 traversals and 188126 observations; global_sequence_index equals the sealed source order/digest, "
        "is unique and strictly increasing with min/max 0/208227, nine gap blocks and 20102 legal omitted internal slots; no non-event field changes."
    )
    spec["stop_conditions"] = [
        "Any source/tool/card/environment/count/schema/identity/priority/pair/index/order/digest drift or non-event field change.",
        "Any C09/C10/M-TARE/model/training access, timeout, disk overrun, V1 artifact reuse, ID renumbering, retry or old-run overwrite.",
    ]
    spec["expected_evidence"].append(
        "Typed sparse-global-index audit with exact source/output SHA-256, order, uniqueness, monotonicity, range and legal-gap metrics."
    )
    tool_paths = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "runner": "tools/v3/run_gse_corrected_causal_teacher_manifest_v1r.py",
        "runner_base": "tools/v3/run_gse_corrected_causal_teacher_manifest_v1.py",
        "executor": "tools/v3/execute_gse_corrected_causal_teacher_manifest_v1r.py",
        "manifest_builder": "src/mtare_topo/data/gse_corrected_teacher_manifest.py",
        "association_teacher": "src/mtare_topo/teacher/gse_association_teacher.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "freezer": "tools/v3/freeze_gse_corrected_causal_teacher_manifest_spec_v1r.py",
    }
    spec["frozen_tools"] = {name: _record(path) for name, path in tool_paths.items()}
    input_paths = tuple(spec["frozen_inputs"]) + (
        "configs/v3/gate2/data_cards/gse_corrected_causal_teacher_manifest_v1.json",
        "configs/v3/gate2/gse_corrected_causal_teacher_manifest_v1.json",
        "results/gate2_representation/gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0/RUN_STATE.json",
        "results/gate2_representation/gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0/metrics/summary.json",
        "results/gate2_representation/gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0/artifacts/evidence_sha256.txt",
    )
    spec["frozen_inputs"] = {
        path: _sha256(PROJECT_ROOT / path) for path in dict.fromkeys(input_paths)
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH)
    print(SPEC_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
