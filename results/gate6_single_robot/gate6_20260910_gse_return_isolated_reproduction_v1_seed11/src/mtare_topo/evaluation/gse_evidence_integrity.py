"""Complete source-run seal verification for formal GSE downstream evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mtare_topo.governance import load_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_complete_run_seal(
    project_root: Path,
    run: Path,
    expected_status: str,
) -> dict[str, Any]:
    """Require an identity-correct, leakage-free run and an exact full-file seal."""

    root = project_root.resolve()
    run = run.resolve()
    run.relative_to(root)
    state_path = run / "RUN_STATE.json"
    summary_path = run / "metrics/summary.json"
    seal = run / "artifacts/evidence_sha256.txt"
    state = load_json(state_path)
    summary = load_json(summary_path)
    if (
        state.get("run_id") != run.name
        or state.get("state") != "COMPLETED"
        or state.get("overall_status") != expected_status
        or summary.get("overall_status") != expected_status
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError(f"required source is not completed leakage-free PASS: {run.name}")
    return _verify_exact_run_seal(root, run, seal, state_path, summary_path, expected_status)


def _verify_exact_run_seal(
    root: Path,
    run: Path,
    seal: Path,
    state_path: Path,
    summary_path: Path,
    expected_status: str,
) -> dict[str, Any]:
    if not seal.is_file():
        raise RuntimeError(f"source run has no evidence seal: {run.name}")

    sealed_paths: set[Path] = set()
    lines = seal.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"source evidence seal is empty: {run.name}")
    for line in lines:
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"malformed source evidence seal line: {run.name}") from exc
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeError(f"invalid source evidence digest: {run.name}")
        path = (root / relative).resolve()
        try:
            path.relative_to(run)
        except ValueError as exc:
            raise RuntimeError(f"source evidence path escapes its run: {relative}") from exc
        if path == seal.resolve() or path in sealed_paths:
            raise RuntimeError(f"source evidence seal has a duplicate or self entry: {relative}")
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source seal drift: {relative}")
        sealed_paths.add(path)

    actual_paths = {
        path.resolve()
        for path in run.rglob("*")
        if path.is_file() and path.resolve() != seal.resolve()
    }
    if sealed_paths != actual_paths:
        missing = sorted(str(path.relative_to(run)) for path in actual_paths - sealed_paths)
        extra = sorted(str(path.relative_to(run)) for path in sealed_paths - actual_paths)
        raise RuntimeError(
            f"source evidence seal does not exactly cover its run: {run.name}; missing={missing[:5]}; extra={extra[:5]}"
        )
    required = {state_path.resolve(), summary_path.resolve()}
    if not required.issubset(sealed_paths):
        raise RuntimeError(f"source evidence seal omits required state or summary: {run.name}")
    return {
        "run": str(run.relative_to(root)),
        "status": expected_status,
        "entries": len(sealed_paths),
        "seal_sha256": _sha256(seal),
        "exact_full_file_coverage": True,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


def verify_failed_component_run_seal(
    project_root: Path,
    run: Path,
    expected_status: str,
) -> dict[str, Any]:
    """Verify an immutable failed run whose explicitly qualified components are reused.

    This does not turn the failed run into a PASS.  A downstream completed PASS
    must separately identify which components remain valid and supply the
    corrective evidence for the failed component.
    """

    root = project_root.resolve()
    run = run.resolve()
    run.relative_to(root)
    state_path = run / "RUN_STATE.json"
    summary_path = run / "metrics/summary.json"
    seal = run / "artifacts/evidence_sha256.txt"
    state = load_json(state_path)
    summary = load_json(summary_path)
    strict_reads = summary.get("strict_test_worlds_read", summary.get("c10_worlds_read"))
    if (
        state.get("run_id") != run.name
        or state.get("state") != "FAILED"
        or state.get("overall_status") != expected_status
        or summary.get("overall_status") != expected_status
        or strict_reads != 0
        or summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError(f"required failed component source is not immutable and leakage-free: {run.name}")
    result = _verify_exact_run_seal(root, run, seal, state_path, summary_path, expected_status)
    result["component_source_only"] = True
    result["run_state"] = "FAILED"
    return result


__all__ = ["verify_complete_run_seal", "verify_failed_component_run_seal"]
