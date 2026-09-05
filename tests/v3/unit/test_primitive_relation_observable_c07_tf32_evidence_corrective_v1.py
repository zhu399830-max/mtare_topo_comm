from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.v3.run_primitive_relation_observable_c07_tf32_evidence_corrective_v1 import (
    source_bug_signature,
    verify_source_seal,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_source_seal_accepts_exact_scoped_population(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "artifacts").mkdir(parents=True)
    payload = source / "metrics.json"
    payload.write_text("{}\n", encoding="utf-8")
    (source / "artifacts/evidence_sha256.txt").write_text(
        f"{_sha(payload)}  source/metrics.json\n", encoding="utf-8"
    )
    entries = verify_source_seal(source, tmp_path, 1)
    assert list(entries) == ["source/metrics.json"]


def test_verify_source_seal_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "artifacts").mkdir(parents=True)
    payload = source / "metrics.json"
    payload.write_text("{}\n", encoding="utf-8")
    (source / "artifacts/evidence_sha256.txt").write_text(
        f"{'0' * 64}  source/metrics.json\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source seal mismatch"):
        verify_source_seal(source, tmp_path, 1)


def test_source_bug_signature_requires_seal_and_both_len_calls(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text(
        'entries = _seal(run)\n{"entries": len(entries)}\n{"evidence_entries": len(entries)}\n',
        encoding="utf-8",
    )
    assert source_bug_signature(runner)
    runner.write_text("entries = _seal(run)\n", encoding="utf-8")
    assert not source_bug_signature(runner)
