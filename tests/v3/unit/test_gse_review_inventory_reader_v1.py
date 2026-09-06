"""Temporary synthetic Zarr and complete runner paths; no real source reads."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_review_inventory_reader_v1 import (
    IdentityStore, ReviewIdentityInventoryReader, collect_identity_seals, project_path, sha_file,
)
from mtare_topo.governance import ValidationReport, build_run_id
from tests.v3.unit.test_gse_review_source_index_v1 import inputs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root):
    parents = ("S01_synthetic_C01", "S01_synthetic_C07", "S02_synthetic_C07")
    scope = {"parent_ids": list(parents), "tasks": [], "source_roots": {
        role: {p: f"source/{role}/{p}" for p in ("fit", "c07")} for role in ("sensor", "teacher")},
        "source_seals": {}, "expected_counts": {"parents": 3, "tasks": 9, "logical_sequences": 102, "variant_sequences": 306}}
    for parent in parents:
        data = inputs(parent=parent)
        part = "c07" if parent.endswith("C07") else "fit"
        for variant in data["sensor_attrs_by_variant"]:
            task = parent + "__" + variant
            scope["tasks"].append({"task": task, "parent_id": parent, "partition": part, "variant": variant})
            for role in ("sensor", "teacher"):
                group = zarr.open_group(str(root / scope["source_roots"][role][part] / (task + ".zarr")), mode="w")
                group.attrs.update(data[role + "_attrs_by_variant"][variant])
                for key, array in data[role + "_arrays_by_variant"][variant].items():
                    group.create_dataset(key, data=array, chunks=array.shape, compressor=None)
                # A forbidden field exists but must never be opened or hashed.
                group.create_dataset("forbidden_scan_or_labels", data=np.ones((3, 3)))
    for role in ("sensor", "teacher"):
        seal = root / f"{role}_seal.txt"
        seal.write_text("".join(f"{sha_file(p)}  {p.relative_to(root)}\n" for p in sorted((root / "source" / role).rglob("*")) if p.is_file())
            + "f" * 64 + "  source/excluded_C08/never_resolve.zarr/.zattrs\n")
        scope["source_seals"][role] = {"path": seal.name, "sha256": sha_file(seal)}
    return scope


def reader(root, scope):
    hashes, _ = collect_identity_seals(root, scope)
    return ReviewIdentityInventoryReader(root, scope, hashes)


def test_actual_zarr_read_only_all_rows_three_variants_and_filtered_seal(tmp_path, monkeypatch):
    scope = fixture(tmp_path)
    hashes, seals = collect_identity_seals(tmp_path, scope)
    assert len(seals) == 2
    assert not any("forbidden" in p or "C08" in p for p in hashes)
    r = reader(tmp_path, scope)
    report = r.read_parent(scope["parent_ids"][0])
    assert report.counts["logical_sequences"] == 34
    assert report.counts["eligible_21_decision_windows"] == 6
    assert all(sha_file(path) == digest for path, digest in r.opened.items())
    assert not any("forbidden" in p or "C08" in p for p in r.opened)
    with pytest.raises(PermissionError): r.read_parent("S01_synthetic_C08")
    assert report.counts["structure_labels_created"] == 0


def test_drift_records_actual_failed_read_and_missing_sealed_chunk_fails(tmp_path):
    scope = fixture(tmp_path); r = reader(tmp_path, scope)
    chunk = next(Path(p) for p in r.expected if p.endswith("/global_frame_index/0")
                 and (scope["parent_ids"][0] + "__") in p)
    original = chunk.read_bytes(); chunk.write_bytes(b"x" + original[1:])
    with pytest.raises(ValueError, match="changed"): r.read_parent(scope["parent_ids"][0])
    assert r.opened[str(chunk)] == sha_file(chunk) != r.expected[str(chunk)]
    chunk.unlink()
    with pytest.raises(ValueError, match="missing"): r.read_parent(scope["parent_ids"][0])


def test_seal_drift_remains_in_failed_index_read_ledger(tmp_path):
    scope = fixture(tmp_path)
    path = tmp_path / scope["source_seals"]["sensor"]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    ledger = {}
    with pytest.raises(ValueError, match="seal index drift"): collect_identity_seals(tmp_path, scope, ledger)
    assert ledger[str(path)] == sha_file(path)


@pytest.mark.parametrize("key", ["range_m/0.0.0", "primitive_index/.zarray", "../secret", ".zmetadata", "route_arc_m/../../x"])
def test_store_rejects_unlisted_keys(tmp_path, key):
    scope = fixture(tmp_path)
    root = tmp_path / scope["source_roots"]["sensor"]["fit"] / (scope["tasks"][0]["task"] + ".zarr")
    store = IdentityStore(root, ("route_arc_m",), {}, {})
    with pytest.raises(PermissionError): store[key]


def test_symlink_escape_rejected_without_reading_target(tmp_path):
    other = tmp_path / "other"; other.mkdir()
    link = tmp_path / "link"; link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"): project_path(tmp_path, "link/item")


def load_runner(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools/v3"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("review_inventory_runner_test", tools / "run_gse_review_source_inventory_v1.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The real70-parent card validator has its own tests. These three synthetic
    # parents exercise executor I/O and failure paths, not production approval.
    monkeypatch.setattr(module, "validate_identity_inventory_card", lambda _: ValidationReport(True, ()))
    return module


def prepared_run(root, scope):
    card_path = "config/card.json"; card = {"scope": scope}
    write(root / card_path, card)
    source = root / "synthetic_source.py"; source.write_text("# synthetic frozen source\n")
    spec = {"gate": 3, "date": "20260906", "slug": "synthetic_review_inventory", "seed": 0, "operation": "audit",
        "data_card": card_path, "source_sha256": {source.name: sha_file(source)},
        "expected_versions": {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__},
        "command": ["synthetic", "no actual subprocess"]}
    run = root / "results/gate3_semantics" / build_run_id(spec)
    for part in ("artifacts", "logs", "metrics", "config"): (run / part).mkdir(parents=True)
    write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    write(run / "config/run_spec.json", spec); write(run / "config/data_card.json", card)
    (run / "config/command.txt").write_text(shlex.join(spec["command"]) + "\n")
    return spec, run


def test_end_to_end_synthetic_inventory_seal_and_no_retry(tmp_path, monkeypatch):
    scope = fixture(tmp_path); spec, run = prepared_run(tmp_path, scope)
    executor = load_runner(monkeypatch)
    assert executor.execute(spec, run, tmp_path) == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["completed_parents"] == 3 and summary["error"] is None
    assert summary["result"]["variant_sequences"] == 306
    assert summary["result"]["selected_segments"] == summary["result"]["human_labels"] == 0
    assert summary["result"]["zero_frame_traversals_audited"] is False
    assert len(json.loads((run / "artifacts/parent_split.json").read_text())) == 3
    for line in (run / "artifacts/evidence_sha256.txt").read_text().splitlines():
        digest, relative = line.split(None, 1)
        assert sha_file(tmp_path / relative) == digest
    with pytest.raises(ValueError, match="fresh"): executor.execute(spec, run, tmp_path)


def test_population_mismatch_is_failed_with_evidence_not_reselected(tmp_path, monkeypatch):
    scope = fixture(tmp_path); scope["expected_counts"]["logical_sequences"] += 1
    spec, run = prepared_run(tmp_path, scope)
    assert load_runner(monkeypatch).execute(spec, run, tmp_path) == 1
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert "population differs" in summary["error"]
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert (run / "artifacts/source_reads_sha256.json").is_file()
