"""Metadata selection and real executor tests using only synthetic Zarr shards."""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_head_development_selection import FIELDS, HeadDevelopmentMetadataReader, quantile_rows
from mtare_topo.governance import build_run_id
from tests.v3.unit.test_gse_head_development_card import card


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(tmp_path, issue=None):
    value = card()
    root = tmp_path / value["source_roots"]["teacher"]
    for task in value["tasks"]:
        group = zarr.open_group(str(root / (task + ".zarr")), mode="w")
        group.attrs.update(parent_id=task.split("__")[0], partition="fit", geometry_realization="c1_mixed",
                           maximum_slots=32, window_frames=5, student_identity_input_forbidden=True,
                           traversal_ids=[f"traversal_{row // 9}" for row in range(36)])
        frames = np.arange(36)[:, None] + np.arange(5)[None]
        source = np.arange(100, 136)
        mask = (np.arange(32)[None] < (1 + np.arange(36)[:, None] % 3)).astype(np.int32)
        if issue == "shape":
            frames = frames[:, :4]
        elif issue == "float":
            frames = frames.astype(float)
        elif issue == "causal":
            frames[1, 1] = frames[1, 0]
        elif issue == "negative_frame":
            frames[1] -= 20
        elif issue == "duplicate_source":
            source[3] = source[1]
        elif issue == "mask_empty":
            mask[1] = 0
        elif issue == "mask_nonbinary":
            mask[1, 0] = 2
        elif issue == "mask_negative":
            mask[1, 2] = -1  # Positive total remains; signed masks must still be binary.
        for name, data in (("frame_row", frames), ("source_global_sequence_index", source), ("primitive_mask", mask)):
            if issue == "missing_field" and name == "primitive_mask":
                continue
            group.create_dataset(name, data=data, chunks=(6,) + data.shape[1:])
        # Present but forbidden fields ensure the reader has something it must not open.
        group.create_dataset("axis_control_current_sensor_m", data=np.full((36, 32, 3, 3), np.nan))
        group.create_dataset("range_m", data=np.full((36, 4), np.nan))
    expected = {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}
    return value, root, expected


@pytest.mark.parametrize("count", [18, 19, 35, 36, 37, 1001])
def test_fixed_quantile_indices_are_exact_unique_and_in_bounds(count):
    rows = quantile_rows(count)
    assert rows == [((2 * k + 1) * count) // 36 for k in range(18)]
    assert len(rows) == len(set(rows)) == 18
    assert rows == sorted(rows) and 0 <= rows[0] <= rows[-1] < count


@pytest.mark.parametrize("count", [0, 17, -1, 18., True, None])
def test_short_or_noninteger_population_rejected(count):
    with pytest.raises(ValueError):
        quantile_rows(count)


@pytest.mark.parametrize("issue", ["other_split", "duplicate", "nine", "eleven", "order", "variant", "escape"])
def test_exact_c02_allowlist_rejected_before_io(issue):
    tasks = card()["tasks"]
    if issue == "other_split": tasks[0] = tasks[0].replace("C02", "C07")
    elif issue == "duplicate": tasks[1] = tasks[0]
    elif issue == "nine": tasks.pop()
    elif issue == "eleven": tasks.append("S11_synthetic_C02__c1_mixed")
    elif issue == "order": tasks.reverse()
    elif issue == "variant": tasks[0] = tasks[0].replace("c1_mixed", "c0_baseline")
    else: tasks[0] = "../escape"
    with pytest.raises(ValueError):
        HeadDevelopmentMetadataReader("/no-synthetic-root-exists", tasks, {})


def test_only_three_metadata_fields_opened_and_nonregistered_task_refused(tmp_path):
    value, root, expected = _fixture(tmp_path)
    reader = HeadDevelopmentMetadataReader(root, value["tasks"], expected)
    selected, summary = reader.read_task(value["tasks"][0])
    assert [row["row_index"] for row in selected] == list(range(1, 36, 2))
    assert summary["unique_source_frames"] == 39
    assert summary["visible_fragments"] == 36
    assert summary["selected_traversals"] == 4
    for path in reader.opened:
        key = Path(path).relative_to(root / (value["tasks"][0] + ".zarr")).parts[0]
        assert key in FIELDS | {".zgroup", ".zattrs"}
    with pytest.raises(PermissionError):
        reader.read_task(value["tasks"][0].replace("C02", "C01"))


@pytest.mark.parametrize("issue", ["shape", "float", "causal", "negative_frame", "duplicate_source",
                                   "mask_empty", "mask_nonbinary", "mask_negative", "missing_field"])
def test_invalid_metadata_rejected(tmp_path, issue):
    value, root, expected = _fixture(tmp_path, issue)
    reader = HeadDevelopmentMetadataReader(root, value["tasks"], expected)
    with pytest.raises((ValueError, KeyError)):
        reader.read_task(value["tasks"][0])


@pytest.mark.parametrize("issue", ["tamper", "missing_chunk", "unsealed"])
def test_exact_sealed_chunks_required(tmp_path, issue):
    value, root, expected = _fixture(tmp_path)
    chunk = root / (value["tasks"][0] + ".zarr") / "frame_row/0.0"
    assert chunk.is_file()
    if issue == "tamper": chunk.write_bytes(chunk.read_bytes() + b"tamper")
    elif issue == "missing_chunk": chunk.unlink()
    else: expected.pop(str(chunk))
    with pytest.raises(ValueError):
        HeadDevelopmentMetadataReader(root, value["tasks"], expected).read_task(value["tasks"][0])


def test_labels_and_mask_values_do_not_choose_observation_rows(tmp_path):
    value, root, expected = _fixture(tmp_path)
    task = value["tasks"][0]
    first, _ = HeadDevelopmentMetadataReader(root, value["tasks"], expected).read_task(task)
    group = zarr.open_group(str(root / (task + ".zarr")), mode="a")
    group["axis_control_current_sensor_m"][:] = 1.e6
    group["range_m"][:] = -999
    group["primitive_mask"][:] = 1
    # New fixture bytes receive a new seal; this is not an input-drift bypass.
    updated = {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}
    second, _ = HeadDevelopmentMetadataReader(root, value["tasks"], updated).read_task(task)
    assert [r["row_index"] for r in first] == [r["row_index"] for r in second]
    assert [r["source_global_sequence_index"] for r in first] == [r["source_global_sequence_index"] for r in second]
    assert {r["visible_fragments"] for r in second} == {32}


def test_real_runner_freezes180_rows_actual_population_and_seal(tmp_path, monkeypatch):
    value, root, expected = _fixture(tmp_path)
    seal_path = tmp_path / value["source_seals"]["teacher"]
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text("".join(f"{digest}  {Path(path).relative_to(tmp_path)}\n" for path, digest in sorted(expected.items())))
    value["sealed_sources"] = {str(seal_path.relative_to(tmp_path)): _sha(seal_path)}
    card_path = tmp_path / "configs/synthetic_card.json"
    _write(card_path, value)
    source = tmp_path / "synthetic_source.txt"
    source.write_text("Synthetic provenance only, not executed")
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_head_metadata", "seed": 0,
            "operation": "audit", "data_card": str(card_path.relative_to(tmp_path)),
            "source_sha256": {source.name: _sha(source)},
            "expected_versions": {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}}
    spec_path = tmp_path / "configs/synthetic_spec.json"
    _write(spec_path, spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for directory in ("config", "logs", "metrics", "previews", "artifacts"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    _write(run / "config/run_spec.json", spec)
    _write(run / "config/data_card.json", value)
    _write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    before = {Path(path): digest for path, digest in expected.items()}
    before.update({path: _sha(path) for path in (seal_path, source, card_path, spec_path)})
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_head_development_metadata_v1")

    def contained(relative):
        path = (tmp_path / relative).resolve()
        path.relative_to(tmp_path)
        return path

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "contained", contained)
    monkeypatch.setattr(sys, "argv", ["synthetic-run", "--spec", str(spec_path), "--run-dir", str(run)])
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["error"] is None and summary["status"] == "METADATA_COMPLETE"
    result = summary["result"]
    assert (result["observations"], result["parents"], result["frame_references"], result["unique_source_frames"],
            result["visible_fragments"], result["selected_traversals"], result["available_sequences"]) == (180, 10, 900, 390, 360, 40, 360)
    for key in ("model_inference", "checkpoint_reads", "sensor_frames_decoded", "geometry_targets_read", "optimizer_steps", "new_labels"):
        assert result[key] == 0
    assert result["training_ready"] is result["scientific_gate_pass"] is False
    selected = json.loads((run / "artifacts/selected_rows.json").read_text())
    assert len(selected) == 180
    assert result["selection_sha256"] == _sha(run / "artifacts/selected_rows.json")
    opened = json.loads((run / "artifacts/source_reads_sha256.json").read_text())
    assert opened and all("range_m" not in path and "axis_control" not in path for path in opened)
    seal = run / "artifacts/evidence_sha256.txt"
    entries = [line.split(None, 1) for line in seal.read_text().splitlines()]
    assert len(entries) == len([p for p in run.rglob("*") if p.is_file() and p != seal])
    assert all(_sha(tmp_path / path) == digest for digest, path in entries)
    assert all(_sha(path) == digest for path, digest in before.items())
    digest = _sha(seal)
    with pytest.raises(ValueError, match="exact fresh metadata"):
        runner.main()
    assert _sha(seal) == digest
