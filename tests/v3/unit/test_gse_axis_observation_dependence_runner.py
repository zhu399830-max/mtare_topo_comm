"""Real cached-output runner on temporary synthetic evidence; no project assets."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from mtare_topo.evaluation.gse_axis_observation_dependence import same_parent_derangement
from mtare_topo.governance import build_run_id
from mtare_topo.governance_inventory import validate_scoped_inventory_card
from mtare_topo.representation.gse_point_axis_loss import axis_set_metrics
from tests.v3.unit.test_gse_scoped_inventory import card as inventory_card


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    wrapper = importlib.import_module("run_gse_axis_observation_dependence_v1r")
    # The wrapper deliberately replaces the original loader. Register its old
    # value so this test leaves the imported module unchanged when it finishes.
    monkeypatch.setattr(wrapper.original, "load_json", wrapper.original.load_json)
    return wrapper


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["sample_manifest.json", "observation_audit.json"])
def test_only_declared_list_artifacts_accept_lists_and_reject_objects(tmp_path, runner, name):
    path = tmp_path / name
    _write(path, [{"synthetic": 1}])
    assert runner.load_declared_json(path) == [{"synthetic": 1}]
    for value in ({"synthetic": 1}, None, "wrong", 1):
        _write(path, value)
        with pytest.raises(ValueError):
            runner.load_declared_json(path)


@pytest.mark.parametrize("name", ["run_spec.json", "observation_metrics.json", "summary.json"])
def test_other_json_remains_object_only(tmp_path, runner, name):
    path = tmp_path / name
    _write(path, {"synthetic": 1})
    assert runner.load_declared_json(path) == {"synthetic": 1}
    _write(path, [{"synthetic": 1}])
    with pytest.raises(ValueError):
        runner.load_declared_json(path)


def test_real_main_restores_scores_plots_and_seals_all180_synthetic_observations(tmp_path, monkeypatch, runner):
    original = runner.original
    base = "synthetic_training"
    manifest, coordinate_rows, old_metrics = [], [], []
    predictions = np.empty((180, 32, 3, 3), dtype=np.float32)
    for index in range(180):
        parent, row_index = divmod(index, 18)
        task = f"S{parent + 1:02d}_fixture_C01__c1_mixed"
        count = 9 if index < 12 else 8
        row = {"task": task, "row_index": row_index,
               "source_global_sequence_index": index, "visible_fragments": count}
        manifest.append(row)
        axis = np.array([[-1., row_index, 0.], [0., row_index, 0.], [1., row_index, 0.]], dtype=np.float32)
        predictions[index] = axis
        controls = [{"teacher_slot_scoring_only": slot, "control_index": control,
                     "query_current_sensor_m": axis[control].tolist()}
                    for slot in range(count) for control in range(3)]
        coordinate_rows.append({"task": task, "row_index": row_index,
                                "coordinate_support": {"control_points": controls}})
        target = np.zeros((1, 32, 3, 3), dtype=np.float32)
        target[:, :count] = axis
        mask = np.arange(32)[None] < count
        old_metrics.append(axis_set_metrics(torch.from_numpy(predictions[index:index + 1]),
                                            torch.from_numpy(target), torch.from_numpy(mask))[0])
    files = {
        f"{base}/artifacts/sample_manifest.json": manifest,
        f"{base}/artifacts/observation_metrics.json": {method: old_metrics for method in original.METHODS},
        "synthetic_coordinate/artifacts/observation_audit.json": coordinate_rows,
    }
    for relative, value in files.items():
        _write(tmp_path / relative, value)
    npz_relative = f"{base}/artifacts/all_predictions.npz"
    np.savez(tmp_path / npz_relative, **{method: predictions for method in original.METHODS})
    sealed = {relative: _sha(tmp_path / relative) for relative in [*files, npz_relative]}
    card = inventory_card()
    card.update(selected_rows=manifest, observation_count=180,
                worlds=sorted({row["task"].split("__")[0] for row in manifest}), sealed_sources=sealed)
    assert validate_scoped_inventory_card(card).passed
    card_relative = "configs/synthetic_card.json"
    _write(tmp_path / card_relative, card)
    source_relative = "synthetic_source.txt"
    (tmp_path / source_relative).write_text("synthetic source, never executed")
    versions = {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "matplotlib", "zarr")}}
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_axis_correspondence", "seed": 0,
            "operation": "audit", "data_card": card_relative, "expected_versions": versions,
            "training_run": base, "coordinate_rows": "synthetic_coordinate/artifacts/observation_audit.json",
            "source_sha256": {source_relative: _sha(tmp_path / source_relative)},
            "shuffled_prediction_indices": same_parent_derangement(manifest)}
    spec_path = tmp_path / "configs/synthetic_spec.json"
    _write(spec_path, spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for directory in ("config", "logs", "metrics", "previews", "artifacts"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    _write(run / "config/run_spec.json", spec)
    _write(run / "config/data_card.json", card)
    _write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    immutable_paths = [tmp_path / path for path in [*sealed, source_relative, card_relative]] + [spec_path]
    before = {path: _sha(path) for path in immutable_paths}

    def contained(relative):
        path = (tmp_path / relative).resolve()
        path.relative_to(tmp_path)
        return path

    monkeypatch.setattr(original, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(original, "contained", contained)
    monkeypatch.setattr(sys, "argv", ["synthetic-runner", "--spec", str(spec_path), "--run-dir", str(run)])
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["error"] is None
    assert summary["status"] == "DIAGNOSTIC_COMPLETE"
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "COMPLETED"
    for key in ("optimizer_steps", "model_inference", "checkpoint_reads", "raw_sensor_frames", "new_labels"):
        assert summary[key] == 0
    assert summary["scientific_gate_pass"] is False
    for method in original.METHODS:
        for condition in ("correct", "shuffled"):
            path = run / f"artifacts/{method}_{condition}.json"
            assert len(json.loads(path.read_text())) == 180
            assert summary["result"][method][condition]["macro"]["matched_targets"] == 1452
        assert summary["result"][method]["comparison"]["parents_correct_better"] == 10
    svg = run / "previews/correspondence_comparison.svg"
    assert ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    assert len((run / "logs/progress.jsonl").read_text().splitlines()) == 8
    seal = run / "artifacts/evidence_sha256.txt"
    entries = [line.split(None, 1) for line in seal.read_text().splitlines()]
    assert len(entries) == len([path for path in run.rglob("*") if path.is_file() and path != seal])
    assert all(_sha(tmp_path / path) == digest for digest, path in entries)
    assert all(_sha(path) == digest for path, digest in before.items())
    seal_before = _sha(seal)
    with pytest.raises(ValueError, match="exact fresh run"):
        runner.main()
    assert _sha(seal) == seal_before
