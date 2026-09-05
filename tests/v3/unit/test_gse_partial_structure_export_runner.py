"""Real 180-row cache/bridge/export lifecycle, entirely temporary synthetic IO.

Only the governance validator is replaced: exact artificial fixture counts are
explicit in its spec. Reader, identity joins, Hungarian direction matching,
target transport, NPZ/SVG/log writing and evidence sealing are all real.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy
import torch
import zarr

from mtare_topo.data.gse_partial_structure_cache import PREDICTION_KEYS
from mtare_topo.governance import build_run_id
from tests.v3.unit.test_gse_partial_structure_cache import fixture as cache_fixture, rewrite


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch, *, issue=None):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_partial_structure_export_v1")
    sources, values, _, mask = cache_fixture(tmp_path)
    axes = np.zeros((180, 32, 3, 3), np.float32)
    for slot in range(32):
        axes[:, slot] = [[slot * 10., 0., 0.], [slot * 10., 1., 0.], [slot * 10., 2., 0.]]
    # Deliberately include every surplus prediction, not just active targets.
    rewrite(tmp_path, sources, "predictions", {key: axes if key == "raw_coordinates"
        else np.array([object()], dtype=object) for key in PREDICTION_KEYS})
    rewrite(tmp_path, sources, "scoring_targets", {"axis_control_m": np.where(mask[..., None, None], axes, 0), "mask": mask})
    for index, row in enumerate(values["partial_targets"]):
        row.update(observation_label_complete=False, regions=[{
            "construction_node_id_teacher_only": f"synthetic-{index}",
            "center_current_sensor_m": [5., 0., 0.], "center_valid": True,
            "event_target": "junction" if index % 3 == 0 else "corridor", "event_valid": True,
            "directional_member_target": [1, 0, 0, 0] + [0] * 60,
            "directional_member_valid": [True, False, True, False] + [False] * 60,
        }])
    if issue == "frame_identity":
        values["partial_targets"][0]["frame_rows"] = [901, 902, 903, 904, 905]
    rewrite(tmp_path, sources, "partial_targets", values["partial_targets"])
    source_path = tmp_path / "synthetic_source.json"
    write(source_path, {"fixture": "no source data or checkpoint"})
    card = {"sources": sources, "sealed_sources": {r["path"]: r["sha256"] for r in sources.values()},
            "selected_rows": [{k: row[k] for k in ("task", "row_index", "source_global_sequence_index")}
                              for row in values["identity_audit"]]}
    write(tmp_path / "configs/card.json", card)
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_partial_structure_export", "seed": 0,
        "operation": "data_export", "data_card": "configs/card.json", "wall_time_cap_s": 120,
        "source_sha256": {"synthetic_source.json": sha(source_path)},
        "expected_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__},
        "expected_counts": {"observations": 180, "parents": 10, "unique_frames": 900,
                            "visible_fragments": 1452, "center_labels": 180,
                            "member_positive": 180, "member_negative": 180,
                            "events": {"corridor": 120, "junction": 60, "terminal": 0}}}
    if issue == "expected_counts":
        spec["expected_counts"]["member_positive"] = 179
    if issue == "source_hash":
        source_path.write_text("changed after frozen digest")
    write(tmp_path / "configs/spec.json", spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for directory in ("config", "artifacts", "metrics", "previews", "logs"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    write(run / "config/run_spec.json", spec)
    write(run / "config/data_card.json", card)
    write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "validate_partial_structure_export_card",
                        lambda actual: SimpleNamespace(passed=actual == card, errors=[]))
    return runner, spec, run, axes, mask, sources


def check_seal(tmp_path, run):
    seal = run / "artifacts/evidence_sha256.txt"
    expected = {str(p.relative_to(tmp_path)) for p in run.rglob("*") if p.is_file() and p != seal}
    observed = set()
    for line in seal.read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(tmp_path / name) == digest
        assert name not in observed
        observed.add(name)
    assert observed == expected


def test_real_main_cache_bridge_npz_logs_svg_and_immutable_lifecycle(tmp_path, monkeypatch):
    runner, spec, run, axes, mask, sources = fixture(tmp_path, monkeypatch)
    originals = {p: sha(p) for p in tmp_path.rglob("*") if p.is_file() and not p.is_relative_to(run)}
    monkeypatch.setattr(sys, "argv", ["export", "--spec", str(tmp_path / "configs/spec.json"), "--run-dir", str(run)])
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["error"] is None and summary["scientific_gate_pass"] is False
    assert summary["status"] == "PARTIAL_STRUCTURE_INPUTS_EXPORTED"
    result = summary["result"]
    assert result["original_counts"] == spec["expected_counts"]
    assert result["model_inference"] == result["optimizer_steps"] == result["new_sensor_frames"] == 0
    assert not result["capacity_ready"] and not result["full_three_class_ready"]
    assert result["source_files_read"] == 5
    with np.load(run / "artifacts/partial_training_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["predicted_axes"], axes)
        assert saved["predicted_axes"].shape == (180, 32, 3, 3)
        assert np.count_nonzero(saved["predicted_axes"][~mask]) > 0
        np.testing.assert_array_equal(saved["loss_only_gt_mask"], mask)
        np.testing.assert_array_equal(saved["gt_axes"], np.where(mask[..., None, None], axes, 0))
        for branch in ("gt", "predicted"):
            assert not saved[branch + "__label_complete"].any()
            assert saved[branch + "__member_valid"].sum() == 360
            assert saved[branch + "__center_valid"].sum() == 180
            assert saved[branch + "__event_valid"].sum() == 180
            assert saved[branch + "__teacher_direction"].shape == (180, 64)
            effective = result["effective_counts"][branch]
            assert effective["usable_member_positive"] == effective["usable_member_negative"] == 180
            assert effective["unsupported_transported_members"] == 0
            assert effective["all_labels_incomplete"] is True
        assert saved["predicted__input_direction_valid"].sum() == 180 * 64
        assert saved["gt__input_direction_valid"].sum() == 1452 * 2
    manifest = json.loads((run / "artifacts/manifest.json").read_text())
    records = json.loads((run / "artifacts/target_transport.json").read_text())
    assert len(manifest) == len(records["gt"]) == len(records["predicted"]) == 180
    for branch in ("gt", "predicted"):
        for identity, record in zip(manifest, records[branch]):
            assert all(record[k] == value for k, value in identity.items())
            assert record["known_region_targets"] == record["eligible_region_targets"] == 1
    assert len((run / "logs/export.log").read_text().splitlines()) == 360
    svg = ET.parse(run / "previews/parent_target_transport.svg")
    text = " ".join(svg.getroot().itertext())
    for parent in range(1, 11):
        assert f"S{parent:02d}_synthetic_C01__c1_mixed" in text
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "COMPLETED"
    assert json.loads((run / "artifacts/source_reads_sha256.json").read_text()) == {
        r["path"]: r["sha256"] for r in sources.values()}
    check_seal(tmp_path, run)
    assert all(sha(p) == digest for p, digest in originals.items())
    before_retry = {p: sha(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no overwrite/retry"):
        runner.execute(spec, run)
    assert all(sha(p) == digest for p, digest in before_retry.items())


@pytest.mark.parametrize("issue,expected", [("frame_identity", "frame rows disagree"),
    ("source_hash", "frozen source drift"), ("expected_counts", "original count drift")])
def test_failed_export_preserves_error_seal_and_refuses_retry(tmp_path, monkeypatch, issue, expected):
    runner, spec, run, _, _, _ = fixture(tmp_path, monkeypatch, issue=issue)
    assert runner.execute(spec, run) == 1
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert expected in (run / "logs/error.log").read_text()
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["scientific_gate_pass"] is False and expected in summary["error"]
    assert not (run / "artifacts/partial_training_inputs.npz").exists()
    check_seal(tmp_path, run)
    before = {p: sha(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no overwrite/retry"):
        runner.execute(spec, run)
    assert all(sha(p) == digest for p, digest in before.items())
