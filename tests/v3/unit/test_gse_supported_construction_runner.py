"""Pilot lifecycle with synthetic scoped-reader/kernel boundaries.

Real card validation, file I/O, all-row SVGs, aggregation, failure policy and
SHA seals run unchanged. This is not proof of the actual teacher algorithm.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy
import torch
import zarr

from mtare_topo.governance import build_run_id
from mtare_topo.governance_supported_teacher import validate_supported_teacher_card
from tests.v3.unit.test_gse_supported_teacher_card import card


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, issue=None):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_supported_construction_teacher_v1")
    value = card()
    for name in value["sealed_sources"]:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic source metadata")
    value["sealed_sources"] = {name: sha(tmp_path / name) for name in value["sealed_sources"]}
    assert validate_supported_teacher_card(value).passed
    card_path = tmp_path / "configs/card.json"; write(card_path, value)
    source_path = tmp_path / "synthetic_source.json"; write(source_path, {"unchanged": True})
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_supported_construction", "seed": 0,
        "operation": "teacher_generation", "data_card": "configs/card.json", "native_geometry": value["native_geometry"],
        "source_sha256": {"synthetic_source.json": sha(source_path)}, "wall_time_cap_s": 600,
        "expected_versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                              "zarr": zarr.__version__, "scipy": scipy.__version__},
        "expected_counts": {"observations": 180, "parents": 10, "unique_frames": 900, "visible_fragments": 1452}}
    spec_path = tmp_path / "configs/spec.json"; write(spec_path, spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for folder in ("config", "artifacts", "previews", "metrics", "logs"): (run / folder).mkdir(parents=True, exist_ok=True)
    write(run / "config/run_spec.json", spec); write(run / "config/data_card.json", value)
    write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "selected_source_index", lambda root, card: {})
    class FakeReader:
        def __init__(self, root, card, *, expected_sha256):
            assert root == tmp_path and card == value
            self.selection = {task: [r for r in card["selected_rows"] if r["task"] == task] for task in card["tasks"]}
            self.opened = {str(source_path): sha(source_path)}
        def read_task(self, task):
            records = self.selection[task]
            counts = np.array([9 if r["source_global_sequence_index"] < 12 else 8 for r in records])
            mask = np.arange(32)[None] < counts[:, None]
            teacher = {"axis_control_current_sensor_m": np.zeros((18, 32, 3, 3), np.float32),
                       "primitive_mask": mask, "frame_row": np.arange(90, dtype=np.uint64).reshape(18, 5),
                       "source_global_sequence_index": np.array([r["source_global_sequence_index"] for r in records])}
            return {"teacher": teacher, "sensor": {}, "sensor_frame_rows": np.arange(90),
                    "construction": {}, "codebook": {}, "records": records}
    monkeypatch.setattr(runner, "ScopedSupportedTeacherReader", FakeReader)
    def kernel(*, teacher, sensor, sensor_frame_rows, construction, codebook):
        rows = []
        for i, source in enumerate(teacher["source_global_sequence_index"]):
            rows.append({"source_global_sequence_index": int(source),
                "frame_rows": teacher["frame_row"][i].tolist(),
                "visible_fragments": int(teacher["primitive_mask"][i].sum()),
                "nonincident_source_pairs_teacher_only": [], "regions": [
                {"construction_node_id_teacher_only": str(i % 3), "center_current_sensor_m": [float(i), 0., 0.],
                 "center_valid": True, "event_target": "corridor", "event_valid": True,
                 "directional_member_valid": [True] + [False] * 63,
                 "directional_member_target": [1] + [0] * 63,
                 "members": [{"unknown_reasons": []}]}]})
        if issue in ("terminal", "hidden_terminal"):
            rows[0]["regions"][0]["event_target"] = "terminal"
        if issue == "row_identity": rows[0]["source_global_sequence_index"] = -1
        if issue == "late_source_drift": source_path.write_text("changed during generation")
        counts = {"observations": 17 if issue == "population" else 18, "unique_sensor_frames": 90,
                  "visible_fragments": int(teacher["primitive_mask"].sum()), "candidate_region_instances": 18,
                  "center_labels": 18, "member_positive": 18, "member_negative": 0,
                  "observations_with_nonincident_source_ambiguity": 0,
                  "events": {"corridor": 17 if issue == "terminal" else 18, "junction": 0,
                             "terminal": 1 if issue == "terminal" else 0}}
        return {"rows": rows, "counts": counts, "capacity_ready": issue == "capacity_claim",
                "old_teacher_all_fields_reconstructed_exactly": issue != "parity"}
    monkeypatch.setattr(runner, "build_task_targets", kernel)
    monkeypatch.setattr(sys, "argv", ["runner", "--spec", str(spec_path), "--run-dir", str(run)])
    if issue == "early_source_drift": source_path.write_text("changed before pilot")
    return runner, value, spec, run


def verify_seal(tmp_path, run):
    path = run / "artifacts/evidence_sha256.txt"
    assert path.exists()
    for line in path.read_text().splitlines():
        expected, relative = line.split(None, 1)
        assert sha(tmp_path / relative) == expected
    return sha(path)


def test_full_180_lifecycle_all_svg_rows_partial_status_and_seal(tmp_path, monkeypatch):
    runner, value, spec, run = fixture(tmp_path, monkeypatch)
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    result = summary["result"]
    assert summary["error"] is None and summary["status"] == "SUPPORTED_CONSTRUCTION_PARTIAL_TEACHER_GENERATED"
    assert not summary["scientific_gate_pass"] and result["capacity_ready"] is False
    assert (result["observations"], result["parents"], result["unique_sensor_frames"], result["visible_fragments"]) == (180, 10, 900, 1452)
    assert result["events"] == {"corridor": 180, "junction": 0, "terminal": 0}
    assert result["model_inference_frames"] == result["optimizer_steps"] == result["scan_rerenders"] == 0
    rows = json.loads((run / "artifacts/observation_targets.json").read_text())
    assert [(r["task"], r["row_index"]) for r in rows] == [(r["task"], r["row_index"]) for r in value["selected_rows"]]
    for task in value["tasks"]:
        root = ET.parse(run / f"previews/{task}.svg").getroot()
        text = [node.text for node in root.iter() if node.tag.endswith("text") and node.text]
        for record in [r for r in value["selected_rows"] if r["task"] == task]:
            for view in ("XY", "XZ"):
                assert sum(f"row={record['row_index']} {view};" in t for t in text) == 1
    assert len(list((run / "previews").glob("*.svg"))) == 10
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "COMPLETED"
    before = verify_seal(tmp_path, run)
    with pytest.raises(RuntimeError, match="no overwrite"): runner.execute(spec, run)
    assert verify_seal(tmp_path, run) == before


@pytest.mark.parametrize("issue,reason", [
    ("early_source_drift", "frozen source/tool drift"),
    ("late_source_drift", "source/tool changed during teacher generation"),
    ("population", "output counts differ from actual saved target records"),
    ("terminal", "terminal target produced without source cap evidence"),
    ("row_identity", "new label ordering differs from frozen original selection"),
    ("parity", "target/parity population contract drift"),
    ("capacity_claim", "target/parity population contract drift"),
])
def test_failures_sealed_without_promoting_partial_success(tmp_path, monkeypatch, issue, reason):
    runner, value, spec, run = fixture(tmp_path, monkeypatch, issue=issue)
    assert runner.main() == 1
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["error"] is not None and not summary["scientific_gate_pass"]
    assert reason in summary["error"]
    assert summary["status"] == "TEACHER_GENERATION_FAIL"
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert (run / "logs/error.log").exists()
    before = verify_seal(tmp_path, run)
    with pytest.raises(RuntimeError, match="no overwrite"): runner.execute(spec, run)
    assert verify_seal(tmp_path, run) == before


def test_actual_terminal_row_cannot_hide_behind_zero_summary_count(tmp_path, monkeypatch):
    runner, value, spec, run = fixture(tmp_path, monkeypatch, issue="hidden_terminal")
    assert runner.main() == 1
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert "terminal target produced without source cap evidence" in summary["error"]
    verify_seal(tmp_path, run)
