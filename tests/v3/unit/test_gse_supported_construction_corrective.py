import copy
import importlib
import json
from pathlib import Path

import pytest

from tests.v3.unit.test_gse_supported_teacher_card import card


def fixture(tmp_path, monkeypatch, state="FAILED"):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    freezer = importlib.import_module("freeze_gse_supported_construction_teacher_v1r")
    monkeypatch.setattr(freezer, "PROJECT_ROOT", tmp_path)
    prior = tmp_path / freezer.PREVIOUS / "RUN_STATE.json"
    prior.parent.mkdir(parents=True); prior.write_text(json.dumps({"state": state}))
    original = card()
    spec = {"gate": 3, "date": "20260905", "seed": 0, "slug": "old", "source_sha256": {},
            "command": ["env", "python", "runner.py", "--spec", "old.json", "--run-dir", "oldrun"]}
    monkeypatch.setattr(freezer, "original_documents", lambda: (copy.deepcopy(original), copy.deepcopy(spec)))
    monkeypatch.setattr(freezer, "sha", lambda path: "cc21ecd73b20beb730f91edd96209409fd4c9cbc86b55980b01eb8dda7a92f82"
        if str(path).endswith("evidence_sha256.txt") else "f" * 64)
    return freezer, original


def test_corrective_binds_same_population_different_run_and_preserved_source(tmp_path, monkeypatch):
    freezer, original = fixture(tmp_path, monkeypatch)
    new, spec = freezer.documents()
    for key in ("selected_rows", "selection_sha256", "native_geometry", "teacher_fields", "sensor_fields", "restrictions"):
        assert new[key] == original[key]
    assert new["approval"]["scope"] == new["purpose"]
    assert spec["user_authorization"] == new["approval"]
    assert spec["command"][-3] == str(tmp_path / freezer.SPEC)
    assert "v1r_seed0" in spec["command"][-1]
    assert spec["storage_precision_corrective"]["previous_source_git_commit"] == "256a059"
    assert not (tmp_path / freezer.CARD).exists()


def test_corrective_does_not_replace_success_or_running_prior(tmp_path, monkeypatch):
    freezer, _ = fixture(tmp_path, monkeypatch, state="RUNNING")
    with pytest.raises(ValueError, match="preserved failed"):
        freezer.documents()
