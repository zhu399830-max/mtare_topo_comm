import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from mtare_topo.data.gse_scoped_inventory import EvidenceStore, ScopedCompositionInventory, validate_selection
from mtare_topo.governance import preflight, validate_data_card
from mtare_topo.governance_inventory import validate_scoped_inventory_card


TASK = "S01_flat_tree_small_C01__c1_mixed"
ROW = {"task": TASK, "row_index": 1, "source_global_sequence_index": 11}


def fixture(tmp_path):
    teacher_root, sidecar_root = tmp_path / "teachers", tmp_path / "sidecars"
    teacher = zarr.open_group(str(teacher_root / (TASK + ".zarr")), mode="w")
    teacher.attrs.update({"parent_id": TASK.split("__")[0], "partition": "fit",
                          "geometry_realization": "c1_mixed", "student_identity_input_forbidden": True})
    arrays = {"frame_row": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
              "source_global_sequence_index": np.array([10, 11]), "primitive_index": np.full((2, 32), -1),
              "primitive_mask": np.zeros((2, 32), dtype="u1"), "support_ray_count": np.zeros((2, 32), dtype="i4"),
              "temporal_visibility": np.zeros((2, 5, 32), dtype="u1"),
              "relative_translation_current_sensor_m": np.zeros((2, 5, 3), dtype="f4")}
    arrays["primitive_index"][:, 0] = 0
    arrays["primitive_mask"][:, 0] = 1
    for key, data in arrays.items():
        teacher.create_dataset(key, data=data)
    sidecar = zarr.open_group(str(sidecar_root / (TASK + ".zarr")), mode="w")
    sidecar.attrs.update({"schema_version": "primitive_attachment_observability_sidecar_v1",
                          "hidden_pair_semantics": "unknown_never_negative", "support_band_m": .25})
    sidecar.create_dataset("source_global_sequence_index", data=np.array([10, 11]))
    packed = np.zeros((2, 8), dtype="u1"); packed[:, 0] = 1
    sidecar.create_dataset("endpoint_observed_packed", data=packed)
    return teacher_root, sidecar_root


def test_only_explicit_task_opened_and_no_glob(tmp_path, monkeypatch):
    roots = fixture(tmp_path)
    for root in roots:
        forbidden = root / TASK.replace("C01", "C10")
        forbidden.mkdir()
    def no_glob(*args, **kwargs):
        raise AssertionError("root-wide task discovery is forbidden")
    monkeypatch.setattr(Path, "glob", no_glob)
    reader = ScopedCompositionInventory(*roots, [ROW])
    result = reader.read_task(TASK)
    assert result["frame_row"].tolist() == [[5, 6, 7, 8, 9]]
    assert result["endpoint_observed"].sum() == 1
    assert reader.opened and all(TASK in p and "C10" not in p for p in reader.opened)


@pytest.mark.parametrize("change", [{"task": "../evil"}, {"task": TASK.replace("C01", "C07")},
                                    {"row_index": -1}, {"row_index": 1.5}, {"source_global_sequence_index": True}])
def test_bad_selection_rejected_before_io(change):
    with pytest.raises(ValueError):
        ScopedCompositionInventory("/does-not-exist", "/also-not", [{**ROW, **change}])


def test_duplicate_rows_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([ROW, ROW])


def test_other_task_request_cannot_open_shard(tmp_path):
    reader = ScopedCompositionInventory(*fixture(tmp_path), [ROW])
    with pytest.raises(PermissionError):
        reader.read_task(TASK.replace("C01", "C02"))
    assert reader.opened == {}


def test_store_refuses_sensor_and_writes(tmp_path):
    root, _ = fixture(tmp_path)
    store = EvidenceStore(root / (TASK + ".zarr"), {"frame_row"}, {})
    for key in ("range_m/0", "../outside", "source_global_sequence_index/0"):
        with pytest.raises(PermissionError):
            store[key]
    with pytest.raises(PermissionError):
        store[".zattrs"] = b"{}"


def test_source_identity_drift_rejected(tmp_path):
    reader = ScopedCompositionInventory(*fixture(tmp_path), [{**ROW, "source_global_sequence_index": 999}])
    with pytest.raises(ValueError, match="identity drift"):
        reader.read_task(TASK)


def test_sealed_bytes_verified_and_missing_seal_rejected(tmp_path):
    roots = fixture(tmp_path)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for root in roots for p in root.rglob("*") if p.is_file()}
    reader = ScopedCompositionInventory(*roots, [ROW], expected_sha256=hashes)
    reader.read_task(TASK)
    reader = ScopedCompositionInventory(*roots, [ROW], expected_sha256={})
    with pytest.raises(ValueError, match="unsealed"):
        reader.read_task(TASK)


def test_symlink_escape_rejected(tmp_path):
    roots = fixture(tmp_path)
    task = TASK.replace("C01", "C02")
    (roots[0] / (task + ".zarr")).symlink_to(tmp_path, target_is_directory=True)
    reader = ScopedCompositionInventory(*roots, [{**ROW, "task": task}])
    with pytest.raises(PermissionError, match="symlink"):
        reader.read_task(task)


def test_missing_sealed_chunk_not_silently_filled_with_zeros(tmp_path):
    roots = fixture(tmp_path)
    chunk = roots[0] / (TASK + ".zarr") / "source_global_sequence_index" / "0"
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for root in roots for p in root.rglob("*") if p.is_file()}
    chunk.unlink()  # only a synthetic temporary pytest fixture
    reader = ScopedCompositionInventory(*roots, [ROW], expected_sha256=hashes)
    with pytest.raises(ValueError, match="sealed chunk missing"):
        reader.read_task(TASK)


def card():
    return {"schema_version": "v3_scoped_inventory_card_v1", "card_id": "fixture", "purpose": "fixture metadata",
            "teacher_source": "sealed synthetic teacher", "sampling_rule": "one fixed row",
            "time_basis": "distance_sampled_no_acquisition_clock", "duration_s": None,
            "license_or_allowed_use": "synthetic fixture", "approval": {"status": "APPROVED", "approved_by": "test",
            "approved_at": "2026-09-05", "scope": "fixture", "confirmation_reference": "test only",
            "authorized_operations": ["audit"], "authorized_gates": [3]},
            "restrictions": {key: True for key in ("read_only_sources", "no_sensor_decoding", "no_model_or_checkpoint",
            "no_optimizer", "no_teacher_generation", "no_calibration", "no_test_worlds", "explicit_task_rows_only")},
            "selected_rows": [ROW], "worlds": [TASK.split("__")[0]], "observation_count": 1,
            "independent_sampling_unit": "topology_parent", "sealed_sources": {"fixture.json": "0" * 64}}


def test_inventory_accepts_honest_untimed_source_not_training_card():
    assert validate_scoped_inventory_card(card()).passed
    assert not validate_data_card(card()).passed


@pytest.mark.parametrize("key,value", [("duration_s", 90), ("time_basis", "made_up"), ("worlds", []),
                                       ("observation_count", 900), ("sealed_sources", {})])
def test_inventory_invalid_contracts_rejected(key, value):
    value_card = card(); value_card[key] = value
    assert not validate_scoped_inventory_card(value_card).passed


def test_inventory_cannot_authorize_training_or_generation():
    for op in ("training", "data_export", "teacher_generation", "threshold_calibration"):
        value = card(); value["approval"]["authorized_operations"] = [op]
        assert not validate_scoped_inventory_card(value).passed


def test_preflight_inventory_dispatch_is_audit_only(tmp_path):
    # Use existing generic spec fixtures; a training request cannot use this
    # schema even if someone changes its authorization operation.
    from tests.v3.unit.test_governance import make_spec, make_status
    path = tmp_path / "card.json"; path.write_text(json.dumps(card()))
    spec = make_spec(3, "audit"); spec["data_card"] = "card.json"
    assert preflight(spec, make_status(3), tmp_path).passed
    spec["operation"] = "training"
    value = card(); value["approval"]["authorized_operations"] = ["training"]
    path.write_text(json.dumps(value))
    assert not preflight(spec, make_status(3), tmp_path).passed
