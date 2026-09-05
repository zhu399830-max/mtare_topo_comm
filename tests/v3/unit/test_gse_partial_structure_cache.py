"""Synthetic-only sealed-cache joins; no experiment, source data or checkpoint."""
import hashlib
import json

import numpy as np
import pytest

from mtare_topo.data.gse_partial_structure_cache import (
    PREDICTION_KEYS, SOURCE_NAMES, load_partial_structure_cache,
)


def fixture(tmp_path):
    manifest, audit, targets = [], [], []
    axes = np.arange(180 * 32 * 9, dtype=np.float32).reshape(180, 32, 3, 3)
    mask = np.zeros((180, 32), bool)
    for i in range(180):
        task = f"S{i // 18 + 1:02d}_synthetic_C01__c1_mixed"
        count = 9 if i < 12 else 8
        mask[i, :count] = True
        identity = {"task": task, "row_index": (i % 18) * 13,
                    "source_global_sequence_index": i * 7,
                    "frame_rows": list(range((i % 18) * 5, (i % 18) * 5 + 5))}
        manifest.append({"task": task, "row_index": identity["row_index"], "visible_fragments": count})
        audit.append({**identity, "visible_primitive_fragments": count,
                      "target_node_scoring_only": "not-a-model-input", "old_score": 999.})
        targets.append({**identity, "visible_fragments": count, "regions": [], "capacity_ready": False})
    values = {"sample_manifest": manifest, "identity_audit": audit, "partial_targets": targets}
    sources = {}
    for name, basename in SOURCE_NAMES.items():
        path = tmp_path / basename
        if name in values:
            path.write_text(json.dumps(values[name]))
        elif name == "predictions":
            # Unselected branches are deliberately un-decodable with pickle
            # disabled, demonstrating that only final raw coordinates are read.
            np.savez(path, **{key: axes if key == "raw_coordinates" else np.array([object()], dtype=object)
                             for key in PREDICTION_KEYS})
        else:
            np.savez(path, axis_control_m=np.where(mask[..., None, None], axes + 1, 0), mask=mask)
        sources[name] = {"path": basename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return sources, values, axes, mask


def rewrite(tmp_path, sources, name, value):
    path = tmp_path / sources[name]["path"]
    if isinstance(value, dict) and name in ("predictions", "scoring_targets"):
        np.savez(path, **value)
    else:
        path.write_text(json.dumps(value))
    sources[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_cache_all_slots_loss_only_masks_and_ignored_other_branches(tmp_path):
    sources, values, axes, masks = fixture(tmp_path)
    result = load_partial_structure_cache(project_root=tmp_path, sources=sources)
    np.testing.assert_array_equal(result.prediction_axes, axes)
    np.testing.assert_array_equal(result.gt_axes, np.where(masks[..., None, None], axes + 1, 0))
    np.testing.assert_array_equal(result.loss_only_gt_mask, masks)
    assert result.prediction_axes.shape == (180, 32, 3, 3)
    assert np.count_nonzero(result.prediction_axes[~masks]) > 0
    assert len(result.manifest) == len(result.partial_targets) == 180
    assert sum(r["visible_fragments"] for r in result.manifest) == 1452
    assert result.manifest[0]["frame_rows"] == list(range(5))
    assert "target_node_scoring_only" not in result.manifest[0]
    assert "old_score" not in result.manifest[0]
    assert result.read_hashes == {r["path"]: r["sha256"] for r in sources.values()}
    assert result.provenance["cache_embeds_frame_identity"] is False
    assert result.provenance["producer_frame_parity_is_inherited_not_independent_cache_id_check"] is True
    with pytest.raises(ValueError): result.prediction_axes[0, 0, 0, 0] = 0


def test_identity_audit_and_teacher_order_irrelevant_but_prediction_order_preserved(tmp_path):
    sources, values, axes, _ = fixture(tmp_path)
    for name in ("identity_audit", "partial_targets"):
        rewrite(tmp_path, sources, name, values[name][::-1])
    result = load_partial_structure_cache(project_root=tmp_path, sources=sources)
    assert [r["source_global_sequence_index"] for r in result.partial_targets] == [i * 7 for i in range(180)]
    np.testing.assert_array_equal(result.prediction_axes, axes)


@pytest.mark.parametrize("name", list(SOURCE_NAMES))
def test_every_source_hash_verified_before_decode(tmp_path, name):
    sources, _, _, _ = fixture(tmp_path)
    (tmp_path / sources[name]["path"]).write_bytes(b"drifted")
    with pytest.raises(ValueError, match="sealed source drift"):
        load_partial_structure_cache(project_root=tmp_path, sources=sources)


@pytest.mark.parametrize("branch", ["mean_broadcast_coordinates", "raw_coordinates_initial", "legacy_frozen", None])
def test_only_final_raw_branch(tmp_path, branch):
    sources, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="only the frozen final"):
        load_partial_structure_cache(project_root=tmp_path, sources=sources, prediction_branch=branch)


@pytest.mark.parametrize("name,field,value", [
    ("sample_manifest", "task", "S01_synthetic_C02__c1_mixed"),
    ("sample_manifest", "row_index", True),
    ("sample_manifest", "row_index", 0.0),
    ("sample_manifest", "visible_fragments", 9.0),
    ("sample_manifest", "visible_fragments", 8),
    ("sample_manifest", "frame_rows", [5, 6, 7, 8, 9]),
    ("identity_audit", "frame_rows", None),
    ("identity_audit", "frame_rows", [4, 3, 2, 1, 0]),
    ("identity_audit", "frame_rows", [False, 1, 2, 3, 4]),
    ("identity_audit", "source_global_sequence_index", 0.0),
    ("identity_audit", "visible_primitive_fragments", 8),
    ("partial_targets", "source_global_sequence_index", 999999),
    ("partial_targets", "frame_rows", [900, 901, 902, 903, 904]),
    ("partial_targets", "regions", None),
    ("partial_targets", "visible_fragments", 8),
])
def test_identity_type_population_or_teacher_drift(tmp_path, name, field, value):
    sources, values, _, _ = fixture(tmp_path)
    values[name][0][field] = value
    rewrite(tmp_path, sources, name, values[name])
    with pytest.raises(ValueError): load_partial_structure_cache(project_root=tmp_path, sources=sources)


@pytest.mark.parametrize("name", ["sample_manifest", "identity_audit", "partial_targets"])
@pytest.mark.parametrize("change", ["duplicate", "missing", "not_list"])
def test_no_row_guessing_for_incomplete_or_duplicate_sources(tmp_path, name, change):
    sources, values, _, _ = fixture(tmp_path)
    rows = values[name]
    if change == "duplicate": rows[1] = rows[0]
    if change == "missing": rows.pop()
    if change == "not_list": rows = {"rows": rows}
    rewrite(tmp_path, sources, name, rows)
    with pytest.raises(ValueError): load_partial_structure_cache(project_root=tmp_path, sources=sources)


@pytest.mark.parametrize("change", ["nan_prediction", "prediction_shape", "prediction_float64", "missing_branch",
                                    "mask_float", "mask_population", "target_shape", "nan_active_target",
                                    "nan_inactive_target", "nonzero_inactive_target"])
def test_geometry_schema_and_counts(tmp_path, change):
    sources, _, axes, mask = fixture(tmp_path)
    if change in ("nan_prediction", "prediction_shape", "prediction_float64", "missing_branch"):
        prediction = axes.copy()
        if change == "nan_prediction": prediction[0, 31, 0, 0] = np.nan
        if change == "prediction_shape": prediction = prediction[:, :8]
        if change == "prediction_float64": prediction = prediction.astype(np.float64)
        payload = {k: prediction for k in PREDICTION_KEYS}
        if change == "missing_branch": payload.pop("raw_coordinates")
        rewrite(tmp_path, sources, "predictions", payload)
    else:
        axes = np.where(mask[..., None, None], axes + 1, 0)
        if change == "mask_float": mask = mask.astype(np.float32)
        if change == "mask_population": mask[0, 0] = False
        if change == "target_shape": axes = axes[:, :8]
        if change == "nan_active_target": axes[0, 0, 0, 0] = np.nan
        if change == "nan_inactive_target": axes[0, 31, 0, 0] = np.nan
        if change == "nonzero_inactive_target": axes[0, 31, 0, 0] = 1
        rewrite(tmp_path, sources, "scoring_targets", {"axis_control_m": axes, "mask": mask})
    with pytest.raises(ValueError): load_partial_structure_cache(project_root=tmp_path, sources=sources)


def test_identity_bridge_mandatory_not_inferred_from_teacher(tmp_path):
    sources, _, _, _ = fixture(tmp_path)
    sources.pop("identity_audit")
    with pytest.raises(ValueError, match="identity_audit"):
        load_partial_structure_cache(project_root=tmp_path, sources=sources)


def test_changing_gt_mask_slot_layout_does_not_filter_or_reorder_predictions(tmp_path):
    sources, _, axes, mask = fixture(tmp_path)
    rewrite(tmp_path, sources, "scoring_targets", {
        "axis_control_m": np.where(mask[:, ::-1, None, None], axes + 1, 0), "mask": mask[:, ::-1]})
    result = load_partial_structure_cache(project_root=tmp_path, sources=sources)
    np.testing.assert_array_equal(result.prediction_axes, axes)
    np.testing.assert_array_equal(result.loss_only_gt_mask, mask[:, ::-1])


def test_source_path_symlink_escape_denied_before_payload_read(tmp_path):
    sources, _, _, _ = fixture(tmp_path)
    nested = tmp_path / "inside"
    nested.mkdir()
    (nested / "all_predictions.npz").symlink_to(tmp_path / "all_predictions.npz")
    # Other paths need not exist: all declared paths are validated before I/O.
    with pytest.raises(ValueError, match="source escape"):
        load_partial_structure_cache(project_root=nested, sources=sources)


@pytest.mark.parametrize("path", ["../all_predictions.npz", "/all_predictions.npz", "x/../all_predictions.npz"])
def test_path_escape_rejected(tmp_path, path):
    sources, _, _, _ = fixture(tmp_path)
    sources["predictions"]["path"] = path
    with pytest.raises(ValueError, match="sealed source"):
        load_partial_structure_cache(project_root=tmp_path, sources=sources)
