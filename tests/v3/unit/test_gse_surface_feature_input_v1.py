import hashlib
import io

import numpy as np
import pytest

from mtare_topo.data.gse_surface_feature_input_v1 import bind_feature_input


def fixture():
    return {"ranges_m": np.full((16, 5, 16, 720), 20, dtype=np.float32),
            "valid_mask": np.ones((16, 5, 16, 720), dtype=np.uint8),
            "relative_translation_current_sensor_m": np.zeros((16, 5, 3), dtype=np.float32),
            "relative_yaw_current_sensor_deg": np.zeros((16, 5), dtype=np.float32),
            "frame_rows": np.arange(80, dtype=np.int32).reshape(16, 5),
            "source_sequence_ids": np.arange(16, dtype=np.int64)}


def call(arrays, **overrides):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    args = dict(expected_sha256=hashlib.sha256(raw).hexdigest(), task="S01_flat_tree_small_C01__ellipse",
                row=0, source_sequence_id=0, frame_rows=[0, 1, 2, 3, 4])
    args.update(overrides)
    return bind_feature_input(raw, **args)


def test_preserves_domain_and_legacy_normalization():
    result = call(fixture())
    assert result.range_valid.shape == (5, 2, 16, 720)
    assert np.all(result.range_valid[:, 0] == np.float32(.4))
    assert np.all(result.range_valid[:, 1] == 1)  # All 20 m returns remain valid.
    assert result.input_binding_sha256 == call(fixture()).input_binding_sha256
    other = call(fixture(), row=1, source_sequence_id=1, frame_rows=[5, 6, 7, 8, 9])
    assert result.input_binding_sha256 != other.input_binding_sha256


@pytest.mark.parametrize("change", [dict(row=1), dict(source_sequence_id=1),
    dict(frame_rows=[1, 2, 3, 4, 5]), dict(task="S01_flat_tree_small_C08__ellipse"),
    dict(expected_sha256="0" * 64)])
def test_source_mismatch_rejected(change):
    with pytest.raises(ValueError):
        call(fixture(), **change)


@pytest.mark.parametrize("kind", ["teacher", "dtype", "mask", "motion", "range", "nan"])
def test_invalid_payload_rejected(kind):
    arrays = fixture()
    if kind == "teacher":
        arrays["true_node_id"] = np.ones(16)
    elif kind == "dtype":
        arrays["ranges_m"] = arrays["ranges_m"].astype(np.float64)
    elif kind == "mask":
        arrays["valid_mask"][0, 0, 0, 0] = 2
    elif kind == "motion":
        arrays["relative_translation_current_sensor_m"][0, -1, 0] = 1
    else:
        arrays["ranges_m"][0, 0, 0, 0] = 51 if kind == "range" else float("nan")
    with pytest.raises(ValueError):
        call(arrays)
