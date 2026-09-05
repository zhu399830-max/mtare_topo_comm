from __future__ import annotations

import json

import pytest

from execute_gse_spatial_multi_event_teacher_feasibility_v1 import _load_json_object_array


def test_rare_evidence_reader_accepts_the_sealed_four_row_json_array(tmp_path):
    path = tmp_path / "rows.json"
    rows = [{"global_sequence_index": value} for value in (44299, 44300, 44301, 110361)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert _load_json_object_array(path) == rows


def test_rare_evidence_reader_rejects_an_object_root(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="array of objects"):
        _load_json_object_array(path)
