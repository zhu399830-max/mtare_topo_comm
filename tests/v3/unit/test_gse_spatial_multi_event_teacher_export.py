from __future__ import annotations

import sys
import types

import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))

from execute_gse_spatial_multi_event_teacher_export_v1 import (
    EVENT_TYPE_TO_INDEX,
    MAXIMUM_EVENT_TOKENS,
    _write_array,
)


class _Group:
    def __init__(self):
        self.calls = []

    def create_dataset(self, name, **kwargs):
        self.calls.append((name, kwargs))


def test_export_contract_keeps_sixteen_slots_and_two_decision_types():
    assert MAXIMUM_EVENT_TOKENS == 16
    assert EVENT_TYPE_TO_INDEX == {"terminal": 0, "junction": 1}


def test_export_array_writer_preserves_shape_and_bounded_first_chunk():
    group = _Group()
    values = np.zeros((3, 16, 3), dtype=np.float32)
    _write_array(group, "event_relative_xyz_m", values, first_chunk=2048, compressor="frozen")
    name, kwargs = group.calls[0]
    assert name == "event_relative_xyz_m"
    assert kwargs["data"].shape == (3, 16, 3)
    assert kwargs["chunks"] == (3, 16, 3)
    assert kwargs["compressor"] == "frozen"
