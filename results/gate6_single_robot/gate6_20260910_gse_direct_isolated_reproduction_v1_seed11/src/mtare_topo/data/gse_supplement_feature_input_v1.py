"""Exact variable task-size supplemental inputs; original fixed16 binder unchanged."""
from .gse_surface_feature_input_v1 import _bind_feature_input


def bind_feature_input(payload, *, expected_sha256, task, row, source_sequence_id,
                       frame_rows, observation_count):
    # Count supplied by an independently frozen task manifest, never inferred
    # from payload length. Same six student-only arrays and normalization.
    return _bind_feature_input(payload,expected_sha256=expected_sha256,task=task,row=row,
        source_sequence_id=source_sequence_id,frame_rows=frame_rows,
        observation_count=observation_count)
