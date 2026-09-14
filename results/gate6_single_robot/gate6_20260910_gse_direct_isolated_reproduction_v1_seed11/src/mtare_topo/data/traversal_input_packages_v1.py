"""Storage-only partitioning; never invent graph continuity or alter samples."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import numpy as np
from .continuous_model_input_export_v1 import FIELDS


def package_indices(rows):
    groups = defaultdict(list)
    identities = set()
    for i, row in enumerate(rows):
        task, traversal = row['task'], row['traversal_id']
        if not task or not traversal:
            raise ValueError('explicit task and traversal required')
        identity = (task, row['source_sequence_id'])
        if identity in identities:
            raise ValueError('duplicate source identity')
        identities.add(identity)
        groups[(task, traversal)].append(i)
    result = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda i: rows[i]['sequence_row'])
        if len({rows[i]['sequence_row'] for i in indices}) != len(indices):
            raise ValueError('duplicate source row')
        # A long traversal can span multiple storage packages, without adding
        # any graph segment or discarding the boundary observation.
        result.extend(tuple(indices[j:j + 64]) for j in range(0, len(indices), 64))
    return tuple(result)


def slice_package(value, rows, indices):
    indices = tuple(indices)
    if not 1 <= len(indices) <= 64 or len(set(indices)) != len(indices):
        raise ValueError('one to64 distinct indices required')
    if any(type(i) is not int or i < 0 or i >= len(rows) for i in indices):
        raise ValueError('package index outside source rows')
    selected = [rows[i] for i in indices]
    if len({(r['task'], r['traversal_id']) for r in selected}) != 1:
        raise ValueError('package cannot mix traversals')
    if any(r['task'] != value.task for r in rows):
        raise ValueError('task mismatch')
    if (value.frame_rows.tolist() != [r['frame_rows'] for r in rows]
            or value.source_sequence_ids.tolist() != [r['source_sequence_id'] for r in rows]):
        raise ValueError('original source alignment mismatch')
    arrays = {field: np.asarray(getattr(value, field))[list(indices)].copy() for field in FIELDS}
    report = dict(original_task_read_report=deepcopy(value.read_report),
                  package_source_indices=list(indices), storage_partition_only=True)
    return replace(value, **arrays, read_report=report), deepcopy(selected)
