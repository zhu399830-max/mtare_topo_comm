"""Lossless, exclusive grid artifacts for an authorized population executor.

This module grants no data access: its caller supplies source-bound bundles.
Partial exports remain artifacts, never a completed population manifest.
"""
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .development_reference_grid import build_bound_reference_grid
from mtare_topo.representation.gse_surface_ray_evidence_v1 import (
    ObservedRayGrid, SHAPE, _digest,
)

ARRAYS = ('state', 'free_frame_bits', 'occupied_frame_bits')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_grid(grid):
    if type(grid) is not ObservedRayGrid or grid.physical_connectivity is not False:
        raise ValueError('observation-only typed grid required')
    for name in ARRAYS:
        a = getattr(grid, name)
        if a.shape != SHAPE or a.dtype != np.uint8:
            raise ValueError('grid shape/dtype mismatch')
    if (np.any(grid.free_frame_bits > 31) or np.any(grid.occupied_frame_bits > 31)
            or not np.array_equal(grid.state, np.where(grid.occupied_frame_bits, 2,
                                  np.where(grid.free_frame_bits, 1, 0)))
            or not np.isfinite(grid.numerical_bound_m) or grid.numerical_bound_m < 0
            or _digest(*(getattr(grid, n) for n in ARRAYS),
                       np.asarray([grid.numerical_bound_m], dtype='<f8')) != grid.content_sha256):
        raise ValueError('grid content mismatch')


def encode_grid(grid, binding):
    validate_grid(grid)
    metadata = {f.name: getattr(grid, f.name) for f in fields(grid) if f.name not in ARRAYS}
    header = dict(schema='development_observed_grid_v1', binding=binding, metadata=metadata)
    stream = io.BytesIO()
    np.savez_compressed(stream, **{n: getattr(grid, n) for n in ARRAYS},
                        header=np.frombuffer(canonical(header), dtype=np.uint8))
    return stream.getvalue()


def decode_grid(payload, *, expected_sha256, expected_binding):
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('artifact hash mismatch')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {*ARRAYS, 'header'}:
            raise ValueError('unexpected grid fields')
        header = json.loads(archive['header'].tobytes())
        if (set(header) != {'schema', 'binding', 'metadata'}
                or header['schema'] != 'development_observed_grid_v1'
                or header['binding'] != expected_binding):
            raise ValueError('grid source binding mismatch')
        grid = ObservedRayGrid(**{n: archive[n].copy() for n in ARRAYS}, **header['metadata'])
    validate_grid(grid)
    for n in ARRAYS:
        getattr(grid, n).setflags(write=False)
    return grid


def export_observation(bundle, *, expected_binding, output_dir):
    """Build and preserve one exact observation; existing files are never replaced."""
    directory = Path(output_dir).resolve(strict=True)
    identity = hashlib.sha256(canonical(expected_binding)).hexdigest()
    target = directory / (identity + '.npz')
    # Claim the identity before expensive work. A failed artifact blocks retry.
    with target.open('xb') as stream:
        grid = build_bound_reference_grid(bundle, expected_binding=expected_binding)
        payload = encode_grid(grid, expected_binding)
        digest = hashlib.sha256(payload).hexdigest()
        restored = decode_grid(payload, expected_sha256=digest, expected_binding=expected_binding)
        stream.write(payload)
    return dict(file=target.name, sha256=digest, bytes=len(payload), binding=expected_binding,
                content_sha256=restored.content_sha256,
                source_geometry_sha256=restored.source_geometry_sha256,
                state_counts={str(i): int(np.count_nonzero(restored.state == i)) for i in range(3)},
                whole_region_complete=False, physical_connectivity=False)
