"""Lossless Zarr repacking for the primitive-relation sensor corpus.

The corrective changes only the physical compressor.  Logical array names,
shapes, chunks, dtypes, fill values, order, filters, attributes, and every
array element remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass(frozen=True)
class RepackedArrayEvidence:
    name: str
    shape: tuple[int, ...]
    dtype: str
    raw_sha256: str
    chunks_checked: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "raw_sha256": self.raw_sha256,
            "chunks_checked": self.chunks_checked,
        }


def _update_raw_digest(digest: "hashlib._Hash", values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(memoryview(contiguous).cast("B"))


def _first_axis_slices(shape: tuple[int, ...], chunk: int):
    if not shape:
        yield ()
        return
    for start in range(0, shape[0], chunk):
        yield (slice(start, min(start + chunk, shape[0])),) + (slice(None),) * (len(shape) - 1)


def repack_zarr_group_losslessly(
    source_path: Path,
    destination_path: Path,
    *,
    compressor,
) -> tuple[RepackedArrayEvidence, ...]:
    """Repack one flat Zarr group and prove bit-exact logical arrays.

    P1a shards intentionally contain only top-level arrays.  Nested groups are
    rejected so that a later schema change cannot be silently omitted.
    Destination chunks retain the exact source chunk grid; only the compressor
    is replaced.
    """

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    source = zarr.open_group(str(source_path), mode="r")
    if list(source.group_keys()):
        raise ValueError("lossless P1a corrective does not permit nested groups")
    destination = zarr.open_group(str(destination_path), mode="w")
    destination.attrs.update(dict(source.attrs))
    evidence: list[RepackedArrayEvidence] = []
    for name in sorted(source.array_keys()):
        source_array = source[name]
        kwargs = {
            "shape": source_array.shape,
            "chunks": source_array.chunks,
            "dtype": source_array.dtype,
            "compressor": compressor,
            "fill_value": source_array.fill_value,
            "order": source_array.order,
            "filters": source_array.filters,
        }
        dimension_separator = getattr(source_array, "_dimension_separator", None)
        if dimension_separator is not None:
            kwargs["dimension_separator"] = dimension_separator
        destination_array = destination.create_dataset(name, **kwargs)
        destination_array.attrs.update(dict(source_array.attrs))
        source_digest = hashlib.sha256()
        destination_digest = hashlib.sha256()
        chunks_checked = 0
        first_chunk = int(source_array.chunks[0]) if source_array.shape else 1
        for selection in _first_axis_slices(tuple(source_array.shape), first_chunk):
            source_values = np.asarray(source_array[selection])
            destination_array[selection] = source_values
            destination_values = np.asarray(destination_array[selection])
            if source_values.shape != destination_values.shape:
                raise RuntimeError(f"shape drift while repacking {name}")
            if source_values.dtype != destination_values.dtype:
                raise RuntimeError(f"dtype drift while repacking {name}")
            source_bytes = np.ascontiguousarray(source_values).view(np.uint8)
            destination_bytes = np.ascontiguousarray(destination_values).view(np.uint8)
            if not np.array_equal(source_bytes, destination_bytes):
                raise RuntimeError(f"bit drift while repacking {name}")
            _update_raw_digest(source_digest, source_values)
            _update_raw_digest(destination_digest, destination_values)
            chunks_checked += 1
        if source_digest.digest() != destination_digest.digest():
            raise RuntimeError(f"raw digest drift while repacking {name}")
        evidence.append(
            RepackedArrayEvidence(
                name=name,
                shape=tuple(int(value) for value in source_array.shape),
                dtype=source_array.dtype.str,
                raw_sha256=source_digest.hexdigest(),
                chunks_checked=chunks_checked,
            )
        )
    reopened = zarr.open_group(str(destination_path), mode="r")
    if dict(reopened.attrs) != dict(source.attrs):
        raise RuntimeError("group attribute drift after lossless repack")
    if sorted(reopened.array_keys()) != sorted(source.array_keys()):
        raise RuntimeError("array inventory drift after lossless repack")
    return tuple(evidence)


__all__ = ["RepackedArrayEvidence", "repack_zarr_group_losslessly"]
